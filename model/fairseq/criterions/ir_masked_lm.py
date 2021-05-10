# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('ir_masked_lm')
class IRMaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.inst_padding_idx = task.instruction_dictionary.pad()
        self.state_padding_idx = task.state_dictionary.pad()
        self.task = task
        self.args = args

    def forward(self, model, sample, reduce=True, train=True):
        return self.forward_train_or_mlm_valid(model, sample, reduce, train)

    def forward_train_or_mlm_valid(self, model, sample, reduce=True, train=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_inst_tokens = sample['target'][0].ne(self.inst_padding_idx)
        masked_state_tokens = sample['target'][1].ne(self.state_padding_idx)
        sample_size = masked_inst_tokens.int().sum().item() + masked_state_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None
        
        has_state = not self.args.no_state

        output = model(**sample['net_input'], masked_tokens=(masked_inst_tokens, masked_state_tokens),
            moco_head=train, has_state=has_state)[0]
        targets = model.get_targets(sample, [output])
        
        if sample_size != 0:
            targets_inst = targets[0][masked_inst_tokens]
        else:
            targets_inst = targets[0]

        targets_indices = targets[6]

        mlm_loss = F.nll_loss(
            F.log_softmax(
                output[0].view(-1, output[0].size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets_inst.view(-1).cuda(),
            reduction='sum',
            ignore_index=self.inst_padding_idx,
        )
        
        if has_state:
            if sample_size != 0:
                targets_state = targets[1][masked_state_tokens]
            else:
                targets_state = targets[1]
            mlm_loss = mlm_loss + F.nll_loss(
                F.log_softmax(
                    output[1].view(-1, output[1].size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets_state.view(-1).cuda(),
                reduction='sum',
                ignore_index=self.state_padding_idx,
            )
        
        loss = mlm_loss
        if train and output[2] is not None:
            moco_labels = output[2][1].view(-1).cuda()
            moco_logits = output[2][0].view(-1, output[2][0].size(-1))
            moco_softmax = F.log_softmax(
                moco_logits,
                dim=-1,
                dtype=torch.float32,
            )
            moco_loss = F.nll_loss(
                moco_softmax,
                moco_labels,
                reduction='sum',
            )
            moco_ncorrect = (moco_logits.max(1).indices == moco_labels).sum().item()
            loss = self.args.mlm_loss_coeff * loss + self.args.moco_loss_coeff * moco_loss
        else:
            moco_loss = torch.zeros(loss.shape)
            moco_ncorrect = 0
            
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(mlm_loss.data) if reduce else mlm_loss.data,
            'mlm_loss': utils.item(mlm_loss.data) if reduce else mlm_loss.data,
            'moco_loss': utils.item(moco_loss.data) if reduce else moco_loss.data,
            'moco_ncorrect': moco_ncorrect,
            'nsentences': len(sample['target'][2]),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss = sum(log.get('nll_loss', 0) for log in logging_outputs)
        mlm_loss = sum(log.get('mlm_loss', 0) for log in logging_outputs)
        moco_loss = sum(log.get('moco_loss', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        moco_ncorrect = sum(log.get('moco_ncorrect', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss / nsentences / math.log(2),
            'nll_loss': nll_loss / sample_size / math.log(2),
            'mlm_loss': mlm_loss / sample_size / math.log(2),
            'mlm_loss2': mlm_loss / len(logging_outputs),
            'moco_loss': moco_loss / nsentences / math.log(2),
            'moco_loss2': moco_loss / len(logging_outputs),
            'moco_acc': moco_ncorrect / nsentences,
            'ntokens': 1,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
