# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import json

import torch
import torch.distributed as dist
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('ir_prediction')
class IRPredictionCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.task = task
        self.args = args
        
    @staticmethod
    def add_args(parser):
        parser.add_argument('--save-predictions', type=str, default='')

    def forward(self, model, sample, reduce=True, train=True):
        src = sample['net_input']['src']
        output = model(src=src, masked_tokens=None, features_only=True, moco_head=False,
            moco_head_only_proj=False, lm_head=False, classification_head_name='classification_head', classification_head_pooling_indices=None, has_state=not self.args.no_state)
        logits = output[2]
        targets = sample['target']
        indices = sample['indices']
        subset = sample['subset']
        sample_size = targets.size(0)
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='sum',
        )
        if self.args.num_classes == 2:
            tp = ((logits[:, 0] <= logits[:, 1]) & (targets == 1)).long().sum()
            fp = ((logits[:, 0] <= logits[:, 1]) & (targets == 0)).long().sum()
            fn = ((logits[:, 0] > logits[:, 1]) & (targets == 1)).long().sum()
            tn = ((logits[:, 0] > logits[:, 1]) & (targets == 0)).long().sum()
            assert (tp + fp + tn + fn) == targets.size(0), 'invalid size'

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = logits.max(dim=1)[1]
        
        ncorrect = (preds == targets).sum().item()
        logging_output.update(
            ncorrect=ncorrect
        )
        if self.args.num_classes == 2:
            logging_output.update(tp=utils.item(tp.data) if reduce else tp.data)
            logging_output.update(fp=utils.item(fp.data) if reduce else fp.data)
            logging_output.update(fn=utils.item(fn.data) if reduce else fn.data)
            logging_output.update(tn=utils.item(tn.data) if reduce else tn.data)
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': 1,
            'ncorrect': ncorrect,
            'nsentences': nsentences,
            'classification_nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect / nsentences)

            tp_sum = sum(log.get('tp', 0) for log in logging_outputs)
            fp_sum = sum(log.get('fp', 0) for log in logging_outputs)
            fn_sum = sum(log.get('fn', 0) for log in logging_outputs)
            tn_sum = sum(log.get('tn', 0) for log in logging_outputs)
            if tp_sum + fp_sum + fn_sum + tn_sum > 0:
                assert tp_sum + fp_sum + fn_sum + tn_sum == sample_size, 'invalid size when aggregating'
                acc = (tp_sum + tn_sum) / sample_size
                tmp = 2 * tp_sum + fp_sum + fn_sum
                f1 = (2 * tp_sum) / tmp if tmp else 0
                tmp = (tp_sum + fp_sum) * (tp_sum + fn_sum) * (tn_sum + fp_sum) * (tn_sum + fn_sum)
                mcc = (tp_sum * tn_sum - fp_sum * fn_sum) / (tmp ** 0.5) if tmp else 0
                agg_output.update(f1=f1)
                agg_output.update(mcc=mcc)
                agg_output.update(acc_f1=0.5 * (acc + f1))

        return agg_output