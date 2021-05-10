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

def compute_loss_with_global_feature(feature, loss_func, enable_grad=True):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    shape = (world_size, ) + feature.shape
    all_feature = torch.zeros(shape, device=feature.device, dtype=feature.dtype)
    all_feature[rank] = feature.detach().clone()
    dist.all_reduce(all_feature)
    all_feature[rank] = feature
    loss = loss_func(all_feature)
    return loss

@register_criterion('poj_similarity')
class PojSimilarityLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.inst_padding_idx = task.instruction_dictionary.pad()
        self.state_padding_idx = task.state_dictionary.pad()
        self.task = task
        self.args = args

    def forward(self, model, sample, reduce=True, train=True):
        no_state = self.args.no_state
        no_pce = self.args.no_pce
        pooling = self.args.use_pooling
        output = model(**sample['net_input'], masked_tokens=None, features_only=True, moco_head=False,
            moco_head_only_proj=False, lm_head=False, classification_head_name=None,
            has_state=not no_state, has_pce=not no_pce, pooling_instruction=pooling)
        feature = output[0][2]
        feature = feature.float().view(-1, 3, feature.size(-1))
        labels = sample['label'].view(-1, 3)[:, 0]
        m = dist.get_world_size()
        n = feature.size(0)
        r = dist.get_rank()
        all_labels = torch.cuda.LongTensor(m, n).fill_(0)
        all_labels[r, :] = labels
        dist.all_reduce(all_labels)
        all_labels = all_labels.view(m * n)
        dim = feature.size(-1)
        sqrtd = math.sqrt(dim)
        def loss_func(features):
            x = features[:, :, 0, :]
            xp = features[:, :, 1, :]
            xn = features[:, :, 2, :]
            x = x.reshape(-1, x.size(-1))
            xp = xp.reshape(-1, xp.size(-1))
            xn = xn.reshape(-1, xn.size(-1))
            prob_1 = (x * xp).sum(-1)
            prob_2 = (x * xn).sum(-1)
            temp = torch.cat((x, xp), 0)
            temp_labels = torch.cat((all_labels, all_labels), 0)
            prob_3 = torch.mm(x, temp.t())
            prob_1 = prob_1 / sqrtd
            prob_2 = prob_2 / sqrtd
            prob_3 = prob_3 / sqrtd
            mask = all_labels[:, None] == temp_labels[None, :]
            
            prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()
            prob = torch.softmax(torch.cat((prob_1[:, None], prob_2[:, None], prob_3), -1), -1)
            ncorrect = (torch.argmax(prob, dim=1) == 0).sum().item()
            loss = torch.log(prob[:, 0] + 1e-10)
            loss = -loss.mean() * m
            prob_3[mask] = float('-inf')
            return loss, ncorrect
        loss, ncorrect = compute_loss_with_global_feature(feature, loss_func, train)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nsentences': n,
            'ncorrect': ncorrect,
            'sample_size': n,
        }
        return loss, 1, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        m = dist.get_world_size()
        agg_output = {
            'loss': loss / m / m / math.log(2),
            'ntokens': 1,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'acc': ncorrect / nsentences / m,
        }
        return agg_output
