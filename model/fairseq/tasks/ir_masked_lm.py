# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    iterators,
    IdDataset,
    FairseqDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    SortDataset,
    IRMaskTokensDataset,
    IRPadDataset,
    IRPositionDataset,
    IRDataset,
    IRNumTokensDataset,
    IRLengthFilterDataset,
    IRPairPadDataset,
    SeqOfSeqDataset,
    RandomChoiceMultipleDataset,
    IRMocoValidDataset,
    IRMocoValidPadDataset,
    IRMocoValidIndexDataset,
)
from fairseq.tasks import FairseqTask, register_task


@register_task('ir_masked_lm')
class IRMaskedLMTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--function-length', default=255, type=int)
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--moco-loss-coeff', type=float, help='MoCo loss term coefficient')
        parser.add_argument('--mlm-loss-coeff', type=float, help='MLM loss term coefficient')
        parser.add_argument('--augmented-variants', type=int, help='Number of augmented variants')
        parser.add_argument('--moco-valid-size', type=int, help='Number of sentences used to validate MoCo')
        parser.add_argument('--no-state', action='store_true')

    def __init__(self, args, instruction_dictionary, state_dictionary):
        super().__init__(args)
        self.instruction_dictionary = instruction_dictionary
        self.state_dictionary = state_dictionary
        self.seed = args.seed
        if not hasattr(args, 'no_state'):
            args.no_state = False
        if not hasattr(args, 'no_pce'):
            args.no_pce = False

        # add mask token
        self.inst_mask_idx = instruction_dictionary.add_symbol('<mask>')
        self.state_mask_idx = state_dictionary.add_symbol('<mask>')
        instruction_dictionary.add_symbol('<t>')
        state_dictionary.add_symbol('<t>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(':')
        assert len(paths) > 0
        instruction_dictionary = Dictionary.load(os.path.join(paths[0], 'inst_dict.txt'))
        state_dictionary = Dictionary.load(os.path.join(paths[0], 'state_dict.txt'))
        print('| instruction dictionary: {} types'.format(len(instruction_dictionary)))
        print('| state dictionary: {} types'.format(len(state_dictionary)))
        return cls(args, instruction_dictionary, state_dictionary)

    def load_dataset(self, split, epoch=0, combine=False, data_selector=None):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        print('Loading dataset')
        
        src_datasets = []
        tgt_datasets = []
        
        for i in range(self.args.augmented_variants):
            data_path = os.path.join(self.args.data, str(i))
            dataset_inst = data_utils.load_indexed_dataset(
                os.path.join(data_path, 'insts', split),
                self.instruction_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            
            dataset_state = data_utils.load_indexed_dataset(
                os.path.join(data_path, 'states', split),
                self.state_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            
            if dataset_inst is None or dataset_state is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))
    
            dataset_inst = SeqOfSeqDataset(dataset_inst, self.instruction_dictionary)
            dataset_state = SeqOfSeqDataset(dataset_state, self.state_dictionary)
            dataset_pos = IRPositionDataset(os.path.join(data_path, 'pos', split))
            dataset = IRDataset(dataset_inst, dataset_state, dataset_pos)
            
            block_size = self.args.function_length
    
            print('| loaded {} batches from: {} and {}'.format(len(dataset),
                os.path.join(data_path, 'insts', split), os.path.join(data_path, 'states', split)))

            src_dataset, tgt_dataset = IRMaskTokensDataset.apply_mask(
                dataset,
                self.instruction_dictionary,
                self.state_dictionary,
                inst_pad_idx=self.instruction_dictionary.pad(),
                state_pad_idx=self.state_dictionary.pad(),
                inst_mask_idx=self.inst_mask_idx,
                state_mask_idx=self.state_mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
            )
            
            src_datasets.append(src_dataset)
            tgt_datasets.append(tgt_dataset) 

        src_dataset = RandomChoiceMultipleDataset(src_datasets, seed=self.args.seed)
        tgt_dataset = RandomChoiceMultipleDataset(tgt_datasets, seed=self.args.seed, only_first=True)

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src': IRPairPadDataset(
                                src_dataset,
                                inst_pad_idx=self.instruction_dictionary.pad(),
                                state_pad_idx=self.state_dictionary.pad(),
                                inst_mask_idx=self.inst_mask_idx,
                                state_mask_idx=self.state_mask_idx,
                                inst_cls_idx=self.instruction_dictionary.index('<t>'),
                                state_cls_idx=self.state_dictionary.index('<t>'),
                                smallbert_insts_per_input=self.args.smallbert_insts_per_group,
                                smallbert_states_per_input=self.args.smallbert_insts_per_group,
                                max_length=block_size,
                                inst_pad_length=32,
                                state_pad_length=16,
                            )
                    },
                    'target': IRPadDataset(
                        tgt_dataset,
                        inst_pad_idx=self.instruction_dictionary.pad(),
                        state_pad_idx=self.state_dictionary.pad(),
                        inst_mask_idx=self.inst_mask_idx,
                        state_mask_idx=self.state_mask_idx,
                        inst_cls_idx=self.instruction_dictionary.index('<t>'),
                        state_cls_idx=self.state_dictionary.index('<t>'),
                        smallbert_insts_per_input=self.args.smallbert_insts_per_group,
                        smallbert_states_per_input=self.args.smallbert_insts_per_group,
                        max_length=block_size,
                        inst_pad_length=32,
                        state_pad_length=16,
                    ),
                },
                sizes=[src_dataset.sizes[:, 0]],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes[:, 0],
            ],
        )

    def update_step(self, num_updates, model=None):
        if model is not None:
            model.decoder.moco_head.update()

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, train=False)
        return loss, sample_size, logging_output

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        
        if args.no_state:
            model.remove_state()

        return model

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        return None
    
    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None