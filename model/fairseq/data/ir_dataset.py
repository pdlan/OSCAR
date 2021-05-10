import numpy as np
import torch
import json
import time
import math
import itertools
from functools import lru_cache

from fairseq.data import FairseqDataset, data_utils, Dictionary

from . import BaseWrapperDataset, LRUCacheDataset

class SeqOfSeqDataset(FairseqDataset):
    def __init__(self, dataset, dictionary):
        super().__init__()
        self.dataset = dataset
        self.dictionary = dictionary

    def __getitem__(self, idx):
        delimiter = self.dictionary.index('</s>')
        padding = self.dictionary.index('<pad>')
        array1d = self.dataset[idx]
        delimiter_idx = torch.nonzero(array1d == delimiter, as_tuple=False)
        seqs = []

        for i in range(len(delimiter_idx) + 1):
            start = 0 if i == 0 else delimiter_idx[i - 1] + 1
            end = len(array1d) if i == len(delimiter_idx) else delimiter_idx[i]
            if end == start:
                continue
            seqs.append(torch.cat((torch.tensor([self.dictionary.index('<t>')], dtype=torch.long), array1d[start:end])))

        return seqs

    def __len__(self):
        return len(self.dataset)

class IRPositionDataset(BaseWrapperDataset):
    def __init__(self, path):
        self.dataset = data_utils.load_indexed_dataset(path, None, 'mmap', combine=None)

    def __getitem__(self, idx):
        return self.dataset[idx].view(3, -1)

class IRDataset(FairseqDataset):
    def __init__(self, inst_dataset, state_dataset, pos_dataset):
        super().__init__()
        self.inst_dataset = inst_dataset
        self.state_dataset = state_dataset
        self.pos_dataset = pos_dataset
        self.sizes = pos_dataset.sizes / 3

    def __getitem__(self, idx):
        return self.inst_dataset[idx], self.state_dataset[idx] if self.state_dataset is not None else None, self.pos_dataset[idx], idx

    def __len__(self):
        return len(self.pos_dataset)

    def size(self, index):
        return len(self.pos_dataset)

class IRLengthFilterDataset(FairseqDataset):
    def __init__(self, dataset, min_len, max_len):
        self.dataset = dataset
        self.min_len = min_len
        self.max_len = max_len
        indexes = []
        sizes = []
        for i, length in enumerate(dataset.sizes):
            print(length)
            if length >= min_len and length <= max_len:
                indexes.append(i)
                sizes.append(length)
        self.indexes = torch.tensor(indexes, dtype=torch.long)
        self.sizes = torch.tensor(sizes, dtype=torch.long)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]

    def __len__(self):
        return len(self.indexes)

class IRMaskTokensDataset(BaseWrapperDataset):

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        inst_dict: Dictionary,
        state_dict: Dictionary,
        inst_pad_idx: int,
        state_pad_idx: int,
        inst_mask_idx: int,
        state_mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.inst_dict = inst_dict
        self.state_dict = state_dict
        self.inst_pad_idx = inst_pad_idx
        self.state_pad_idx = state_pad_idx
        self.inst_mask_idx = inst_mask_idx
        self.state_mask_idx = state_mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                inst_weights = np.array(self.inst_dict.count)
                state_weights = np.array(self.state_dict.count)
            else:
                inst_weights = np.ones(len(self.inst_dict))
                state_weights = np.ones(len(self.state_dict))
            inst_weights[:self.inst_dict.nspecial] = 0
            state_weights[:self.state_dict.nspecial] = 0
            self.inst_weights = inst_weights / inst_weights.sum()
            self.state_weights = state_weights / state_weights.sum()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def generate_mask(self, mask, sz):
        # decide unmasking and random replacement
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
            if self.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(sz) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask
        return mask, rand_mask

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item[2][0])

            # decide elements to mask
            inst_mask = np.full(sz, False)
            state_mask = np.full(sz, False)
            
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz + np.random.rand()
            )
            if num_mask >= sz:
                num_mask -= 1
            mask_index = np.random.choice(sz, num_mask * 2, replace=False)
            mask_index_inst = mask_index[:num_mask]
            mask_index_state = mask_index[num_mask:]
            inst_mask[mask_index_inst] = True
            state_mask[mask_index_state] = True

            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                new_item_inst = [torch.from_numpy(np.full(len(x), self.inst_pad_idx))
                    for x in item[0]]
                new_item_state = [torch.from_numpy(np.full(len(x), self.state_pad_idx))
                    for x in item[0]]
                for i in range(sz):
                    if inst_mask[i]:
                        new_item_inst[i] = item[0][i]
                    if state_mask[i]:
                        new_item_state[i] = item[1][i]
                return (new_item_inst, new_item_state, item[2], index)

            mask_inst, rand_mask_inst = self.generate_mask(inst_mask, sz)
            mask_state, rand_mask_state = self.generate_mask(state_mask, sz)

            new_item_inst = []
            new_item_state = []
            for i in range(sz):
                if inst_mask[i]:
                    new_item_inst.append(torch.from_numpy(np.full(len(item[0][i]), self.inst_mask_idx)))
                else:
                    new_item_inst.append(item[0][i])
                if state_mask[i]:
                    new_item_state.append(torch.from_numpy(np.full(len(item[1][i]), self.state_mask_idx)))
                else:
                    new_item_state.append(item[1][i])
            if rand_mask_inst is not None:
                num_rand = rand_mask_inst.sum()
                if num_rand > 0:
                    for i in range(sz):
                        if rand_mask_inst[i]:
                            new_item_inst[i] = torch.from_numpy(np.random.choice(len(self.inst_dict),
                                    len(item[0][i]), p=self.inst_weights))
            if rand_mask_state is not None:
                num_rand = rand_mask_state.sum()
                if num_rand > 0:
                    for i in range(sz):
                        if rand_mask_state[i]:
                            new_item_state[i] = torch.from_numpy(np.random.choice(len(self.state_dict),
                                    len(item[1][i]), p=self.state_weights))

            return new_item_inst, new_item_state, item[2], index

class IRPadDataset(BaseWrapperDataset):
    def __init__(self, dataset, inst_pad_idx, state_pad_idx, inst_mask_idx, state_mask_idx,
        inst_cls_idx, state_cls_idx, smallbert_insts_per_input, smallbert_states_per_input, max_length, inst_pad_length, state_pad_length, pair=False):
        super().__init__(dataset)
        self.inst_pad_idx = inst_pad_idx
        self.state_pad_idx = state_pad_idx
        self.inst_mask_idx = inst_mask_idx
        self.state_mask_idx = state_mask_idx
        self.inst_cls_idx = inst_cls_idx
        self.state_cls_idx = state_cls_idx
        self.smallbert_insts_per_input = smallbert_insts_per_input
        self.smallbert_states_per_input = smallbert_states_per_input
        self.max_length = max_length
        self.inst_pad_length = inst_pad_length
        self.state_pad_length = state_pad_length
        self.pair = pair

    def collater(self, samples, pair=False):
        li = self.smallbert_insts_per_input
        ls = self.smallbert_states_per_input
        small_bert_length = li * ls // math.gcd(li, ls)
        length = max(len(s[0]) for s in samples)
        length = (length + small_bert_length - 1) // small_bert_length * small_bert_length
        res_inst = torch.full((len(samples), length, self.inst_pad_length), self.inst_pad_idx, dtype=torch.long)
        res_state = torch.full((len(samples), length, self.state_pad_length), self.state_pad_idx, dtype=torch.long)
        res_inst[:, :, 0] = self.inst_cls_idx
        res_state[:, :, 0] = self.state_cls_idx
        res_cpos = torch.full((len(samples), length), -1, dtype=torch.long)
        res_tpos = torch.full((len(samples), length), -1, dtype=torch.long)
        res_fpos = torch.full((len(samples), length), -1, dtype=torch.long)
        padding_mask = torch.ones((len(samples), length), dtype=torch.long)
        
        for i, v in enumerate(samples):
            l = len(v[0])
            for j in range(l):
                inst_tokens = v[0][j]
                res_inst[i, j, :min(len(inst_tokens), self.inst_pad_length)].copy_(inst_tokens[:min(len(inst_tokens), self.inst_pad_length)])
                state_tokens = v[1][j]
                res_state[i, j, :min(len(state_tokens), self.state_pad_length)].copy_(state_tokens[:min(len(state_tokens), self.state_pad_length)])
            res_cpos[i, :l] = v[2][0]
            res_tpos[i, :l] = v[2][1]
            res_fpos[i, :l] = v[2][2]
            padding_mask[i, :l] = 0
        data = res_inst, res_state, res_cpos, res_tpos, res_fpos, padding_mask, torch.tensor([s[3] for s in samples], dtype=torch.long)
        if pair or self.pair:
            return data, None
        else:
            return data

class RandomChoiceMultipleDataset(FairseqDataset):
    def __init__(self, datasets, seed=1, only_first=False):
        self.num_datasets = len(datasets)
        self.datasets = datasets
        self.seed = seed
        self.epoch = 0
        self.length = len(self.datasets[0])
        self.only_first = only_first
        self.sizes = torch.zeros((self.length, 2), dtype=torch.long)
        self.index = torch.zeros((self.length, 2), dtype=torch.long)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch
        with data_utils.numpy_seed(self.seed, self.epoch):
            for i in range(self.length):
                idx1, idx2 = np.random.choice(self.num_datasets, 2, replace=False)
                self.index[i][0] = idx1
                self.index[i][1] = idx2
                self.sizes[i][0] = self.datasets[idx1].sizes[i]
                self.sizes[i][1] = self.datasets[idx2].sizes[i]

    def __getitem__(self, idx):
        if self.only_first:
            return self.datasets[self.index[idx][0]][idx]
        else:
            return (self.datasets[self.index[idx][0]][idx], self.datasets[self.index[idx][1]][idx])

    def __len__(self):
        return self.length

class IRPairPadDataset(BaseWrapperDataset):
    def __init__(self, dataset, inst_pad_idx, state_pad_idx, inst_mask_idx, state_mask_idx,
        inst_cls_idx, state_cls_idx, smallbert_insts_per_input, smallbert_states_per_input, max_length, inst_pad_length, state_pad_length):
        super().__init__(dataset)
        self.inst_pad_idx = inst_pad_idx
        self.state_pad_idx = state_pad_idx
        self.inst_mask_idx = inst_mask_idx
        self.state_mask_idx = state_mask_idx
        self.inst_cls_idx = inst_cls_idx
        self.state_cls_idx = state_cls_idx
        self.smallbert_insts_per_input = smallbert_insts_per_input
        self.smallbert_states_per_input = smallbert_states_per_input
        self.max_length = max_length
        self.inst_pad_length = inst_pad_length
        self.state_pad_length = state_pad_length

    @property
    def sizes(self):
        return self.dataset.sizes[:, 0]

    def collater(self, samples):
        s1 = [s[0] for s in samples]
        s2 = [s[1] for s in samples]
        return self.collater_(s1), self.collater_(s2)

    def collater_(self, samples):
        li = self.smallbert_insts_per_input
        ls = self.smallbert_states_per_input
        small_bert_length = li * ls // math.gcd(li, ls)
        length = max(len(s[0]) for s in samples)
        length = (length + small_bert_length - 1) // small_bert_length * small_bert_length
        res_inst = torch.full((len(samples), length, self.inst_pad_length), self.inst_pad_idx, dtype=torch.long)
        res_state = torch.full((len(samples), length, self.state_pad_length), self.state_pad_idx, dtype=torch.long)
        res_inst[:, :, 0] = self.inst_cls_idx
        res_state[:, :, 0] = self.state_cls_idx
        res_cpos = torch.full((len(samples), length), -1, dtype=torch.long)
        res_tpos = torch.full((len(samples), length), -1, dtype=torch.long)
        res_fpos = torch.full((len(samples), length), -1, dtype=torch.long)
        padding_mask = torch.ones((len(samples), length), dtype=torch.long)
        
        for i, v in enumerate(samples):
            l = len(v[0])
            for j in range(l):
                inst_tokens = v[0][j]
                res_inst[i, j, :min(len(inst_tokens), self.inst_pad_length)].copy_(inst_tokens[:min(len(inst_tokens), self.inst_pad_length)])
                state_tokens = v[1][j]
                res_state[i, j, :min(len(state_tokens), self.state_pad_length)].copy_(state_tokens[:min(len(state_tokens), self.state_pad_length)])
            res_cpos[i, :l] = v[2][0]
            res_tpos[i, :l] = v[2][1]
            res_fpos[i, :l] = v[2][2]
            padding_mask[i, :l] = 0
        return res_inst, res_state, res_cpos, res_tpos, res_fpos, padding_mask, torch.tensor([s[3] for s in samples], dtype=torch.long)

class IRNumTokensDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        return sum(len(x) for x in self.dataset[idx][0]) + sum(len(x) for x in self.dataset[idx][1])

    def collater(self, samples):
        return sum(samples)

class IRMocoValidPadDataset(BaseWrapperDataset):
    def __init__(self, dataset, inst_pad_idx, state_pad_idx, inst_mask_idx, state_mask_idx,
        inst_cls_idx, state_cls_idx, smallbert_insts_per_input, smallbert_states_per_input, max_length, inst_pad_length, state_pad_length):
        super().__init__(dataset)
        self.inst_pad_idx = inst_pad_idx
        self.state_pad_idx = state_pad_idx
        self.inst_mask_idx = inst_mask_idx
        self.state_mask_idx = state_mask_idx
        self.inst_cls_idx = inst_cls_idx
        self.state_cls_idx = state_cls_idx
        self.smallbert_insts_per_input = smallbert_insts_per_input
        self.smallbert_states_per_input = smallbert_states_per_input
        self.max_length = max_length
        self.inst_pad_length = inst_pad_length
        self.state_pad_length = state_pad_length

    def collater(self, samples):
        li = self.smallbert_insts_per_input
        ls = self.smallbert_states_per_input
        small_bert_length = li * ls // math.gcd(li, ls)
        length = max(len(s[0][0]) for s in samples)
        length = (length + small_bert_length - 1) // small_bert_length * small_bert_length
        res_inst = torch.full((len(samples), length, self.inst_pad_length), self.inst_pad_idx, dtype=torch.long)
        res_state = torch.full((len(samples), length, self.state_pad_length), self.state_pad_idx, dtype=torch.long)
        res_inst[:, :, 0] = self.inst_cls_idx
        res_state[:, :, 0] = self.state_cls_idx
        res_cpos = torch.full((len(samples), length), -1, dtype=torch.long)
        res_tpos = torch.full((len(samples), length), -1, dtype=torch.long)
        res_fpos = torch.full((len(samples), length), -1, dtype=torch.long)
        padding_mask = torch.ones((len(samples), length), dtype=torch.long)
        
        for i, s in enumerate(samples):
            v = s[0]
            l = len(v[0])
            for j in range(l):
                inst_tokens = v[0][j]
                res_inst[i, j, :min(len(inst_tokens), self.inst_pad_length)].copy_(inst_tokens[:min(len(inst_tokens), self.inst_pad_length)])
                state_tokens = v[1][j]
                res_state[i, j, :min(len(state_tokens), self.state_pad_length)].copy_(state_tokens[:min(len(state_tokens), self.state_pad_length)])
            res_cpos[i, :l] = v[2][0]
            res_tpos[i, :l] = v[2][1]
            res_fpos[i, :l] = v[2][2]
            padding_mask[i, :l] = 0
        return (res_inst, res_state, res_cpos, res_tpos, res_fpos, padding_mask, torch.tensor([s[0][3] for s in samples], dtype=torch.long)), None

class IRMocoValidDataset(FairseqDataset):
    def __init__(self, datasets, length, seed):
        self.datasets = datasets
        self.length = length
        full_len = len(datasets[0])
        assert length <= full_len
        for s in datasets[1:]:
            assert len(s) == full_len
        with data_utils.numpy_seed(seed):
            indices = torch.from_numpy(np.random.choice(full_len, length, replace=False))
        sizes = torch.zeros(len(datasets), length)
        for i, d in enumerate(datasets):
            sizes[i] = torch.tensor(d.sizes[indices])
        self.indices = indices
        self.sizes = torch.flatten(sizes)

    def __len__(self):
        return self.length * len(self.datasets)

    def __getitem__(self, idx):
        idx_d = idx // self.length
        idx_i = idx % self.length
        return self.datasets[idx_d][self.indices[idx_i]], None

class IRMocoValidIndexDataset(FairseqDataset):
    def __init__(self, datasets, length):
        self.num_datasets = len(datasets)
        self.length = length

    def __len__(self):
        return self.length * self.num_datasets

    def __getitem__(self, idx):
        idx_d = idx // self.length
        idx_i = idx % self.length
        return idx_d, idx_i

    def collater(self, samples):
        return samples

class IRMultiFunctionDataset(FairseqDataset):
    def __init__(self, dataset, indices, batch_size=-1):
        self.dataset = dataset
        if batch_size == -1:
            self.indices = indices
        else:
            self.indices = [x[:batch_size] for x in indices]
        self.length = len(self.indices)
        self.batch_size = batch_size
        self.sizes = torch.zeros(self.length, dtype=torch.long)
        for i in range(self.length):
            self.sizes[i] = self.dataset.sizes[self.indices[i]].mean()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [self.dataset[x] for x in self.indices[idx]]

    def collater(self, samples):
        all_samples = []
        for s in samples:
            for x in s:
                all_samples.append(x)
        return self.dataset.collater(all_samples), [len(s) for s in samples]