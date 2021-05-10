import os
import sys
import torch
import argparse

from fairseq import checkpoint_utils, options, progress_bar, utils
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
    IRBlockDataset,
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
    ListDataset,
)

parser = argparse.ArgumentParser()
parser.add_argument('--smallbert-encoder-layers', type=int)
parser.add_argument('--smallbert-insts-per-group', type=int)
parser.add_argument('--smallbert-num-attention-heads', type=int)
parser.add_argument('--function-length', type=int)
parser.add_argument('--encoder-layers', type=int)
parser.add_argument('--no-pooling', action='store_true')
parser.add_argument('--no-state', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()

def main():
    use_fp16 = args.fp16
    use_cuda = True
    extra_args = {
        'data':'../data-bin/pretrain',
        'smallbert_num_encoder_layers': args.smallbert_encoder_layers,
        'encoder_layers': args.encoder_layers,
        'smallbert_insts_per_group': args.smallbert_insts_per_group,
        'no_pooling': args.no_pooling,
        'no_state': args.no_state,
        'smallbert_num_attention_heads': args.smallbert_num_attention_heads}
    models, model_args = checkpoint_utils.load_model_ensemble(
        [args.checkpoint], extra_args
    )
    model = models[0]

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    print(model_args)

    data_path = args.data
    instruction_dictionary = Dictionary.load(os.path.join(data_path, 'insts', 'dict.txt'))
    state_dictionary = Dictionary.load(os.path.join(data_path, 'states', 'dict.txt'))
    inst_mask_idx = instruction_dictionary.add_symbol('<mask>')
    state_mask_idx = state_dictionary.add_symbol('<mask>')
    instruction_dictionary.add_symbol('<t>')
    state_dictionary.add_symbol('<t>')
    split = 'test'
    dataset_inst = data_utils.load_indexed_dataset(
        os.path.join(data_path, 'insts', split),
        instruction_dictionary,
        'mmap',
    )
    
    dataset_state = data_utils.load_indexed_dataset(
        os.path.join(data_path, 'states', split),
        state_dictionary,
        'mmap',
    )
    
    if dataset_inst is None or dataset_state is None:
        raise FileNotFoundError('Dataset not found: {} '.format(split))
    
    dataset_inst = SeqOfSeqDataset(dataset_inst, instruction_dictionary)
    dataset_state = SeqOfSeqDataset(dataset_state, state_dictionary)
    dataset_pos = IRPositionDataset(os.path.join(data_path, 'pos', split))
    dataset = IRDataset(dataset_inst, dataset_state, dataset_pos)
    
    
    dataset = IRPadDataset(
        dataset,
        inst_pad_idx=instruction_dictionary.pad(),
        state_pad_idx=state_dictionary.pad(),
        inst_mask_idx=inst_mask_idx,
        state_mask_idx=state_mask_idx,
        inst_cls_idx=instruction_dictionary.index('<t>'),
        state_cls_idx=state_dictionary.index('<t>'),
        smallbert_insts_per_input=args.smallbert_insts_per_group,
        smallbert_states_per_input=args.smallbert_insts_per_group,
        max_length=args.function_length,
        inst_pad_length=32,
        state_pad_length=16,
    )
    dataset_len = len(dataset)
    features = torch.cuda.FloatTensor(dataset_len, 768)
    with torch.no_grad():
        model.eval()
        for i in range(0, dataset_len, 16):
            end = min(i + 16, dataset_len)
            data = [dataset[i] for i in range(i, end)]
            output = model(src=dataset.collater(data, pair=True), features_only=True, moco_head=False, lm_head=False, moco_head_only_proj=False, classification_head_name=None, has_state=not args.no_state)
            feature = output[0][2]
            features[i:end, :] = feature
    torch.save(features.cpu(), args.output)

def cli_main():
    main()

if __name__ == '__main__':
    cli_main()
