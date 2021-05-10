import sys
import os
import json
import torch
import numpy as np

from fairseq import options, tasks, utils
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer

def make_dataset(infile, outfile, idxfile):
    builder = indexed_dataset.MMapIndexedDatasetBuilder(outfile, np.int16)
    with open(infile, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line == '\n':
                print(i)
            pos = json.loads(line)
            builder.add_item(torch.tensor(pos, dtype=torch.long).transpose(0, 1))
    builder.finalize(idxfile)

def main(indir, outdir):
    make_dataset(os.path.join(indir, 'pos_train.json'), os.path.join(outdir, 'train.bin'), os.path.join(outdir, 'train.idx'))
    make_dataset(os.path.join(indir, 'pos_valid.json'), os.path.join(outdir, 'valid.bin'), os.path.join(outdir, 'valid.idx'))

if __name__ == '__main__':
    if len(sys.argv) == 4:
        if sys.argv[1] == 'moco':
            indir = sys.argv[2]
            outdir = sys.argv[3]
            main(indir, outdir)
        elif sys.argv[1] == 'bindifftrain':
            indir = sys.argv[2]
            outdir = sys.argv[3]
            make_dataset(os.path.join(indir, 'train', 'pos.json'), os.path.join(outdir, 'train.bin'), os.path.join(outdir, 'train.idx'))
            make_dataset(os.path.join(indir, 'valid', 'pos.json'), os.path.join(outdir, 'valid.bin'), os.path.join(outdir, 'valid.idx'))
        elif sys.argv[1] == 'bindiff':
            infile = sys.argv[2]
            outdir = sys.argv[3]
            make_dataset(infile, os.path.join(outdir, 'test.bin'), os.path.join(outdir, 'test.idx'))
        elif sys.argv[1] == 'poj':
            indir = sys.argv[2]
            outdir = sys.argv[3]
            make_dataset(os.path.join(indir, 'train', 'pos.json'), os.path.join(outdir, 'train.bin'), os.path.join(outdir, 'train.idx'))
            make_dataset(os.path.join(indir, 'valid', 'pos.json'), os.path.join(outdir, 'valid.bin'), os.path.join(outdir, 'valid.idx'))
            make_dataset(os.path.join(indir, 'test', 'pos.json'), os.path.join(outdir, 'test.bin'), os.path.join(outdir, 'test.idx'))
        elif sys.argv[1] == 'classification':
            indir = sys.argv[2]
            outdir = sys.argv[3]
            make_dataset(os.path.join(indir, 'train', 'pos.json'), os.path.join(outdir, 'train.bin'), os.path.join(outdir, 'train.idx'))
            make_dataset(os.path.join(indir, 'valid', 'pos.json'), os.path.join(outdir, 'valid.bin'), os.path.join(outdir, 'valid.idx'))