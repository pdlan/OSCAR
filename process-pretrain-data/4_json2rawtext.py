import os
import sys
import random
import json
import multiprocessing

def split_raw(data, train, valid, vocab=None, min_freq=0):
    random.seed(0)
    with open(data) as fi, open(train, 'w') as ft, open(valid, 'w') as fv:
        for line in fi:
            data = json.loads(line)
            tokens = []
            for d in data:
                if tokens:
                    tokens.append('</s>')
                if vocab is None:
                    tokens += d
                else:
                    for t in d:
                        if not t:
                            continue
                        elif t in vocab and vocab[t] >= min_freq:
                            tokens.append(t)
                        elif t[0] in ['a', 'v', 'm']:
                            tokens.append('<unk>')
                        else:
                            tokens.append('<const>')
            if random.random() > 0.95:
                fv.write(' '.join(tokens) + '\n')
            else:
                ft.write(' '.join(tokens) + '\n')

def split_json(data, train, valid):
    random.seed(0)
    with open(data) as fi, open(train, 'w') as ft, open(valid, 'w') as fv:
        for line in fi:
            if random.random() > 0.95:
                fv.write(line)
            else:
                ft.write(line)

def main(in2_dir, in3_dir, out_dir, num_options, num_proc, dict_filename, min_freq):
    pool = multiprocessing.Pool(processes=num_proc)
    state_dict = {}
    for line in open(dict_filename):
        s = line.strip().split()
        if len(s) == 2:
            token, freq = s
            state_dict[token] = int(freq)
    for i in range(num_options):
        try:
            os.mkdir('%s/%d' % (out_dir, i))
        except:
            pass
        pool.apply_async(split_raw, ('%s/%d/insts.json' % (in3_dir, i),
            '%s/%d/insts_train.txt' % (out_dir, i), '%s/%d/insts_valid.txt' % (out_dir, i)))
        pool.apply_async(split_raw, ('%s/%d/states.json' % (in2_dir, i),
            '%s/%d/states_train.txt' % (out_dir, i), '%s/%d/states_valid.txt' % (out_dir, i), state_dict, min_freq))
        pool.apply_async(split_json, ('%s/%d/pos.json' % (in2_dir, i),
            '%s/%d/pos_train.json' % (out_dir, i), '%s/%d/pos_valid.json' % (out_dir, i)))
    pool.close()
    pool.join()

if __name__ == '__main__':
    if len(sys.argv) == 8:
        in2_dir = sys.argv[1]
        in3_dir = sys.argv[2]
        out_dir = sys.argv[3]
        num_options = int(sys.argv[4])
        num_proc = int(sys.argv[5])
        dict_filename = sys.argv[6]
        min_freq = int(sys.argv[7])
        main(in2_dir, in3_dir, out_dir, num_options, num_proc, dict_filename, min_freq)
    else:
        print('Usage: python3 4_json2rawtext.py <stage 2 out dir> <stage3 out dir> <stage4 out dir> <num options> <num proc> <state dict> <state min freq>')