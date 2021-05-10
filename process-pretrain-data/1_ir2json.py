import os
import sys
import random
import multiprocessing
import subprocess
import json
import random
import numpy as np

np.random.seed(0)

o1_options = [s.strip() for s in open('compile-options/o1.txt', 'r').readlines()]
o2_options = [s.strip() for s in open('compile-options/o2.txt', 'r').readlines()]
o3_options = [s.strip() for s in open('compile-options/o3.txt', 'r').readlines()]
options = [s.strip() for s in open('compile-options/opt.txt', 'r').readlines()]
o2_inline_options = [s.strip() for s in open('compile-options/o2_inline.txt', 'r').readlines()]
o3_inline_options = [s.strip() for s in open('compile-options/o3_inline.txt', 'r').readlines()]
inline_options = [s.strip() for s in open('compile-options/opt_inline.txt', 'r').readlines()]

def drop_list(l, n):
    idx = np.sort(np.random.choice(len(l), len(l) - n, replace=False))
    return np.array(l)[idx]

def shuffle(l, n):
    n = min(len(l) // 2, n)
    idx = np.random.choice(len(l), 2 * n, replace=False)
    i1 = idx[:n]
    i2 = idx[n:]
    l[i1], l[i2] = l[i2], l[i1]
    return list(l)

def generate_random_opt_with_inline(max_drop_opt=1.0, max_shuffle_pairs=20):
    inline_index = inline_options.index('-inline')
    head = inline_options[:inline_index+1]
    l = inline_options[inline_index+1:]
    l = drop_list(l, np.random.randint(len(l) * max_drop_opt))
    l = shuffle(l, np.random.randint(max_shuffle_pairs))
    return head + l

def generate_random_opt(max_drop_opt=1.0, max_shuffle_pairs=20):
    l = options.copy()
    l = drop_list(l, np.random.randint(len(l) * max_drop_opt))
    l = shuffle(l, np.random.randint(max_shuffle_pairs))
    return l

def generate_opts(inline_prob=0.5, n=20):
    if np.random.uniform() <= inline_prob:
        options = [o2_inline_options.copy(), o3_inline_options.copy()]
        for _ in range(n - 2):
            options.append(generate_random_opt_with_inline())
    else:
        options = [[], o1_options.copy(), o2_options.copy(), o3_options.copy()]
        for _ in range(n - 4):
            options.append(generate_random_opt())
    return options

def process(options, in_file, inst_file, state_file, pos_file):
    cmd = 'opt-11 -load ../bin/libanalysis.so'.split()
    cmd += options
    cmd += '-staticanalysis -max-len 511 -min-len 8'.split()
    cmd += ['-inst-out', inst_file]
    cmd += ['-state-out', state_file]
    cmd += ['-pos-out', pos_file]
    cmd += '-func-name -o /dev/null'.split()
    subprocess.run(cmd, stdin=open(in_file), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def main(files, out_dir, inline_prob, num_options, num_proc):
    options = []
    pool = multiprocessing.Pool(processes=num_proc)
    for file in files:
        prob = 1.0 if 'linux' in file else inline_prob
        options.append(generate_opts(prob, num_options))
    #open('opt.json', 'w').write(json.dumps(options))
    for i in range(num_options):
        try:
            os.mkdir('%s/%d' % (out_dir, i))
        except:
            pass
        for j, f in enumerate(files):
            inst_file = '%s/%d/insts.%d.json' % (out_dir, i, j)
            state_file = '%s/%d/states.%d.json' % (out_dir, i, j)
            pos_file = '%s/%d/pos.%d.json' % (out_dir, i, j)
            pool.apply_async(process, (options[j][i], f, inst_file, state_file, pos_file))
    pool.close()
    pool.join()

if __name__ == '__main__':
    if len(sys.argv) == 6:
        in_dir = sys.argv[1]
        #files = [os.path.join(in_dir, x) for x in os.listdir(in_dir)]
        out_dir = sys.argv[2]
        inline_prob = float(sys.argv[3])
        num_options = int(sys.argv[4])
        num_proc = int(sys.argv[5])
        max_size = 8 * 1024 * 1024
        files = []
        for x in os.listdir(in_dir):
            f = os.path.join(in_dir, x)
            if os.path.getsize(f) <= max_size:
                files.append(f)
        
        main(files, out_dir, inline_prob, num_options, num_proc)
    else:
        print('Usage: python3 1_ir2json.py <in dir> <out dir> <inline prob> <num of options> <num of processes> <max size>')