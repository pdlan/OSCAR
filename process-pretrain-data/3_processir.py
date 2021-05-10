import os
import sys
import multiprocessing
import subprocess

def process(in_dir, out_dir, i, irlexer, bpe_program, max_len):
    cmd = [irlexer, 'process-inst']
    option_out_dir = '%s/%d/' % (out_dir, i)
    cmd += ['%s/%d/insts.json' % (in_dir, i)]
    cmd += [
        option_out_dir + 'total_dict.txt',
        option_out_dir + 'symbol_dict.txt',
        bpe_program,
        out_dir + '/bpe_codes',
        max_len
    ]
    subprocess.call(cmd, stdout=open('%s/%d/insts.json' % (out_dir, i), 'w'))

def main(in_dir, out_dir, num_options, num_proc, irlexer, bpe_program, max_len):
    pool = multiprocessing.Pool(processes=num_proc)
    for i in range(num_options):
        try:
            os.mkdir('%s/%d' % (out_dir, i))
        except:
            pass
        cmd = [irlexer, 'build-dict']
        option_out_dir = '%s/%d/' % (out_dir, i)
        cmd += ['%s/%d/insts.json' % (in_dir, i)]
        cmd += [
            option_out_dir + 'total_dict.txt',
            option_out_dir + 'symbol_dict.txt',
            option_out_dir + 'symbol_tokens.txt',
        ]
        cmd += [max_len, '50']
        pool.apply_async(subprocess.call, (cmd, ))
        #subprocess.call(cmd)
    pool.close()
    pool.join()
    print('irlexer 1')

    filenames = ['%s/%d/symbol_tokens.txt' % (out_dir, i) for i in range(num_options)]
    cmd = ['cat'] + filenames
    subprocess.call(cmd, stdout=open(out_dir + '/symbol_tokens.txt', 'w'))
    cmd = [bpe_program, 'learnbpe', '10000', out_dir + '/symbol_tokens.txt']
    subprocess.call(cmd, stdout=open(out_dir + '/bpe_codes', 'w'))
    print('fastbpe')

    pool = multiprocessing.Pool(processes=num_proc)
    for i in range(num_options):
        pool.apply_async(process, (in_dir, out_dir, i, irlexer, bpe_program, max_len))
    pool.close()
    pool.join()
    print('finish')

if __name__ == '__main__':
    if len(sys.argv) == 8:
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]
        num_options = int(sys.argv[3])
        num_proc = int(sys.argv[4])
        irlexer = sys.argv[5]
        bpe_program = sys.argv[6]
        max_len = sys.argv[7]
        main(in_dir, out_dir, num_options, num_proc, irlexer, bpe_program, max_len)
    else:
        print('Usage: python3 3-processir.py <stage2 out dir> <stage3 out dir> <num options> <num_procs> <irlexer program> <bpe program> <max length>')