import sys
import os
import subprocess
import json
import shutil

in_dir = sys.argv[1]
program = sys.argv[2]
variants = sys.argv[3].split(',')
out_dir = sys.argv[4]
inst_dict_file = sys.argv[5]
state_dict_file = sys.argv[6]
max_len = sys.argv[7]

state_dict = {}
for line in open(state_dict_file):
    s = line.strip().split()
    if len(s) == 2:
        token, freq = s
        state_dict[token] = int(freq)

def process_inst(in_file, out_file, dict_file):
    cmd = ['../bin/irlexer', 'process-inst', in_file, dict_file, dict_file, '../bin/fast', '../data-bin/pretrain/bpe_codes', max_len]
    subprocess.run(cmd, stdout=open(out_file, 'w'), stderr=subprocess.STDOUT)

def to_rawtext(in_file, out_file, vocab=None):
    with open(in_file) as fi, open(out_file, 'w') as fo:
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
                        elif t in vocab:
                            tokens.append(t)
                        elif t[0] in ['a', 'v', 'm']:
                            tokens.append('<unk>')
                        else:
                            tokens.append('<const>')
            fo.write(' '.join(tokens) + '\n')

for v in variants:
    try:
        os.mkdir(os.path.join(out_dir, v))
    except:
        pass
    process_inst(os.path.join(in_dir, v, 'insts.json'), os.path.join(out_dir, v, 'insts.json'), inst_dict_file)
    to_rawtext(os.path.join(out_dir, v, 'insts.json'), os.path.join(out_dir, v, 'insts.txt'))
    to_rawtext(os.path.join(in_dir, v, 'states.json'), os.path.join(out_dir, v, 'states.txt'), state_dict)
    shutil.copyfile(os.path.join(in_dir, v, 'pos.json'), os.path.join(out_dir, v, 'pos.json'))
