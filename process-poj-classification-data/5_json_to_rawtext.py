import os
import sys
import subprocess
import json
import shutil

json_dir = sys.argv[1]
out_dir = sys.argv[2]
moco_path = sys.argv[3]
max_len = sys.argv[4]

inst_dict_file = os.path.join(moco_path, 'inst_dict.txt')
state_dict_file = os.path.join(moco_path, 'state_dict.txt')

state_dict = {}
for line in open(state_dict_file):
    s = line.strip().split()
    if len(s) == 2:
        token, freq = s
        state_dict[token] = int(freq)

def process_inst(in_file, out_file, dict_file):
    cmd = ['../bin/irlexer', 'process-inst', in_file, dict_file, dict_file, '../bin/fast', os.path.join(moco_path, 'bpe_codes'), max_len]
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

process_inst(os.path.join(json_dir, 'insts.json'), os.path.join(out_dir, 'insts.json'), inst_dict_file)
to_rawtext(os.path.join(out_dir, 'insts.json'), os.path.join(out_dir, 'insts.txt'))
to_rawtext(os.path.join(json_dir, 'states.json'), os.path.join(out_dir, 'states.txt'), state_dict)
shutil.copyfile(os.path.join(json_dir, 'pos.json'), os.path.join(out_dir, 'pos.json'))
