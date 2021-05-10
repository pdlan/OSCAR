import sys
import os
import subprocess

in_dir = sys.argv[1]
program = sys.argv[2]
variants = sys.argv[3].split(',')
out_dir = sys.argv[4]
max_len = sys.argv[5]

def process(in_file, inst_file, state_file, pos_file):
    cmd = 'opt-11 -load ../bin/libanalysis.so'.split()
    cmd += '-staticanalysis -max-len'.split()
    cmd += [max_len, '-truncate']
    cmd += ['-inst-out', inst_file]
    cmd += ['-state-out', state_file]
    cmd += ['-pos-out', pos_file]
    cmd += '-func-name -o /dev/null'.split()
    subprocess.run(cmd, stdin=open(in_file), stderr=subprocess.STDOUT)

for v in variants:
    try:
        os.mkdir(os.path.join(out_dir, v))
    except:
        pass
    pos = program.find('.')
    if pos != -1:
        program = program[:pos]
    ll_file = os.path.join(in_dir, v, program + '.ll')
    with open(ll_file, 'r') as f:
        data = f.read()
    data = data.replace('%wide-string', 'zeroinitializer')
    with open(ll_file, 'w') as f:
        f.write(data)
    process(ll_file, os.path.join(out_dir, v, 'insts.json'),
        os.path.join(out_dir, v, 'states.json'), os.path.join(out_dir, v, 'pos.json'))
