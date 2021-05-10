import os
import sys
import subprocess
import json

in_dir = sys.argv[1]
out_dir = sys.argv[2]
max_len = sys.argv[3]

def process(in_file, inst_file, state_file, pos_file):
    cmd = 'opt-11 -load ../bin/libanalysis.so'.split()
    cmd += '-staticanalysis -max-len'.split()
    cmd += [max_len, '-truncate']
    cmd += ['-inst-out', inst_file]
    cmd += ['-state-out', state_file]
    cmd += ['-pos-out', pos_file]
    cmd += '-concat-func -o /dev/null'.split()
    subprocess.run(cmd, stdin=open(in_file), stderr=subprocess.STDOUT)

for f in os.listdir(in_dir):
    in_filename = os.path.join(in_dir, f)
    if os.path.isfile(in_filename):
        if f.endswith('.ll'):
            index = f[:-3]
            print('Processing %s' % index)
            process(os.path.join(in_dir, index + '.ll'), os.path.join(out_dir, index + '.insts.json'),
                os.path.join(out_dir, index + '.states.json'), os.path.join(out_dir, index + '.pos.json'))
