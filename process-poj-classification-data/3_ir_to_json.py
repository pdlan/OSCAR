import os
import sys
import subprocess
import json
from multiprocessing import Pool

def process(in_file, inst_file, state_file, pos_file, max_len):
    cmd = 'opt-11 -load ../bin/libanalysis.so'.split()
    cmd += '-staticanalysis -max-len'.split()
    cmd += [max_len, '-truncate', '-concat-func']
    cmd += ['-inst-out', inst_file]
    cmd += ['-state-out', state_file]
    cmd += ['-pos-out', pos_file]
    cmd += '-o /dev/null'.split()
    subprocess.run(cmd, stdin=open(in_file), stderr=subprocess.STDOUT)

in_dir = sys.argv[1]
out_dir = sys.argv[2]
classes = int(sys.argv[3])
processes = int(sys.argv[4])
max_len = sys.argv[5]

pool = Pool(processes=processes)

for c in range(1, classes + 1):
    try:
        for f in os.listdir(os.path.join(in_dir, str(c))):
            in_filename = os.path.join(in_dir, str(c), f)
            name = f[:-3]
            #print('Processing %s' % name)
            args = (in_filename, os.path.join(out_dir, '%d.%s.insts.json' % (c, name)),
                os.path.join(out_dir, '%d.%s.states.json' % (c, name)), os.path.join(out_dir, '%d.%s.pos.json' % (c, name)), max_len)
            pool.apply_async(process, args)
    except:
        pass

pool.close()
pool.join()

#for f in os.listdir(in_dir):
#    in_filename = os.path.join(in_dir, f)
#    if os.path.isfile(in_filename):
#        if f.endswith('.ll'):
#            index = f[:-3]
#            print('Processing %s' % index)
#            process(os.path.join(in_dir, index + '.ll'), os.path.join(out_dir, index + '.insts.json'),
#                os.path.join(out_dir, index + '.states.json'), os.path.join(out_dir, index + '.pos.json'))