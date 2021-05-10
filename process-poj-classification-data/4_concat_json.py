import os
import sys
import subprocess
import json
import base64

ir_dir = sys.argv[1]
json_dir = sys.argv[2]
out_dir = sys.argv[3]
label = sys.argv[4]
classes = int(sys.argv[5])
out_inst = open(os.path.join(out_dir, 'insts.json'), 'w')
out_state = open(os.path.join(out_dir, 'states.json'), 'w')
out_pos = open(os.path.join(out_dir, 'pos.json'), 'w')
label_list = []

def process_inst(in_file):
    with open(os.path.join(json_dir, in_file + '.insts.json')) as f:
        func = []
        for line in f:
            data = json.loads(line.strip())
            if data is not None:
                func += data
        out_inst.write(json.dumps(func) + '\n')

def process_states(in_file):
    with open(os.path.join(json_dir, in_file + '.states.json')) as f:
        func = []
        for line in f:
            data = json.loads(line.strip())
            if data is not None:
                func += data
        out_state.write(json.dumps(func) + '\n')

def process_pos(in_file):
    with open(os.path.join(json_dir, in_file + '.pos.json')) as f:
        func = []
        for line in f:
            data = json.loads(line.strip())
            if data is not None:
                func += data
        out_pos.write(json.dumps(func) + '\n')

for c in range(1, classes + 1):
    try:
        for f in os.listdir(os.path.join(ir_dir, str(c))):
            name = f[:-3]
            filename = '%d.%s' % (c, name)
            if f.endswith('.ll'):
                l = str(c)
                #print('Processing %s' % f)
                label_list.append(l)
                process_inst(filename)
                process_states(filename)
                process_pos(filename)
    except Exception as e:
        print(e)

with open(os.path.join(out_dir, label), 'w') as f:
    f.write('\n'.join(label_list) + '\n')