import os
import sys
import subprocess
import json
import base64

ir_dir = sys.argv[1]
json_dir = sys.argv[2]
out_dir = sys.argv[3]
funclist = sys.argv[4]
original_data_filename = sys.argv[5]
label = sys.argv[6]
function_list = []
index_label_dict = {}
with open(original_data_filename) as f:
    for line in f:
        x = json.loads(line)
        index_label_dict[x['index']] = x['label']
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
        function_list.append(in_file)
        label_list.append(index_label_dict[in_file])

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

for f in os.listdir(ir_dir):
    in_filename = os.path.join(ir_dir, f)
    if os.path.isfile(in_filename):
        if f.endswith('.ll'):
            index = f[:-3]
            print('Processing %s' % f)
            process_inst(index)
            process_states(index)
            process_pos(index)

with open(os.path.join(out_dir, funclist), 'w') as f:
    for func in function_list:
        f.write(json.dumps(func) + '\n')
with open(os.path.join(out_dir, label), 'w') as f:
    f.write('\n'.join(label_list) + '\n')
