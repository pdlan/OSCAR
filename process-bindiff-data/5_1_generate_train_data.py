import sys
import os
import subprocess
import json
import base64

in_dir = sys.argv[1]
program = sys.argv[2]
variants = sys.argv[3].split(',')
out_dir = sys.argv[4]
function_info_file = sys.argv[5]
retdec_dir = sys.argv[6]

num_variants = len(variants)
function_info = json.loads(open(function_info_file, 'r').read())

name_addr_map = []
for m in function_info['name_addr_map']:
    name_addr_map.append({base64.b64decode(x): y for x, y in m.items()})
addr_line_map = []
for m in function_info['addr_line_map']:
    addr_line_map.append({int(x): (base64.b64decode(y[0]), y[1]) for x, y in m.items()})
line_addr_map = {}
for x, y in function_info['line_addr_map'].items():
    x = json.loads(x)
    k = (base64.b64decode(x[0]), base64.b64decode(x[1]))
    y = [(z[0], z[1]) for z in y]
    line_addr_map[k] = y
retdec_name_addr_map = []
retdec_addr = []
for v in variants:
    m = {}
    a = set()
    pos = program.find('.')
    if pos != -1:
        program = program[:pos]
    retdec_file = os.path.join(retdec_dir, v, program + '.config.json')
    retdec_config = json.loads(open(retdec_file, 'r').read())
    for f in retdec_config['functions']:
        addr = int(f['startAddr'], 16)
        m[f['name']] = addr
        a.add(addr)
    retdec_name_addr_map.append(m)
    retdec_addr.append(a)

common_functions = []

for k, v in line_addr_map.items():
    #if len(v) != num_variants:
    #    continue
    f = [None for _ in range(num_variants)]
    is_in_program = True
    for i, j in v:
        i = variants.index(i)
        f[i] = j
        if j not in retdec_addr[i]:
            is_in_program = False
    if None in f:
        continue
    #if is_in_program:
    common_functions.append(f)
def write_json(in_file, out_file, v):
    with open(in_file, 'r') as fi, open(out_file, 'w') as fo:
        lines = {}
        while True:
            name = fi.readline()
            if not name:
                break
            name = name.strip()
            addr = retdec_name_addr_map[v][name]
            data = fi.readline()
            lines[addr] = data
        for f in common_functions:
            line = lines[f[v]]
            fo.write(line)

for i, v in enumerate(variants):
    try:
        os.mkdir(os.path.join(out_dir, v))
    except:
        pass
    write_json(os.path.join(in_dir, v, 'insts.json'), os.path.join(out_dir, v, 'insts.json'), i)
    write_json(os.path.join(in_dir, v, 'states.json'), os.path.join(out_dir, v, 'states.json'), i)
    write_json(os.path.join(in_dir, v, 'pos.json'), os.path.join(out_dir, v, 'pos.json'), i)
