import sys
import os
import subprocess
import json
import base64
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import StringTableSection

in_dir = sys.argv[1]
program = sys.argv[2]
variants = sys.argv[3].split(',')
out_dir = sys.argv[4]
function_info_file = sys.argv[5]
retdec_dir = sys.argv[6]
bin_dir = sys.argv[7]
out_file = sys.argv[8]

num_variants = len(variants)
retdec_name_addr_map = []
retdec_addr = []
function_info = json.loads(open(function_info_file, 'r').read())
addr_name_map = function_info['addr_name_map']
retdec_func_type = []

for v in variants:
    m = {}
    a = set()
    pos = program.find('.')
    if pos != -1:
        prog = program[:pos]
    else:
        prog = program
    retdec_file = os.path.join(retdec_dir, v, prog + '.config.json')
    retdec_config = json.loads(open(retdec_file, 'r').read())
    t = {}
    for f in retdec_config['functions']:
        addr = int(f['startAddr'], 16)
        m[f['name']] = addr
        a.add(addr)
        t[addr] = f['fncType']
    retdec_name_addr_map.append(m)
    retdec_addr.append(a)
    retdec_func_type.append(t)

def get_elf_section_range(filename):
    file = ELFFile(open(filename, 'rb'))
    for sect in file.iter_sections():
        if isinstance(sect, StringTableSection):
            sts = sect
    ranges = []
    for sect in file.iter_sections():
        if sts.get_string(sect['sh_name']) in ['.plt', '.plt.got']:
            start = sect['sh_addr']
            size = sect['sh_size']
            ranges.append((start, start + size))
    def func(addr):
        for r in ranges:
            if r[0] <= addr < r[1]:
                return True
        return False
    return func

pltgot = []
for v in variants:
    pltgot.append(get_elf_section_range(os.path.join(bin_dir, v, program)))

id_addr_map = [None for _ in range(num_variants)]
addr_id_map = [None for _ in range(num_variants)]
pltgot_functions = {}

def write_json(in_file, out_file, v):
    #print(len(retdec_name_addr_map[v]))
    with open(in_file, 'r') as fi, open(out_file, 'w') as fo:
        i = 0
        iam = []
        aim = {}
        while True:
            name = fi.readline()
            if not name:
                break
            name = name.strip()
            addr = retdec_name_addr_map[v][name]
            data = fi.readline()
            #if str(addr) not in addr_name_map[v]:
            #    continue
            if pltgot[v](addr):
                if name in pltgot_functions:
                    pltgot_functions[name].append((v, addr))
                else:
                    pltgot_functions[name] = [(v, addr)]
            fo.write(data)
            iam.append(addr)
            aim[addr] = i
            i += 1
        id_addr_map[v] = iam
        addr_id_map[v] = aim
    #print(i)

for i, v in enumerate(variants):
    try:
        os.mkdir(os.path.join(out_dir, v))
    except:
        pass
    write_json(os.path.join(in_dir, v, 'insts.json'), os.path.join(out_dir, v, 'insts.json'), i)
    write_json(os.path.join(in_dir, v, 'states.json'), os.path.join(out_dir, v, 'states.json'), i)
    write_json(os.path.join(in_dir, v, 'pos.json'), os.path.join(out_dir, v, 'pos.json'), i)

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
matched_functions = []
for k, v in line_addr_map.items():
    if len(v) > 1:
        m = []
        for i, j in v:
            ind = variants.index(i)
            if j == 0:
                print(i, k)
                continue
            m.append((i, addr_id_map[ind][j]))
            #print(retdec_func_type[ind][j])
        matched_functions.append(m)
with open(out_file, 'w') as f:
    f.write(json.dumps({
        'addr': {variants[i]: v for i, v in enumerate(id_addr_map)},
        'match': matched_functions,
        'pltgot': pltgot_functions
    }))