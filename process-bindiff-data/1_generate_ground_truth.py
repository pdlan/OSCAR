import sys
import os
import json
import base64
import numpy as np
from elftools.elf.elffile import ELFFile

in_dir = sys.argv[1]
program = sys.argv[2]
variants = sys.argv[3].split(',')
out_file = sys.argv[4]

name_addr_map = []
addr_line_map = []
addr_name_map = []

for v in variants:
    in_file = os.path.join(in_dir, v, program)
    elffile = ELFFile(open(in_file, 'rb'))
    dwarfinfo = elffile.get_dwarf_info()
    functions = {}
    addresses = []
    addr_func_map = {}

    for CU in dwarfinfo.iter_CUs():
        for DIE in CU.iter_DIEs():
            if DIE.tag == 'DW_TAG_subprogram':
                try:
                    name = DIE.attributes['DW_AT_name'].value
                    addr = DIE.attributes['DW_AT_low_pc'].value
                    name = base64.b64encode(name).decode('utf-8')
                    functions[name] = addr
                    addr_func_map[addr] = name
                    addresses.append(addr)
                except:
                    pass
    addr_name_map.append(addr_func_map)
    addresses = np.array(list(set(addresses)))
    addresses = np.sort(addresses)
    lines = {}
    for CU in dwarfinfo.iter_CUs():
        lineprog = dwarfinfo.line_program_for_CU(CU)
        prevstate = None
        for entry in lineprog.get_entries():
            if entry.state is None:
                continue
            if entry.state.end_sequence:
                prevstate = None
                continue
            if prevstate:
                low = prevstate.address
                high = entry.state.address
                start = np.searchsorted(addresses, low)
                end = np.searchsorted(addresses, high)
                for addr in addresses[start:end]:
                    filename = lineprog['file_entry'][prevstate.file - 1].name
                    line = prevstate.line
                    lines[int(addr)] = (base64.b64encode(filename).decode('utf-8'), line)
            prevstate = entry.state
    
    name_addr_map.append(functions)
    addr_line_map.append(lines)

line_addr_map = {}
for i, v in enumerate(variants):
    for a, l in addr_line_map[i].items():
        #print(a, l[0], addr_name_map[i][a])
        l = (l[0], addr_name_map[i][a])
        l = json.dumps(l, separators=(',', ':'))
        if l in line_addr_map:
            line_addr_map[l].append((v, a))
        else:
            line_addr_map[l] = [(v, a)]
res = {
    'addr_name_map': addr_name_map,
    'name_addr_map': name_addr_map,
    'addr_line_map': addr_line_map,
    'line_addr_map': line_addr_map
}

with open(out_file, 'w') as f:
    f.write(json.dumps(res, separators=(',', ':')))