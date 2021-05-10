import sys
import os
import subprocess

in_dir = sys.argv[1]
program = sys.argv[2]
variants = sys.argv[3].split(',')
out_dir = sys.argv[4]

for v in variants:
    in_file = os.path.join(in_dir, v, program)
    try:
        os.mkdir(os.path.join(out_dir, v))
    except:
        pass
    out_file = os.path.join(out_dir, v, program)
    subprocess.run(['retdec-decompiler', '--keep-unreachable-funcs', '--stop-after', 'bin2llvmir', '-o', out_file, in_file])