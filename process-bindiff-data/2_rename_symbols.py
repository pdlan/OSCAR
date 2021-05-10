import sys
import os
import subprocess
import lief
import tempfile
import numpy as np

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
    fd, tmp_file = tempfile.mkstemp()
    strip = 'aarch64-linux-gnu-strip' if 'aarch64' in v else 'strip'
    subprocess.run([strip, '-g', '-o', tmp_file, in_file])
    elf = lief.parse(tmp_file)
    os.remove(tmp_file)
    
    i = 0
    for symbol in elf.symbols:
        if symbol.is_function and not symbol.imported:
            symbol.name = 'f'# + np.base_repr(i, base=36)
            i += 1
    elf.write(out_file)