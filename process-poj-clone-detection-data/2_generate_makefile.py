import os
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]
makefile_name = sys.argv[3]
out_filenames = []
makefile = ''

for f in os.listdir(in_dir):
    in_filename = os.path.join(in_dir, f)
    if os.path.isfile(in_filename):
        if f.endswith('.cc'):
            index = f[:-3]
            #if index in uncompilable:
            #    continue
            out_filename = os.path.join(out_dir, index + '.ll')
            cmd = 'clang-11 -O0 -S -emit-llvm "%s" -o "%s" -std=c++98 -Wno-everything\n' % (in_filename, out_filename)
            out_filenames.append(out_filename)
            makefile += '%s: %s\n\t%s' % (out_filename, in_filename, cmd)

with open(makefile_name, 'w') as f:
    f.write('all: ' + ' '.join(out_filenames) + '\n')
    f.write(makefile)
