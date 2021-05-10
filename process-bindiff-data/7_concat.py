import sys
import os

out_dir = sys.argv[1]
variants = sys.argv[2].split(',')
in_dir = sys.argv[3:]

#def concat(out_file, in_files):
#    with open(out_file, 'w') as fo:
#        for fi in in_files:
#            fo.write(open(fi, 'r').read())
#for v in variants:
#    try:
#        os.mkdir(os.path.join(out_dir, v))
#    except:
#        pass
#    concat(os.path.join(out_dir, v, 'insts.txt'), [os.path.join(x, v, 'insts.txt') for x in in_dir])
#    concat(os.path.join(out_dir, v, 'states.txt'), [os.path.join(x, v, 'states.txt') for x in in_dir])
#    concat(os.path.join(out_dir, v, 'pos.json'), [os.path.join(x, v, 'pos.json') for x in in_dir])

labels = []

with open(os.path.join(out_dir, 'insts.txt'), 'w') as fo:
    for d in in_dir:
        for v in variants:
            with open(os.path.join(d, v, 'insts.txt'), 'r') as fi:
                fo.write(fi.read())
with open(os.path.join(out_dir, 'states.txt'), 'w') as fo:
    for d in in_dir:    
        for v in variants:
            with open(os.path.join(d, v, 'states.txt'), 'r') as fi:
                fo.write(fi.read())
i = 1
with open(os.path.join(out_dir, 'pos.json'), 'w') as fo:
    for d in in_dir:
        num = 0
        for v in variants:
            with open(os.path.join(d, v, 'pos.json'), 'r') as fi:
                data = fi.read()
                lines = data.count('\n')
                labels += list(range(i, i + lines))
                assert num == 0 or num == lines
                num = lines
                fo.write(data)
        i += num
with open(os.path.join(out_dir, 'label.txt'), 'w') as fo:
    fo.write('\n'.join(map(str, labels)) + '\n')