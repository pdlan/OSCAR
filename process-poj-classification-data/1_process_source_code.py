import os
import sys
import re
import random
import json

seed = 0
in_dir = sys.argv[1]
out_dir = sys.argv[2]
num_classes = int(sys.argv[3])
random.seed(seed)

for i in range(1, num_classes + 1):
    try:
        for f in sorted(os.listdir(os.path.join(in_dir, str(i)))):
            in_filename = os.path.join(in_dir, str(i), f)
            name = '%d-%s' % (i, f[:-4])
            r = random.random()
            if r < 0.6:
                split = 'train'
            elif r < 0.8:
                split = 'valid'
            else:
                split = 'test'
            code = open(in_filename, 'rb').read().decode('utf-8', errors='ignore')
            with open(os.path.join(out_dir, split, name + '.cc'), 'w') as g:
                g.write("""
#include <bits/stdc++.h>
using namespace std;
""")
                code = code.replace('void main', 'int main')
                code = re.sub(r'^\s*main', 'int main', code, flags=re.M)
                g.write(code)
            
    except Exception as e:
        print(e)