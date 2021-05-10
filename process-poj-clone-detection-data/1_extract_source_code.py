import os
import sys
import re
import json

in_filename = sys.argv[1]
out_dir = sys.argv[2]
in_file = open(in_filename)

for line in in_file:
    data = json.loads(line)
    index = data['index']
    code = data['code']
    with open(os.path.join(out_dir, index + '.cc'), 'w') as f:
        f.write("""
#include <bits/stdc++.h>
using namespace std;
""")
        #if ('[N]' in code or '[N+1]' in code) and 'int N' not in code and 'int i,N,k,j,m,x;' not in code:
        #    f.write('#define N 1024\n')
        #if '[LEN]' in code:
        #    f.write('#define LEN 1024\n')
        #if '[SIZE]' in code and 'SIZE=' not in code:
        #    f.write('#define SIZE 1024\n')
        #if '[max+1]' in code:
        #    f.write('#define max 1024\n')
        #if 'MAXN' in code and 'const int MAXN' not in code:
        #    f.write('#define MAXN 1024\n')
        #if 'maxn' in code and 'maxn = ' not in code and 'maxn=' not in code:
        #    f.write('#define maxn 1024\n')
        #if 'COL' in code:
        #    f.write('#define COL 1024\n#define ROW 1024\n')
        #if 'MAX' in code:
        #    f.write('#define MAX 1024\n')
        #if 'WSIZE' in code:
        #    f.write('#define WSIZE 1024\n')
        #if 'MAX_LINE' in code:
        #    f.write('#define MAX_LINE 1024\n')
        #if 'MN' in code:
        #    f.write('#define MN 1024\n')
        #code = code.replace('friend', 'friend_')
        #code = code.replace('y1', 'y1_')
        code = code.replace('void main', 'int main')
        code = re.sub(r'^\s*main', 'int main', code, flags=re.M)
        f.write(code)
