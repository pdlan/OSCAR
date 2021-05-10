import sys
import os
import json
import torch
import torch.nn.functional as F

threshold = 0
program = sys.argv[1]
feature_dir = sys.argv[2]

def eval_bindiff(variant1, variant2):

    features1 = torch.load(os.path.join(feature_dir, variant1 + '.pt')).cuda().double()
    features2 = torch.load(os.path.join(feature_dir, variant2 + '.pt')).cuda().double()
    
    ground_truth = json.loads(open(os.path.join('../data-bin/bindiff-eval', program, 'matched_functions.json'), 'r').read())
    matched_functions = {}
    for m in ground_truth['match']:
        v1 = None
        v2 = None
        for i, j in m:
            if i == variant1:
                v1 = j
            if i == variant2:
                v2 = j
        if v1 is not None and v2 is not None:
            matched_functions[v1] = v2
    
    predict = {}
    d1 = features1.shape[0]
    d2 = features2.shape[0]
    
    fn1 = torch.norm(features1, p=2, dim=1, keepdim=True)
    fn2 = torch.norm(features2, p=2, dim=1, keepdim=True)
    features1 = features1 / fn1
    features2 = features2 / fn2
    sim = torch.mm(features1, features2.t())

    for i in range(d1):
        m = torch.max(sim[i], 0)
        if m.values.item() > threshold:
            predict[i] = m.indices.item()
    tp = 0
    n = len(matched_functions)
    
    for k, v in matched_functions.items():
        if k in predict and predict[k] == v:
            tp += 1
    r = tp / n
    return r

variants = ['O0-gcc7.5.0-amd64', 'O1-gcc7.5.0-amd64', 'O2-gcc7.5.0-amd64', 'O3-gcc7.5.0-amd64']
r = [
    eval_bindiff(variants[0], variants[1]),
    eval_bindiff(variants[0], variants[3]),
    eval_bindiff(variants[1], variants[3]),
    eval_bindiff(variants[2], variants[3])]
print('O0-O1:', r[0])
print('O0-O3:', r[1])
print('O1-O3:', r[2])
print('O2-O3:', r[3])
print('Avg: ', sum(r) / 4)
