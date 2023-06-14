import hashlib, pickle, numpy as np

import pandas as pd
import re

def update_lossdict(lossdict, update, action='append'):
    for k in update.keys():
        if action == 'append':
            if k in lossdict:
                lossdict[k].append(update[k])
            else:
                lossdict[k] = [update[k]]
        elif action == 'sum':
            if k in lossdict:
                lossdict[k] += update[k]
            else:
                lossdict[k] = update[k]
        else:
            raise NotImplementedError
    return lossdict

def depickle(filepath):
#     with open(filepath, mode='rb') as f: return pickle.load(f)
    with open(filepath, mode='rb') as f: return pd.read_pickle(f,compression=None)
def read_txt(filepath):
    with open(filepath, mode='r') as f: return f.read()

# For debugging pandas apply/transform ops.
def print_and_raise(*args, **kwargs):
    print(args, kwargs)
    raise NotImplementedError

def zip_dicts_assert(*dcts):
    d0 = dcts[0]
    s0 = set(d0.keys())
    for d in dcts[1:]:
        s1 = set(d.keys())
        assert d0.keys() == d.keys(), f"Keys Disagree! d0 - d1 = {s0 - s1}, d1 - d0 = {s1 - s0}"

    for i in set(dcts[0]).intersection(*dcts[1:]): yield (i,) + tuple(d[i] for d in dcts)

def zip_dicts(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]): yield (i,) + tuple(d[i] for d in dcts)

def zip_dicts_union(*dcts):
    keys = set(dcts[0].keys())
    for d in dcts[1:]: keys.update(d.keys())

    for k in keys: yield (k,) + tuple(d[k] if k in d else np.NaN for d in dcts)
