import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def build_palette(k=6,s=None):
    if s==None:
        s = 250 // (k-1)
    else:
        assert s*(k-1)<255
    palette = []
    for m0 in range(k):
        for m1 in range(k):
            for m2 in range(k):
                palette.extend([s*m0,s*m1,s*m2])
    return palette




if __name__ == '__main__':
    pass





