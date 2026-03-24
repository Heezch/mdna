import numpy as np
import sys, os

from .read_params import GenStiffness
from .... import math as so3

class Groundstate:
    def __init__(self,method='md'):
        self.genstiff = GenStiffness(method=method)
        
    def gen(self,seq: str):
        stiff, gs = self.genstiff.gen_params(seq,use_group=True)
        conf = np.zeros((len(seq),4,4))
        conf[0] = np.eye(4)
        for i in range(len(seq)-1):
            g = so3.se3_euler2rotmat(gs[i])
            conf[i+1] = conf[i] @ g
        return conf

if __name__ == '__main__':
    
    seq = sys.argv[1]
    gs = Groundstate(method='md')
    conf = gs.gen(seq)
    
    for t in conf:
        print(t)