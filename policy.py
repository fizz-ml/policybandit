import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SIZE_H1 = 5
SIZE_H2 = 10
SIZE_H3 = 6

class Policy(torch.nn.Module):
    """
    Outputs probability over actions.
    Defines custom model.
    Inherits from torch.nn.Module
    """
    def __init__(self, dim_input, dim_output):

        super(Policy, self).__init__()
        self._dim_input = dim_input
        self._dim_output = dim_output

        '''Initialize net layers'''
        self._l1 = torch.nn.Linear(self._dim_input, SIZE_H1)
        self._l2 = torch.nn.Linear(SIZE_H1, SIZE_H2)
        self._l3 = torch.nn.Linear(SIZE_H2, SIZE_H3)
        self._l4 = torch.nn.Linear(SIZE_H3, self._dim_output)

    def forward(self,s_t):
        x = s_t
        x.register_hook(lambda x:print("l3o",x))
        l1_out = F.relu(self._l1(x))
        l1_out.register_hook(lambda x:print("l3o",x))
        l2_out = F.relu(self._l2(l1_out))
        l2_out.register_hook(lambda x:print("l3o",x))
        l3_out = F.relu(self._l3(l2_out))
        l3_out.register_hook(lambda x:print("l3o",x))
        l4_out = self._l4(l3_out)
        l4_out.register_hook(lambda x:print("l4o",x))
        out = F.tanh(l4_out)
        print(out)
        out.register_hook(lambda x:print("out",x))

        return out

"""
    def forward(self,s_t):
        x = s_t
        self._l1_out = F.relu(self._l1(x))
        self._l2_out = F.relu(self._l2(self._l1_out))
        self._l3_out = F.relu(self._l3(self._l2_out))
        self._l3_out.register_hook(lambda x:print("l3o",x))
        self._l4_out = self._l4(self._l3_out)
        self._l4_out.register_hook(lambda x:print("l4o",x))
        self._out = F.tanh(self._l4_out)
        print(self._out)
        self._out.register_hook(lambda x:print("out",x))

        return self._out
        
"""

