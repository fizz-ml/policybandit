import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bayestorch.models import Linear


class SimpleCritic(ParameterGroup):
    """
    Outputs probability over actions.
    Defines custom model.
    Inherits from torch.nn.Module
    """
    def __init__(self, n, s_size, a_size):
        self._dim_input = n * (s_size + a_size)
        # only one reward
        self._dim_output = 1

        SIZE_H1 = 10 * n
        SIZE_H2 = 10 * n
        SIZE_H3 = 10 * n

        std_dev = 1
        '''Initialize net layers'''
        self._l1 = Linear(self._dim_input, SIZE_H1, std_dev)
        self._l2 = Linear(SIZE_H1, SIZE_H2, std_dev)
        self._l3 = Linear(SIZE_H2, SIZE_H3, std_dev)
        self._l4 = Linear( SIZE_H3, self._dim_output, std_dev)
        self._l41 = Linear( SIZE_H3, self._dim_output, std_dev)

        param_dict = {'l1':self._l1, 'l2':self._l2, 'l3':self._l3, 'l4':self._l4, 'l41':self._l41}

        super(SimpleCritic, self).__init__(param_dict)

    def forward(self, s_h, a_h):
        x = torch.cat((s_h, a_h), dim=-1)
        x = x.view(x.size(0), -1)
        self._l1_out = F.relu(self._l1(x))
        self._l2_out = F.relu(self._l2(self._l1_out))
        self._l3_out = F.relu(self._l3(self._l2_out))
        self._out = self._l4(self._l3_out)
        self._out_log_dev = self._l41(self._l3_out)

        return self._out, self._out_log_dev

