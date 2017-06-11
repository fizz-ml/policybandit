import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bayestorch.models import Linear

class SimpleHallucinator(ParameterGroup):
    """
    Outputs probability over actions.
    Defines custom model.
    Inherits from torch.nn.Module
    """
    def __init__(self, n, s_size, noise_dim):
        self._noise_dim = noise_dim

        # Generate n sampled states
        self._s_size = s_size
        self._dim_output = n * s_size


        SIZE_H1 = 10 * n
        SIZE_H2 = 10 * n
        SIZE_H3 = 10 * n

        std_dev = 1
        '''Initialize net layers'''
        self._l1 = Linear(self._noise_dim, SIZE_H1, std_dev)
        self._l2 = Linear(SIZE_H1, SIZE_H2, std_dev)
        self._l3 = Linear(SIZE_H2, SIZE_H3, std_dev)
        self._l4 = Linear( SIZE_H3, self._dim_output, std_dev)

        param_dict = {'l1':self._l1, 'l2':self._l2, 'l3':self._l3, 'l4':self._l4}

        super(SimpleCritic, self).__init__(param_dict)

    def forward(self, noise, batch_size):
        ''' noise must be a batch_size x noise_dim Variable
        '''
        x = noise
        self._l1_out = F.relu(self._l1(x))
        self._l2_out = F.relu(self._l2(self._l1_out))
        self._l3_out = F.relu(self._l3(self._l2_out))
        self._out = self._l4(self._l3_out)
        self._out.view(noise.size(0), n, self._s_size)
        return self._out

