from hmc import *
import torch as t
import torch.nn.functional as f

class Linear(ParameterGroup):
    def __init__(self, d_in, d_out, std_dev):
        self.w = TensorParameter((d_in, d_out), std_dev)
        self.b = TensorParameter((d_out), std_dev)
        super(ParameterGroup, self).__init__({'w': self.w, 'b' : self.b})

    def forward(self, input):
        return f.linear(input, self.w, self.b)

