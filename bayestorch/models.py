from .hmc import *
import torch as t
import torch.nn.functional as f

class Linear(ParameterGroup):
    def __init__(self, d_in, d_out, std_dev):
        self.w = TensorParameter((d_out, d_in), std_dev)
        self.b = TensorParameter((d_out), std_dev)
        super(Linear, self).__init__({'w': self.w, 'b' : self.b})

    def forward(self, input):
        return f.linear(input, self.w.val(), self.b.val())

    def __call__(self, input):
        return self.forward(input)

