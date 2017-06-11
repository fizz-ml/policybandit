import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Evaluator:
    ''' Evaluate policy at end of an episode
    '''
