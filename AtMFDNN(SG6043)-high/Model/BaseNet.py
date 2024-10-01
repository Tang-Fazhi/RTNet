import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(123456)
np.random.seed(0)


class LinearNet(nn.Module):
    def __init__(self,in_N,out_N,width,depth):
        super(LinearNet, self).__init__()
        self.in_N = in_N
        self.width = width
        self.depth = depth
        self.out_N = out_N

        self.net= nn.Sequential()
        #input layer
        self.net.add_module('layer_in', nn.Linear(self.in_N, self.width))
        #hidden layers
        for i in range(self.depth):
            self.net.add_module('layer_'+str(i+2), nn.Linear(self.width, self.width))
        #output layer
        self.net.add_module('layer_out', nn.Linear(self.width, self.out_N))

    def forward(self,x):
        x=self.net(x)
        return x

class NonLinearNet(nn.Module):
    def __init__(self,in_N,out_N,width,depth):
        super(NonLinearNet, self).__init__()
        self.in_N = in_N
        self.width = width
        self.depth = depth
        self.out_N = out_N

        self.net = nn.Sequential()
        # input layer
        self.net.add_module('layer_in', nn.Linear(self.in_N, self.width))
        self.net.add_module('activate_layer_in', nn.Tanh())

        # hidden layers
        for i in range(self.depth):
            self.net.add_module('layer_' + str(i + 2), nn.Linear(self.width, self.width))
            self.net.add_module('activate_layer_'+str(i+2), nn.Tanh())
        # output layer
        self.net.add_module('layer_out', nn.Linear(self.width, self.out_N))
    def forward(self,x):
        x=self.net(x)
        return x
