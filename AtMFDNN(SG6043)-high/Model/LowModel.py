import torch
import torch.nn as nn
import numpy as np

from Model.BaseNet import LinearNet,NonLinearNet

torch.manual_seed(123456)
np.random.seed(0)



class LowModel(nn.Module):
    def __init__(self,in_N,in_W,width,depth,out_W,out_N):
        super(LowModel, self).__init__()

        self.net=nn.Sequential()
        # input layer
        self.net.add_module('layer_in', nn.Linear(in_N, in_W))
        self.net.add_module('activate_layer_in', nn.Tanh())
        self.net.add_module('layer_in_1', nn.Linear(in_W, width))
        self.net.add_module('activate_layer_in_1', nn.Tanh())
        # hidden layers
        for i in range(depth):
            self.net.add_module('layer_' + str(i + 2), nn.Linear(width, width))
            self.net.add_module('activate_layer_' + str(i + 2), nn.Tanh())
        # output layer
        self.net.add_module('layer_out', nn.Linear(width, out_W))
        self.net.add_module('activate_layer_out', nn.Tanh())
        self.net.add_module('layer_out_1', nn.Linear(out_W, out_N))

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == '__main__':
    model=LowModel(1,10,20,4,10,1)
    print(model)