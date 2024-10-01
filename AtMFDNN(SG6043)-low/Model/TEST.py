import torch
import torch.nn as nn
from Model.BaseNet import NonLinearNet,LinearNet


class OnlyLow(nn.Module):
    def __init__(self,ms):
        super(OnlyLow, self).__init__()
        self.in_N = ms[0]
        self.width = ms[1]
        self.depth = ms[2]
        self.out_N = ms[3]
        self.net=NonLinearNet(self.in_N,self.out_N,self.width,self.depth)
    def forward(self,x):
        x=self.net(x)
        return x

class OnlyHigh(nn.Module):
    def __init__(self,ms):
        super(OnlyHigh, self).__init__()
        self.l_width=ms[0]
        self.l_depth=ms[1]
        self.nl_width = ms[2]
        self.nl_depth = ms[3]
        self.nl_net=NonLinearNet(2,1,self.nl_width,self.nl_depth)
        self.l_net=LinearNet(2,1,self.l_width,self.l_depth)
        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self,x,yl):
        highnet_input = torch.cat((x, yl), dim=1)
        y_high_linear = self.l_net(highnet_input)
        y_high_nonlinear = self.nl_net(highnet_input)
        y_high = self.alpha * y_high_linear + (1 - self.alpha) * y_high_nonlinear
        return y_high
