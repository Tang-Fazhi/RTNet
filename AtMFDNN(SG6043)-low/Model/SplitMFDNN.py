import torch
import torch.nn as nn

from Model.BaseNet import NonLinearNet,LinearNet

torch.manual_seed(0)

class LowNet(nn.Module):
    def __init__(self,ms):
        super(LowNet, self).__init__()
        self.in_N = ms[0]
        self.width = ms[1]
        self.depth = ms[2]
        self.out_N = ms[3]
        self.net=NonLinearNet(self.in_N,self.out_N,self.width,self.depth)

    def forward(self,x):
        x=self.net(x)
        return x


class HighNetV2(nn.Module):
    def __init__(self, in_size, out_size, ms):
        super(HighNetV2, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.l_width = ms[0]
        self.l_depth = ms[1]
        self.nl_width = ms[2]
        self.nl_depth = ms[3]
        self.nl_net = NonLinearNet(self.in_size, self.out_size, self.nl_width, self.nl_depth)
        self.l_net = LinearNet(self.in_size, self.out_size, self.l_width, self.l_depth)
        self.merge_net = LinearNet(2 * self.out_size, self.out_size, 10*2 * self.out_size, 2)
        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x, yl):
        highnet_input = torch.cat((x, yl), dim=1)
        y_high_linear = self.l_net(highnet_input)
        y_high_nonlinear = self.nl_net(highnet_input)
        merged_input = torch.cat((y_high_linear, y_high_nonlinear), dim=1)
        y_high = self.merge_net(merged_input)
        return y_high