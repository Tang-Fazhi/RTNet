import torch
import torch.nn as nn
import numpy as np

from Model.BaseNet import LinearNet,NonLinearNet

torch.manual_seed(123456)
np.random.seed(0)


class BaseMFDNN(nn.Module):
    def __init__(self,model_structure):
        super(BaseMFDNN,self).__init__()
        self.lownet_neure_num=model_structure[0]
        self.lownet_depth=model_structure[1]
        self.hlnet_neure_num=model_structure[2]
        self.hlnet_depth=model_structure[3]
        self.hnlnet_neure_num=model_structure[4]
        self.hnlnet_depth=model_structure[5]
        # low net
        self.low_net = NonLinearNet(1, 1, self.lownet_neure_num, self.lownet_depth)
        # high net
        self.high_linear_net = LinearNet(2, 1, self.hlnet_neure_num, self.hlnet_depth)
        self.high_nonlinear_net = NonLinearNet(2, 1, self.hnlnet_neure_num, self.hnlnet_depth)
        # hyper parameters
        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self,x_low,x_high,c32):
        if c32: #只用到后半网络
            highnet_input = torch.cat((x_low, x_high), dim=1)
            y_high_linear = self.high_linear_net(highnet_input)
            y_high_nonlinear = self.high_nonlinear_net(highnet_input)
            y_high = self.alpha * y_high_linear + (1 - self.alpha) * y_high_nonlinear
            return y_high
        else:
            y_low=self.low_net(x_low)
            y_high_by_lownet=self.low_net(x_high)
            highnet_input=torch.cat((y_high_by_lownet,x_high),dim=1)
            y_high_linear=self.high_linear_net(highnet_input)
            y_high_nonlinear=self.high_nonlinear_net(highnet_input)
            y_high=self.alpha*y_high_linear+(1-self.alpha)*y_high_nonlinear
            return y_low,y_high
