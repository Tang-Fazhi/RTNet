import copy

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(123456)
np.random.seed(0)


class MFDNN(nn.Module):
    #               2          1    【线性】
    # 1  20  20  1                  1
    #               2  10  10  1    【非线性】
    def __init__(self):
        super(MFDNN, self).__init__()
        self.net_low = nn.Sequential()
        self.net_low.add_module('layer_1', nn.Linear(1, 20))
        self.net_low.add_module('layer_2', nn.Tanh())
        self.net_low.add_module('layer_3', nn.Linear(20, 20))
        self.net_low.add_module('layer_4', nn.Tanh())
        self.net_low.add_module('layer_5', nn.Linear(20, 1))

        self.net_high_non_linear = nn.Sequential()
        self.net_high_non_linear.add_module('layer_1', nn.Linear(2, 10))
        self.net_high_non_linear.add_module('layer_2', nn.Tanh())
        self.net_high_non_linear.add_module('layer_3', nn.Linear(10, 10))
        self.net_high_non_linear.add_module('layer_4', nn.Tanh())
        self.net_high_non_linear.add_module('layer_5', nn.Linear(10, 1))

        self.net_high_linear = nn.Sequential()
        self.net_high_linear.add_module('layer_1', nn.Linear(2, 1))

        self.alpha=nn.Parameter(torch.tensor([0.5]))

    def forward(self, x_low, x_high):
        y_low = self.net_low(x_low)
        y_high_through_lownet = self.net_low(x_high)
        y_high_nonlinear = self.net_high_non_linear(torch.cat((y_high_through_lownet, x_high), dim=1))
        y_high_linear = self.net_high_linear(torch.cat((y_high_through_lownet, x_high), dim=1))
        y_high = self.alpha * y_high_linear + (1 - self.alpha) * y_high_nonlinear
        return y_low,y_high


class MFDNN2(nn.Module):
    def __init__(self):
        super(MFDNN2, self).__init__()
        self.net_low = nn.Sequential()
        self.net_low.add_module('layer_1', nn.Linear(2, 50))
        self.net_low.add_module('layer_11', nn.Tanh())
        self.net_low.add_module('layer_2', nn.Linear(50, 50))
        self.net_low.add_module('layer_22', nn.Tanh())
        self.net_low.add_module('layer_3', nn.Linear(50, 50))
        self.net_low.add_module('layer_33', nn.Tanh())
        self.net_low.add_module('layer_4', nn.Linear(50, 50))
        self.net_low.add_module('layer_44', nn.Tanh())
        self.net_low.add_module('layer_5', nn.Linear(50, 50))
        self.net_low.add_module('layer_55', nn.Tanh())
        self.net_low.add_module('layer_6', nn.Linear(50, 2))

        self.net_high_non_linear = nn.Sequential()
        self.net_high_non_linear.add_module('layer_1', nn.Linear(4, 20))
        self.net_high_non_linear.add_module('layer_11', nn.Tanh())
        self.net_high_non_linear.add_module('layer_2', nn.Linear(20, 20))
        self.net_high_non_linear.add_module('layer_22', nn.Tanh())
        self.net_high_non_linear.add_module('layer_3', nn.Linear(20, 20))
        self.net_high_non_linear.add_module('layer_33', nn.Tanh())
        self.net_high_non_linear.add_module('layer_4', nn.Linear(20, 20))
        self.net_high_non_linear.add_module('layer_44', nn.Tanh())
        self.net_high_non_linear.add_module('layer_5', nn.Linear(20, 20))
        self.net_high_non_linear.add_module('layer_55', nn.Tanh())
        self.net_high_non_linear.add_module('layer_6', nn.Linear(20, 20))
        self.net_high_non_linear.add_module('layer_66', nn.Tanh())
        self.net_high_non_linear.add_module('layer_7', nn.Linear(20, 20))
        self.net_high_non_linear.add_module('layer_77', nn.Tanh())
        self.net_high_non_linear.add_module('layer_8', nn.Linear(20, 20))
        self.net_high_non_linear.add_module('layer_88', nn.Tanh())
        self.net_high_non_linear.add_module('layer_9', nn.Linear(20, 2))

        self.net_high_linear = nn.Sequential()
        self.net_high_linear.add_module('layer_1', nn.Linear(4, 100))
        self.net_high_linear.add_module('layer_2', nn.Linear(100, 100))
        self.net_high_linear.add_module('layer_3', nn.Linear(100, 100))
        
        # self.net_high_linear.add_module('layer_4', nn.Linear(100, 100))
        # self.net_high_linear.add_module('layer_51', nn.Linear(200, 200))
        # self.net_high_linear.add_module('layer_52', nn.Linear(200, 200))
        # self.net_high_linear.add_module('layer_53', nn.Linear(200, 200))
        # self.net_high_linear.add_module('layer_54', nn.Linear(200, 200))
        self.net_high_linear.add_module('layer_6', nn.Linear(100, 2))
        #
        self.alpha = nn.Parameter(torch.tensor([0.5], requires_grad=True))

    def forward(self, x_low, x_high):
        y_low = self.net_low(x_low)
        y_high_through_lownet = self.net_low(x_high)
        y_high_nonlinear = self.net_high_non_linear(torch.cat((y_high_through_lownet, x_high), dim=1))
        y_high_linear = self.net_high_linear(torch.cat((y_high_through_lownet, x_high), dim=1))
        y_high = self.alpha * y_high_linear + (1 - self.alpha) * y_high_nonlinear
        return y_low, y_high

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

class ThreeLevelMFDNN(nn.Module):
    def __init__(self):
        super(ThreeLevelMFDNN,self).__init__()

        #low net
        self.low_net =NonLinearNet(1,1,20,4)
        #mid net
        self.mid_linear_net =LinearNet(2,1,10,2)
        self.mid_nonlinear_net =NonLinearNet(2,1,20,4)
        #high net
        self.high_linear_net = LinearNet(3,1,30,2)
        self.high_nonlinear_net = NonLinearNet(3,1,30,4)
        #hyper parameters
        self.alpha=nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self,x_low,x_mid,x_high):
        #y low pred
        y_low=self.low_net(x_low)

        #y mid pred
        y_mid_by_lownet=self.low_net(x_mid)
        mid_net_input=torch.cat((y_mid_by_lownet,x_mid),dim=1)
        y_mid_linear=self.mid_linear_net(mid_net_input)
        y_mid_nonlinear=self.mid_nonlinear_net(mid_net_input)
        y_mid=self.alpha*y_mid_linear + (1-self.alpha)*y_mid_nonlinear

        #y high pred
        y_high_by_lownet=self.low_net(x_high)
        mid_net_input_high = torch.cat((y_high_by_lownet, x_high), dim=1)
        y_high_by_mid_linear = self.mid_linear_net(mid_net_input_high)
        y_high_by_mid_nonlinear = self.mid_nonlinear_net(mid_net_input_high)
        y_high_by_midnet = self.alpha * y_high_by_mid_linear + (1 - self.alpha) * y_high_by_mid_nonlinear

        high_net_input=torch.cat((y_high_by_lownet,y_high_by_midnet,x_high),dim=1)
        # high_net_input=torch.cat((y_high_by_midnet,x_high),dim=1)

        y_high=self.beta*self.high_linear_net(high_net_input) + (1-self.beta)*self.high_nonlinear_net(high_net_input)

        return y_low,y_mid,y_high
        pass


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

    def forward(self,x_low,x_high):
        y_low=self.low_net(x_low)
        y_high_by_lownet=self.low_net(x_high)
        highnet_input=torch.cat((y_high_by_lownet,x_high),dim=1)
        y_high_linear=self.high_linear_net(highnet_input)
        y_high_nonlinear=self.high_nonlinear_net(highnet_input)
        y_high=self.alpha*y_high_linear+(1-self.alpha)*y_high_nonlinear
        return y_low,y_high


class LModel(nn.Module):
    def __init__(self):
        super(LModel, self).__init__()
        self.lmodel = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 2),
        )
        for m in self.modules():
            for name, parameter in m.named_parameters():
                if name == 'weight':
                    torch.nn.init.kaiming_normal_(parameter)
                elif name == 'bias':
                    torch.nn.init.zeros_(parameter)
                else:
                    assert "impossible!"

    def forward(self, x):
        y = self.lmodel(x)
        return y