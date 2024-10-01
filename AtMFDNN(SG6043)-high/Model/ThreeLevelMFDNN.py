import torch
import torch.nn as nn

from Model.BaseNet import LinearNet,NonLinearNet
torch.manual_seed(0)

class ThreeLevelMFDNN(nn.Module):
    def __init__(self,in_size,out_size,ms_low,ms_mid,ms_high):
        super(ThreeLevelMFDNN,self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.lownet_width = ms_low[0]
        self.lownet_depth = ms_low[1]
        self.midnet_l_width = ms_mid[0]
        self.midnet_l_depth = ms_mid[1]
        self.midnet_nl_width = ms_mid[2]
        self.midnet_nl_depth = ms_mid[3]
        self.highnet_l_width = ms_high[0]
        self.highnet_l_depth = ms_high[1]
        self.highnet_nl_width = ms_high[2]
        self.highnet_nl_depth = ms_high[3]
        # low net
        self.low_net = NonLinearNet(in_size, out_size, self.lownet_width, self.lownet_depth)
        # mid net
        self.mid_linear_net = LinearNet(in_size + out_size, out_size, self.midnet_l_width, self.midnet_l_depth)
        self.mid_nonlinear_net = NonLinearNet(in_size + out_size, out_size, self.midnet_nl_width, self.midnet_nl_depth)
        # high net
        self.high_linear_net = LinearNet(in_size + out_size * 2, out_size, self.highnet_l_width, self.highnet_l_depth)
        self.high_nonlinear_net = NonLinearNet(in_size + out_size * 2, out_size, self.highnet_nl_width,
                                               self.highnet_nl_depth)
        # merge net
        self.merge_net_mid = LinearNet(2 * self.out_size, self.out_size, 20, 2)
        self.merge_net_high = LinearNet(2 * self.out_size, self.out_size, 20, 2)
        # 以下超参数无作用
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self,x_low,x_mid,x_high):
        #y low pred
        y_low=self.low_net(x_low)

        #y mid pred
        y_mid_by_lownet=self.low_net(x_mid)
        mid_net_input=torch.cat((y_mid_by_lownet,x_mid),dim=1)
        y_mid_linear=self.mid_linear_net(mid_net_input)
        y_mid_nonlinear=self.mid_nonlinear_net(mid_net_input)
        midnet_merge_input=torch.cat((y_mid_linear,y_mid_nonlinear),dim=1)
        y_mid=self.merge_net_mid(midnet_merge_input)

        #y high pred
        y_high_by_lownet=self.low_net(x_high)
        mid_net_input_high = torch.cat((y_high_by_lownet, x_high), dim=1)
        y_high_by_mid_linear = self.mid_linear_net(mid_net_input_high)
        y_high_by_mid_nonlinear = self.mid_nonlinear_net(mid_net_input_high)
        midnet_merge_input_high=torch.cat((y_high_by_mid_linear,y_high_by_mid_nonlinear),dim=1)
        y_high_by_midnet = self.merge_net_mid(midnet_merge_input_high)

        high_net_input=torch.cat((y_high_by_lownet,y_high_by_midnet,x_high),dim=1)
        y_high_linear=self.high_linear_net(high_net_input)
        y_high_nonlinear=self.high_nonlinear_net(high_net_input)
        merged_input_high=torch.cat((y_high_linear,y_high_nonlinear),dim=1)
        y_high=self.merge_net_high(merged_input_high)

        return y_low,y_mid,y_high
