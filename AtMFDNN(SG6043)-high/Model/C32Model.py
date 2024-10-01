import torch
import torch.nn as nn
torch.manual_seed(0)

from Model.BaseNet import NonLinearNet,LinearNet


class C32(nn.Module):
    def __init__(self,ms,alpha_flag=True): #alpha_flag=True:使用超参数聚合，否则，使用线性网络进行聚合
        super(C32, self).__init__()
        self.flag=alpha_flag
        self.in_size=ms[0]
        self.width=ms[1]
        self.depth=ms[2]
        self.out_size=ms[3]
        self.net=LinearNet(self.in_size,self.out_size,self.width,self.depth)
        self.alpha=nn.Parameter(torch.tensor([0.5]))

    def forward(self,x,y1,y2):
        input=torch.cat((x,y1,y2),dim=1)
        y=self.net(input)
        return y