import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hypernetwork_modules import HyperNetwork
from resnet_blocks import ResNetBlock


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim

        h,k = self.z_num

        # for i in range(h):
        #     for j in range(k):
        #         self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))

    def forward(self, hyper_net, z_list):
        # print("shape", z_list.shape)
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(z_list[i*k + j]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.z_dim = z_dim
        self.hope = HyperNetwork(z_dim=self.z_dim)

        self.zs_size = [[1, 1], [1, 1], [2,1],[2,2], [2,2],[2,2],[4,2], [4, 4], [1,1],[1,1]]

        self.filter_size = [[16,16], [16,32], [32,32], [32,64]]

        self.res_net = nn.ModuleList()

        for i in range(4):
            down_sample = False
            if i in [1,3]:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample, my=self.zs_size[2*(i+1)], my1=self.zs_size[2*(i+1)+1]))

        self.zs = nn.ModuleList()

        for i in range(8):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim))
        
        self.break_size = []
        
        for i in range(5):
          a0=self.zs_size[2*i][0]
          a1=self.zs_size[2*i][1]
          b0=self.zs_size[2*i+1][0]
          b1=self.zs_size[2*i+1][1]
          self.break_size.append([a0,a1,b0,b1])

        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,10)

        self.z_list0 = [Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2))]
        self.z_list1 = [Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2))]

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(4):
            # if i != 15 and i != 17:
            print(i)
            w1 = self.zs[2*i](self.hope, self.z_list0)
            w2 = self.zs[2*i+1](self.hope, self.z_list1)
            x, message = self.res_net[i](x, w1, w2)
            print(message.shape)
            a0,a1,b0, b1 = self.break_size[i+1]
            z_list0, z_list1 = torch.split(message, [a0*a1*64,b0*b1*64], dim=-1)
            self.z_list0 = z_list0.view(-1, a0, a1, 64)
            self.z_list1 = z_list1.view(-1, b0, b1, 64)
            # print("resblock ", i, "shape ", x.shape)
        x = self.global_avg(x)
        x = self.final(x.view(-1,64))
        

        return x
