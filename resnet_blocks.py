import torch.nn as nn
import torch.nn.functional as F


class IdentityLayer(nn.Module):

    def forward(self, x):
        return x


class ResNetBlock(nn.Module):

    def __init__(self, in_size=16, out_size=16, downsample = False, my=0, my1=0):
        super(ResNetBlock,self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        # if out_size == in_size:
        #   self.message_size = (out_size//16)**2*2
        # else:
        #   self.message_size = (out_size//16)**2 + out_size//16*in_size//16
        # print(self.message_size)
        self.message_size = my[0]*my[1]+my1[0]*my1[1]
        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()

        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        size_mapping_dict = {16:16*32*32, 32:32*16*16, 64:64*8*8}
        self.out_flatten_size = size_mapping_dict[out_size]
        self.message_layer = nn.Linear(self.out_flatten_size, self.message_size*64)

    def forward(self, x, conv1_w, conv2_w):

        residual = self.reslayer(x)

        out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=self.stride1, padding=1)), inplace=True)
        out = self.bn2(F.conv2d(out, conv2_w, padding=1))

        out += residual

        out = F.relu(out)
        message = self.message_layer(out.view(-1, self.out_flatten_size))

        return out, message
