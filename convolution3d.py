import torch.nn as nn
import torch
import os
class Attentionmap(nn.Module):
    def __init__(self,ch):
        super(Attentionmap, self).__init__()
        self.ch = ch
        self.conv1 = nn.Conv3d(ch, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=False)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(64, ch, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=False)

    def forward(self, x0): # NIR(T)
        x1 = x0

        b,c,h,w = x1.shape
        x0 = x0.unsqueeze(2)
        # x0 = torch.cat([x0,x0,x0,x0,x0,x0,x0,x0],dim=2)
        # print(x0.shape)

        # print(x0.shape,'x0')
        # x1 = x1.unsqueeze(2)
        # x2 = x2.unsqueeze(2)
        # y0 = torch.cat((x0,x1,x2),1)
        y0 = x0
        y2_1 = self.conv1(y0)
        y2_2 = self.relu1(y2_1)
        y3_1 = self.conv2(y2_2)
        y3_2 = self.relu2(y3_1)
        y3_3 = self.conv3(y3_2)
        # print(y3_3.shape)

        y4 = y3_3.view(-1, self.ch, h, w)
        y5 = y4 + x1
        return y5


class Attentionmap_3dencoder(nn.Module):
    def __init__(self):
        super(Attentionmap_3dencoder, self).__init__()

        self.conv1 = nn.Conv3d(2, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2_2 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.relu = nn.ReLU()

    def forward(self, x0, x1, x2): # NIR(T)

        x0 = x0.unsqueeze(2)
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)

        y0 = torch.cat((x0,x1),1)
        y1 = torch.cat((x1,x2),1)

        x = self.relu(self.conv1(y1))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1, 128, 32, 32)

        return x