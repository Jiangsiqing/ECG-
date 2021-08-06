from model.basic_conv import conv1d, maxpool1d
import torch.nn as nn
import torch
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            conv1d(in_ch, out_ch, 3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            conv1d(out_ch, out_ch, 3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet_1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_1d, self).__init__()
        self.pad = nn.ConstantPad1d((60, 60), 0.)
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = maxpool1d(4)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = maxpool1d(4)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = maxpool1d(4)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = maxpool1d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose1d(1024, 512, 2, stride=2)
        # self.up6 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose1d(512, 256, 4, stride=4)
        self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.Upsample(scale_factor=4)
        self.up8 = nn.ConvTranspose1d(256, 128, 4, stride=4)
        self.conv8 = DoubleConv(256, 128)
        # self.up9 = nn.Upsample(scale_factor=4)
        self.up9 = nn.ConvTranspose1d(128, 64, 4, stride=4)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = conv1d(64, out_ch, 1)

    def forward(self, x):
        x = self.pad(x)
        c1 = self.conv1(x)  # 5120
        p1 = self.pool1(c1)  # 1280
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)  # 320
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)     # 80
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)     # 40
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        # print(up_6.size())
        # print(c4.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        # print(merge6.size())
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        c10 = c10[:, :, 60:-60]
        out = nn.Softmax(dim=1)(c10)
        return out


if __name__ == '__main__':
    net = Unet_1d(8, 4)
    x = torch.randn(4, 8, 5000)
    y = net(x)
    print(y.shape)
