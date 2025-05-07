import torch
from torch import nn

class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiScaleConv, self).__init__()
        self.layer1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1)
        self.layer2 = nn.Conv2d(in_ch, out_ch, kernel_size = 5, padding = 2)
        self.layer3 = nn.Conv2d(in_ch, out_ch, kernel_size = 7, padding = 3)

        self.bn = nn.BatchNorm2d(out_ch)

        self.activate = nn.ReLU(inplace=False)


    def forward(self, x):

        l1 = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(x)

        out = (l1 + l2 + l3) / 3

        bn_out = self.bn(out)
        ac_out = self.activate(bn_out)

        return ac_out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace=False),

            MultiScaleConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),

            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),

            MultiScaleConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),

            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),

            MultiScaleConv(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),

            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size = 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1, kernel_size = 1)

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
    
if __name__ == "__main__":

    net_D = Discriminator()



        



