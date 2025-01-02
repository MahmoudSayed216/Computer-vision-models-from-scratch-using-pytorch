import torch.nn as nn
import torchvision.transforms  as T
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )



    def forward(self, x):
        return self.conv_block(x)


class ResidualBlock(nn.Module):
    ## CONV BLOCK SHOULD BE USED HERE  
    def __init__(self, in_channels, channels_across_block, stride):
        
        super().__init__()
        ## padding is always 1, to maintain the same size across the CONVs [how to handle that in case of strided conv though]
        self.in_channels = in_channels
        self.channels_across_block = channels_across_block
        self.stride = stride
        self.Fx = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels_across_block, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(channels_across_block),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels_across_block, out_channels=channels_across_block, kernel_size=3,padding=1, stride=stride),
            nn.BatchNorm2d(channels_across_block)
        )

        self.projection = nn.Sequential(nn.Conv2d(self.in_channels, out_channels=channels_across_block, kernel_size=1, stride=2),
                                        nn.BatchNorm2d(channels_across_block)
                                       ) 
        self.non_linearity = nn.ReLU()

    def forward(self, x):
        Fx = self.Fx(x)
        if self.stride == 2:
            x = self.projection(x)
            
        Hx = torch.add(x, Fx)
        Hx = self.non_linearity(Hx)
        return Hx



class ResNet34(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = 3
        self.convBlock = ConvBlock(in_channels, 64, 7, 2)
        self.maxPool = nn.MaxPool2d(3, 2, padding=1)
        
        self.block1 = nn.ModuleList([
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1)
        ])

        self.block2 = nn.ModuleList([
            ResidualBlock(64, 128, 2),
            *[ResidualBlock(128, 128, 1) for _ in range(3)]
            
        ])

        self.block3 = nn.ModuleList([
            ResidualBlock(128, 256, 2),
            *[ResidualBlock(256,256, 1) for _ in range(5)]
            
        ])

        self.block4 = nn.ModuleList([
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512, 1),
            ResidualBlock(512, 512, 1)
        ])

        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Linear(512, 1)
        self.output = nn.Sigmoid()


    def forward(self, x):
        x = self.convBlock(x)
        x = self.maxPool(x)
        for res_block in self.block1:
            x = res_block(x)
        for res_block in self.block2:
            x = res_block(x)
        for res_block in self.block3:
            x = res_block(x)
        for res_block in self.block4:
            x = res_block(x)

        x = self.avgPool(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        x = self.output(x)
        return x