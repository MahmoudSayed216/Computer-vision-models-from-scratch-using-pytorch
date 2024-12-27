import torch
import torch.nn as nn
import torchvision.transforms as T

print("Hello from the MahmoudNet community")


class _conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, a):
        super().__init__()
        self.a = a

        out_channels = int(a*out_channels)
        self.cb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cb(x)
        return x


class _depthwise_separable_convolution(nn.Module):
    def __init__(self, in_channels, out_channels, stride, a):
        super().__init__()
        self.a = a

        out_channels = int(a*out_channels)
        in_channels = int(a*in_channels)
        self.dwsc_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        

    def forward(self, x):
        x = self.dwsc_layer(x)

        return x

        
class MobileNet(nn.Module):
    def __init__(self, input_channels: int, depth_multiplier: float, resolution_multiplier: float):
        super().__init__()
        self.base_model_layers_sizes = [32, 64, 128, 128, 256, 256, *[512 for i in range(5)], 512, 1024]
        self.strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2]
        self.n_layers = 12
        self.a = depth_multiplier
        self.input_size = int(224 * resolution_multiplier)
        self.input_channels = input_channels
        # self.n_classes = n_classes
        self.dwsc = []
        flattened_size = int(1024*self.a)
        self.ConvBlock1 = _conv_block(self.input_channels, self.base_model_layers_sizes[0], self.a)
        self.dwsc = nn.ModuleList([
            _depthwise_separable_convolution(
                self.base_model_layers_sizes[i], 
                self.base_model_layers_sizes[i+1], 
                self.strides[i], 
                self.a,
            ) for i in range(self.n_layers)
        ])
        self.Pool = nn.AdaptiveAvgPool2d(1)
        self.Flattener = nn.Flatten()
        self.Linear1 = nn.Linear(flattened_size, 1000)
        self.Linear2 = nn.Linear(1000, 1)
        self.Output = nn.Sigmoid()
        self.resizer = T.transforms.Resize((self.input_size, self.input_size))

    def forward(self, x):
        x = self.resizer(x)
        x = self.ConvBlock1(x)
        for i in range(self.n_layers):
            x = self.dwsc[i](x)
        x = self.Pool(x)
        x = self.Flattener(x)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Output(x)
        return x
