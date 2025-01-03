import torch
import torch.nn as nn


class BottleneckResidualBlock(nn.Module):
  def __init__(self, in_channels, middle_channels, stride):
    super().__init__()
    expansion = 4
    self.stride = stride

    self.Fx = nn.Sequential(
      nn.Conv2d(in_channels, middle_channels, kernel_size=1),
      nn.BatchNorm2d(middle_channels),
      nn.ReLU(),
      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride=stride, padding=1),
      nn.BatchNorm2d(middle_channels),
      nn.ReLU(),
      nn.Conv2d(middle_channels, middle_channels*expansion, kernel_size=1),
      nn.BatchNorm2d(middle_channels*expansion),
    )
    self.NonLinearity = nn.ReLU()
  
  def forward(self, x):
    Fx = self.Fx(x)

    if self.stride == 2:
      x = nn.functional.interpolate(x, scale_factor=0.5)

    channels_difference = abs(x.shape[1] - Fx.shape[1])
    if channels_difference != 0:
        width, height = Fx.shape[2], Fx.shape[3]
        batch_size = x.shape[0]
        zero_channels = torch.zeros((batch_size, channels_difference, width, height)).to("cuda")
        x = torch.cat([x, zero_channels], dim=1)
    Hx = torch.add(x, Fx)
    Hx = self.NonLinearity(Hx)
    return Hx

class ResNet50(nn.Module):
  def __init__(self, in_channels):
    super().__init__()

    self.ConvBlock = nn.Sequential(nn.Conv2d(in_channels, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU())
    self.MaxPooling = nn.MaxPool2d(3, 2, 1)

    ## bottleneck  go here
    self.Block1 = nn.ModuleList([
      BottleneckResidualBlock(64, 64, 1),
      BottleneckResidualBlock(256, 64, 1),
      BottleneckResidualBlock(256, 64, 1),
    ])

    self.Block2 = nn.ModuleList([
      BottleneckResidualBlock(256, 128, 2),
      *[BottleneckResidualBlock(512, 128, 1) for _ in range(3)]
    ])

    self.Block3 = nn.ModuleList([
      BottleneckResidualBlock(512, 256, 2),
      *[BottleneckResidualBlock(1024, 256, 1) for _ in range(5)]
    ])

    self.Block4 = nn.ModuleList([
      BottleneckResidualBlock(1024, 512, 2),
      *[BottleneckResidualBlock(2048, 512, 1) for _ in range(2)]
    ])
    self.AvgPooling = nn.AvgPool2d(7)
    self.Flatten = nn.Flatten()
    self.FullyConnected = nn.Linear(2048, 1)
    self.Output = nn.Sigmoid()
  
  def forward(self, x):
    x = self.ConvBlock(x)
    x = self.MaxPooling(x)
    for residualBlock in self.Block1:
      x = residualBlock(x)
    for residualBlock in self.Block2:
      x = residualBlock(x)
    for residualBlock in self.Block3:
      x = residualBlock(x)
    for residualBlock in self.Block4:
      x = residualBlock(x)

    x = self.AvgPooling(x)
    x = self.Flatten(x)
    x = self.FullyConnected(x)
    x = self.Output(x)

    return x
