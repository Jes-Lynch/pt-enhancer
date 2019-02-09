import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class RNet(nn.Module):
    def __init__(self, upscale_factor):
        super(RNet, self).__init__()

        # Layers for input
        self.convLowFirst = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convLowSecond = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convLowThird = nn.Conv2d(in_channels=16, out_channels=int((upscale_factor*2)**2), kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 1
        self.convInt1First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt1Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt1Third = nn.Conv2d(in_channels=16, out_channels=int((upscale_factor*2)**2), kernel_size=1, stride=1, padding=0)
        self.flattenInt1First = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.flattenInt1Second = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.expandInt1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Layers for intermediate 2
        self.convInt2First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt2Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt2Third = nn.Conv2d(in_channels=16, out_channels=int((upscale_factor*2)**2), kernel_size=1, stride=1, padding=0)
        self.flattenInt2First = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.flattenInt2Second = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.expandInt2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Other needed declarations
        self.relu = nn.LeakyReLU()
        self.pixel_shuffle = nn.PixelShuffle(int(upscale_factor*2))
        self._initialize_weights()
        # Downsample layer
        self.downsampleLowFirst= Interpolate(size=(256, 256), mode='bilinear')
        self.downsampleLowSecond = Interpolate(size=(128, 128), mode='bilinear')
        self.downsampleIntFirst = Interpolate(size=(512, 512), mode='bilinear')
        self.downsampleIntSecond = Interpolate(size=(256, 256), mode='bilinear')


    def forward(self, x, i1, i2):
        # First layer
        i2 = self.relu(self.convInt2First(i2))
        i2c = self.downsampleIntFirst(self.relu(i2))
        # i2c size is ([batch size, 32, 728, 728])
        # Need to reduce that to ([batch size, 1, 728, 728]) to use as input to the first conv layer
        i2c = self.flattenInt2First(i2c)
        # Other idea: Get i2c, don't flatten, instead do
        # i1 =  self.relu(self.convInt1First(i1)+i2c)
        i1 =  self.relu(self.convInt1First(i1 + i2c))
        i1c = self.downsampleLowFirst(self.relu(i1))
        i1c = self.flattenInt1First(i1c)
        i2c = self.downsampleLowFirst(i2c)
        # In this case without the flatten layer this would be
        # x = self.relu(self.convLowFirst(x) + i1c + i2c)
        x = self.relu(self.convLowFirst(x + i1c + i2c))


        # Second layer
        i2 = self.relu(self.convInt2Second(i2))
        i2c = self.downsampleIntFirst(self.relu(i2))
        i2c = self.expandInt2(i2c)
        i1 =  self.relu(self.convInt1Second(i1 + i2c))
        i1c = self.downsampleLowFirst(self.relu(i1))
        i1c = self.expandInt1(i1c)
        i2c = self.downsampleLowFirst(i2c)
        x = self.relu(self.convLowSecond(x + i1c + i2c))


        # Third layer
        i2 = self.relu(self.convInt2Third(i2))
        i2c = self.downsampleIntSecond(self.relu(i2))
        i2c = self.flattenInt2Second(i2c)
        i1 = self.relu(self.convInt1Third(i1 + i2c))
        i1c = self.downsampleLowSecond(self.relu(i1))
        i1c = self.flattenInt1Second(i1c)
        i2c = self.downsampleLowSecond(i2c)
        x = self.relu(self.convLowThird(x + i1c + i2c))


        # Subpixel layer
        x = self.pixel_shuffle(x)

        return x

    def _initialize_weights(self):
        init.orthogonal_(self.convLowFirst.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convLowSecond.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convLowThird.weight, init.calculate_gain('leaky_relu'))

        init.orthogonal_(self.convInt1First.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt1Second.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt1Third.weight, init.calculate_gain('leaky_relu'))

        init.orthogonal_(self.convInt2First.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt2Second.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt2Third.weight, init.calculate_gain('leaky_relu'))


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x