import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class RNet(nn.Module):
    def __init__(self, upscale_factor):
        super(RNet, self).__init__()

        # Layers for input
        self.convLowFirst = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convLowSecond = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convLowThird = nn.Conv2d(in_channels=16, out_channels=int(upscale_factor*2), kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 1
        self.convInt1First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt1Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt1Third = nn.Conv2d(in_channels=16, out_channels=int(upscale_factor*2), kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 2
        self.convInt2First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt2Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt2Third = nn.Conv2d(in_channels=16, out_channels=int(upscale_factor*2), kernel_size=1, stride=1, padding=0)
        # Other needed declarations
        self.relu = nn.LeakyReLU()
        self.pixel_shuffle = nn.PixelShuffle(int(upscale_factor/4))
        self._initialize_weights()
        # Upsample layer
        self.upsample = Interpolate(size=(512, 512), mode='bilinear')

    def forward(self, x, i1, i2):
        # First layer
        i2c = self.convInt2First(i2)
        i2 = self.relu(i2c)
        i1c = self.convInt1First(i1)
        i1 = self.relu(self.upsample(i1c) + self.upsample(i2c))
        xc = self.convLowFirst(x)
        x = self.relu(self.upsample(xc) + self.upsample(i1c) + self.upsample(i2c))

        # Second layer
        i2c = self.convInt2Second(i2)
        i2 = self.relu(i2c)
        i1c = self.convInt1Second(i1)
        i1 = self.relu(self.upsample(i1c) + self.upsample(i2c))
        xc = self.convLowSecond(x)
        x = self.relu(self.upsample(xc) + self.upsample(i1c) + self.upsample(i2c))

        # Third layer
        i2c = self.convInt2Third(i2)
        i2 = self.relu(i2c)
        i1c = self.convInt1Third(i1)
        i1 = self.relu(self.upsample(i1c) + self.upsample(i2c))
        xc = self.convLowThird(x)
        x = self.relu(self.upsample(xc) + self.upsample(i1c) + self.upsample(i2c))

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