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
        # Layers for intermediate 2
        self.convInt2First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt2Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt2Third = nn.Conv2d(in_channels=16, out_channels=int((upscale_factor*2)**2), kernel_size=1, stride=1, padding=0)
        # Other needed declarations
        self._initialize_weights()
        self.pixel_shuffle = nn.PixelShuffle(int(upscale_factor * 2))
        self.relu = nn.LeakyReLU()
        # Downsample layer
        self.downsampleLowFirst= Interpolate(size=(256, 256), mode='bilinear')
        self.downsampleLowSecond = Interpolate(size=(128, 128), mode='bilinear')
        self.downsampleIntFirst = Interpolate(size=(512, 512), mode='bilinear')
        self.downsampleIntSecond = Interpolate(size=(256, 256), mode='bilinear')


    def forward(self, x, i1, i2):
        # Operations on first layers
        # Convolve on intermediate 2 input
        i2 = self.relu(self.convInt2First(i2))
        # Perform relu on output of int2
        # Downsample it to match the size of intermdeiate 1
        i2rec = self.downsampleIntFirst(self.relu(i2))
        # Convolve on i1, add the recurrent output of i2, perform relu
        i1 =  self.relu(self.convInt1First(i1)+i2rec)
        # Perform relu on output of int1
        # Downsample it to match the size of low
        i1rec = self.downsampleLowFirst(self.relu(i1))
        # Downsample recurrent output of i2 to match low size
        i2rec = self.downsampleLowFirst(i2rec)
        # Convolve on low, add the recurrent output of i1 and i2, perform relu
        x = self.relu(self.convLowFirst(x) + i1rec + i2rec)


        # Operations on second layers
        i2 = self.relu(self.convInt2Second(i2))
        i2rec = self.downsampleIntSecond(self.relu(i2))
        i1 =  self.relu(self.convInt1Second(i1) + i2rec)
        i1rec = self.downsampleLowSecond(self.relu(i1))
        i2rec = self.downsampleLowSecond(i2rec)
        x = self.relu(self.convLowSecond(x) + i1rec + i2rec)


        # Operations on third layers
        i2 = self.relu(self.convInt2Third(i2))
        i2rec = self.downsampleIntSecond(self.relu(i2))
        i1 = self.relu(self.convInt1Third(i1) + i2rec)
        i1rec = self.downsampleLowSecond(self.relu(i1))
        i2rec = self.downsampleLowSecond(i2rec)
        x = self.relu(self.convLowThird(x) + i1rec + i2rec)


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