import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class RNet(nn.Module):
    def __init__(self, upscale_factor, full_size):
        super(RNet, self).__init__()

        # Siamese layers
        self.convFirst = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convSecond = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.convThird = nn.Conv2d(in_channels=32, out_channels=upscale_factor**2, kernel_size=1, stride=1, padding=0)
        # Other needed declarations
        self._initialize_weights()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.LeakyReLU()
        # Downsample layer
        self.downsampleLow = Interpolate(size=(int(full_size*0.25), int(full_size*0.25)), mode='bilinear')
        self.downsampleInt = Interpolate(size=(int(full_size*0.5), int(full_size*0.5)), mode='bilinear')

    def forward(self, x, i1, i2):
        # Operations on first layers
        # Convolve on intermediate 2 input
        i2 = self.relu(self.convFirst(i2))
        # Perform relu on output of int2
        # Downsample it to match the size of intermdeiate 1
        i2rec = self.downsampleInt(self.relu(i2))
        # Convolve on i1, add the recurrent output of i2, perform relu
        i1 =  self.relu(self.convFirst(i1)+i2rec)
        # Perform relu on output of int1
        # Downsample it to match the size of low
        i1rec = self.downsampleLow(self.relu(i1))
        # Downsample recurrent output of i2 to match low size
        i2rec = self.downsampleLow(i2rec)
        # Convolve on low, add the recurrent output of i1 and i2, perform relu
        x = self.relu(self.convFirst(x) + i1rec + i2rec)


        # Operations on second layers
        i2 = self.relu(self.convSecond(i2))
        i2rec = self.downsampleInt(self.relu(i2))
        i1 =  self.relu(self.convSecond(i1) + i2rec)
        i1rec = self.downsampleLow(self.relu(i1))
        i2rec = self.downsampleLow(i2rec)
        x = self.relu(self.convSecond(x) + i1rec + i2rec)


        # Operations on third layers
        i2 = self.relu(self.convThird(i2))
        i2rec = self.downsampleInt(self.relu(i2))
        i1 = self.relu(self.convThird(i1) + i2rec)
        i1rec = self.downsampleLow(self.relu(i1))
        i2rec = self.downsampleLow(i2rec)
        x = self.relu(self.convThird(x) + i1rec + i2rec)


        # Subpixel layer
        i2 = self.downsampleLow(self.relu(i2))
        i2 = self.pixel_shuffle(i2)

        return i2

    def _initialize_weights(self):
        init.orthogonal_(self.convFirst.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convSecond.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convThird.weight, init.calculate_gain('leaky_relu'))


# Class for upscaling or downscaling images while being fed through the network
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x