import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class RNet(nn.Module):
    def __init__(self, upscale_factor, full_size):
        super(RNet, self).__init__()

        # Siamese layers
        self.convFirst = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convSecond = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.convThird = nn.Conv2d(in_channels=16, out_channels=upscale_factor**2, kernel_size=1, stride=1, padding=0)
        self.convInt1Fourth = nn.Conv2d(in_channels=upscale_factor ** 2, out_channels=(int(upscale_factor / 2) ** 2), kernel_size=1, stride=1, padding=0)
        self.convInt2Fourth = nn.Conv2d(in_channels=upscale_factor ** 2, out_channels=(int(upscale_factor / 4) ** 2), kernel_size=1, stride=1, padding=0)
        # Residual layers
        self.resConv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.resConv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resBN = nn.BatchNorm2d(64, eps=0.0001, momentum=0.95)
        # Other needed declarations
        self._initialize_weights()
        self.subpixel_int2 = nn.PixelShuffle(int(upscale_factor/4))
        self.subpixel_int1 = nn.PixelShuffle(int(upscale_factor/2))
        self.subpixel_low = nn.PixelShuffle(upscale_factor)
        self.relu = nn.LeakyReLU()
        self.resrelu = nn.ReLU()
        # Downsample layer
        self.resizeLow = Interpolate(size=(int(full_size / upscale_factor), int(full_size / upscale_factor)), mode='bilinear')
        self.resizeInt1 = Interpolate(size=(int(full_size / (upscale_factor / 2)), int(full_size / (upscale_factor / 2))), mode='bilinear')
        self.resizeInt2 = Interpolate(size=(int(full_size / (upscale_factor / 4)), int(full_size / (upscale_factor / 4))), mode='bilinear')

    def forward(self, x, i1, i2):
        # Operations on residual layers
        i1Down = self.resizeLow(i1)
        xRes = i1Down - x
        xRes = self.resrelu(self.resConv1(xRes))
        xRes = self.resrelu(self.resBN(self.resConv2(xRes)))
        i2Down = self.resizeInt1(i2)
        i1Res = i2Down - i1
        i1Res = self.resrelu(self.resConv1(i1Res))
        i1Res = self.resrelu(self.resBN(self.resConv2(i1Res)))


        # Operations on first layers
        # Convolve on intermediate 2 input
        i2 = self.relu(self.convFirst(i2))
        # Perform relu on output of int2
        # Downsample it to match the size of intermdeiate 1
        i2rec = self.resizeInt1(self.relu(i2))
        # Convolve on i1, add the recurrent output of i2, perform relu
        i1 = self.relu(self.convFirst(i1) + i2rec)
        # Perform relu on output of int1
        # Downsample it to match the size of low
        i1rec = self.resizeLow(self.relu(i1))
        # Downsample recurrent output of i2 to match low size
        i2rec = self.resizeLow(i2rec)
        # Convolve on low, add the recurrent output of i1 and i2, perform relu
        x = self.relu(self.convFirst(x) + i1rec + i2rec)

        # Operations on second layers
        i2 = self.relu(self.convSecond(i2))
        i2rec = self.resizeInt1(self.relu(i2))
        i1 = self.relu(self.convSecond(i1) + i2rec)
        i1rec = self.resizeLow(self.relu(i1))
        i2rec = self.resizeLow(i2rec)
        x = self.relu(self.convSecond(x) + i1rec + i2rec)

        # Operations on third layers
        i2 = self.relu(self.convThird(i2))
        i2rec = self.resizeInt1(self.relu(i2))
        i1 = self.relu(i1Res + self.convThird(i1) + i2rec)
        i1rec = self.resizeLow(self.relu(i1))
        i2rec = self.resizeLow(i2rec)
        x = self.subpixel_low(xRes + self.convThird(x) + i1rec + i2rec)


        # Operations on fourth layers
        i2 = self.subpixel_int2(self.convInt2Fourth(i2))
        i1 = self.subpixel_int1(self.convInt1Fourth(i1))


        return i2, i1, x

    def _initialize_weights(self):
        init.orthogonal_(self.convFirst.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convSecond.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convThird.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt1Fourth.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt2Fourth.weight, init.calculate_gain('leaky_relu'))


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