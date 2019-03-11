import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class RNet(nn.Module):
    def __init__(self, upscale_factor, full_size):
        super(RNet, self).__init__()

        # Layers for input
        self.convLowFirst = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convLowSecond = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.convLowThird = nn.Conv2d(in_channels=16, out_channels=upscale_factor**2, kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 1
        self.convInt1First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt1Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.convInt1Third = nn.Conv2d(in_channels=16, out_channels=upscale_factor**2, kernel_size=1, stride=1, padding=0)
        self.convInt1Fourth = nn.Conv2d(in_channels=upscale_factor**2, out_channels=(int(upscale_factor/2)**2), kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 2
        self.convInt2First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt2Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.convInt2Third = nn.Conv2d(in_channels=16, out_channels=upscale_factor**2, kernel_size=1, stride=1, padding=0)
        self.convInt2Fourth = nn.Conv2d(in_channels=upscale_factor**2, out_channels=(int(upscale_factor/4)**2), kernel_size=1, stride=1, padding=0)
        # Residual layers
        self.resConv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.resConv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resBN = nn.BatchNorm2d(64, eps=0.0001, momentum = 0.95)
        # Other needed declarations
        self._initialize_weights()
        self.subpixel_int2 = nn.PixelShuffle(int(upscale_factor / 4))
        self.subpixel_int1 = nn.PixelShuffle(int(upscale_factor / 2))
        self.subpixel_low = nn.PixelShuffle(upscale_factor)
        self.relu = nn.LeakyReLU()
        self.resrelu = nn.ReLU()
        # Downsample layers
        self.resizeLow = Interpolate(size=(int(full_size / upscale_factor), int(full_size / upscale_factor)), mode='bilinear')
        self.resizeInt1 = Interpolate(size=(int(full_size / (upscale_factor / 2)), int(full_size / (upscale_factor / 2))), mode='bilinear')
        self.resizeInt2 = Interpolate(size=(int(full_size / (upscale_factor / 4)), int(full_size / (upscale_factor / 4))), mode='bilinear')


    def forward(self, x, i1, i2):
        # Operations on residual layers
        i1Down =  self.resizeLow(i1)
        xRes = i1Down - x
        xRes = self.resrelu(self.resConv1(xRes))
        xRes = self.resrelu(self.resBN(self.resConv2(xRes)))
        i2Down = self.resizeInt1(i2)
        i1Res = i2Down - i1
        i1Res = self.resrelu(self.resConv1(i1Res))
        i1Res = self.resrelu(self.resBN(self.resConv2(i1Res)))


        # Operations on first layers
        # Convolve on intermediate 2 input
        i2 = self.relu(self.convInt2First(i2))
        # Perform relu on output of int2
        # Downsample it to match the size of intermdeiate 1
        i2rec = self.resizeInt1(self.relu(i2))
        # Convolve on i1, add the recurrent output of i2, perform relu
        i1 =  self.relu(self.convInt1First(i1)+i2rec)
        # Perform relu on output of int1
        # Downsample it to match the size of low
        i1rec = self.resizeLow(self.relu(i1))
        # Downsample recurrent output of i2 to match low size
        i2rec = self.resizeLow(i2rec)
        # Convolve on low, add the recurrent output of i1 and i2, perform relu
        x = self.relu(self.convLowFirst(x) + i1rec + i2rec)


        # Operations on second layers
        i2 = self.relu(self.convInt2Second(i2))
        i2rec = self.resizeInt1(self.relu(i2))
        i1 =  self.relu(self.convInt1Second(i1) + i2rec)
        i1rec = self.resizeLow(self.relu(i1))
        i2rec = self.resizeLow(i2rec)
        x = self.relu(self.convLowSecond(x) + i1rec + i2rec)


        # Operations on third layers
        i2 = self.relu(self.convInt2Third(i2))
        i2rec = self.resizeInt1(self.relu(i2))
        i1 = self.relu(i1Res + self.convInt1Third(i1) + i2rec)
        i1rec = self.resizeLow(self.relu(i1))
        i2rec = self.resizeLow(i2rec)
        x = self.subpixel_low(xRes + self.convLowThird(x) + i1rec + i2rec)


        # Residual addition
        #x = x + xRes
        #i1 = i1 + i1Res


        # Operations on fourth layers
        i2 = self.subpixel_int2(self.convInt2Fourth(i2))
        i1 = self.subpixel_int1(self.convInt1Fourth(i1))


        # Subpixel layer
        #i2 = self.subpixel_int2(i2)
        #i1 = self.subpixel_int1(i1)
        #x = self.subpixel_low(x)

        return i2, i1, x


    def _initialize_weights(self):
        init.orthogonal_(self.convLowFirst.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convLowSecond.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convLowThird.weight, init.calculate_gain('leaky_relu'))

        init.orthogonal_(self.convInt1First.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt1Second.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt1Third.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt1Fourth.weight, init.calculate_gain('leaky_relu'))

        init.orthogonal_(self.convInt2First.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt2Second.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt2Third.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt2Fourth.weight, init.calculate_gain('leaky_relu'))

        init.orthogonal_(self.resConv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.resConv2.weight, init.calculate_gain('relu'))


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x