import torch
import torch.nn as nn
import torch.nn.init as init

# TODO Fix issue of loss increasing dramatically
class RNet(nn.Module):
    def __init__(self, upscale_factor):
        super(RNet, self).__init__()

        # Layers for input
        self.convLowFirst = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convLowSecond = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convLowThird = nn.Conv2d(in_channels=16, out_channels=(upscale_factor*2)**2, kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 1
        self.convInt1First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt1Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt1Third = nn.Conv2d(in_channels=16, out_channels=(upscale_factor*2)**2, kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 2
        self.convInt2First = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt2Second = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt2Third = nn.Conv2d(in_channels=16, out_channels=(upscale_factor*2)**2, kernel_size=1, stride=1, padding=0)
        #Other needed declarations
        self.relu = nn.LeakyReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor*2)
        self._initialize_weights()

    def forward(self, x, i1, i2):
        # First layer
        i2 = self.relu(self.convInt2First(i2))
        self.convInt1First.weight.data.add_(self.convInt2First.weight.data)
        i1 = self.relu(self.convInt1First(i1))
        self.convLowFirst.weight.data.add_(self.convInt1First.weight.data)
        x = self.relu(self.convLowFirst(x))

        # Second layer
        i2 = self.relu(self.convInt2Second(i2))
        self.convInt1Second.weight.data.add_(self.convInt2Second.weight.data)
        i1 = self.relu(self.convInt1Second(i1))
        self.convLowSecond.weight.data.add_(self.convInt1Second.weight.data)
        x = self.relu(self.convLowSecond(x))

        # Third layer
        i2 = self.relu(self.convInt2Third(i2))
        self.convInt1Third.weight.data.add_(self.convInt2Third.weight.data)
        i1 = self.relu(self.convInt1Third(i1))
        self.convLowThird.weight.data.add_(self.convInt1Third.weight.data)
        x = self.relu(self.convLowThird(x))

        # Subpixel layer
        i2 = self.pixel_shuffle(i2)
        i1 = self.pixel_shuffle(i1)
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