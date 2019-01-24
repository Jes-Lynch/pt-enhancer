import torch
import torch.nn as nn
import torch.nn.init as init


class RNet(nn.Module):
    def __init__(self, upscale_factor):
        super(RNet, self).__init__()

        # Layers for input
        self.convLow1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convLow2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convLow3 = nn.Conv2d(in_channels=16, out_channels=(upscale_factor*2)**2, kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 1
        self.convInt1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt1_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt1_3 = nn.Conv2d(in_channels=16, out_channels=(upscale_factor*2)**2, kernel_size=1, stride=1, padding=0)
        # Layers for intermediate 2
        self.convInt2_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convInt2_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.convInt2_3 = nn.Conv2d(in_channels=16, out_channels=(upscale_factor*2)**2, kernel_size=1, stride=1, padding=0)
        #Other needed declarations
        self.relu = nn.LeakyReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor*2)
        self._initialize_weights()

    def forward(self, x, i1, i2):
        # First layer
        i2 = self.relu(self.convInt2_1(i2))
        i1 = self.relu(self.convInt1_1(i1))
        x = self.relu(self.convLow1(x))

        # Second layer
        i2 = self.relu(self.convInt2_2(i2))
        i1 = self.relu(self.convInt1_2(i1))
        x = self.relu(self.convLow2(x))

        # Third layer
        i2 = self.relu(self.convInt2_3(i2))
        i1 = self.relu(self.convInt1_3(i1))
        x = self.relu(self.convLow3(x))

        # Subpixel layer
        i2 = self.pixel_shuffle(i2)
        i1 = self.pixel_shuffle(i1)
        x = self.pixel_shuffle(x)

        return x

    def _initialize_weights(self):
        init.orthogonal_(self.convLow1.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convLow2.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convLow3.weight, init.calculate_gain('leaky_relu'))

        init.orthogonal_(self.convInt1_1.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt1_2.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt1_3.weight, init.calculate_gain('leaky_relu'))

        init.orthogonal_(self.convInt2_1.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt2_2.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.convInt2_3.weight, init.calculate_gain('leaky_relu'))