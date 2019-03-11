from __future__ import print_function
import argparse
from data import get_training_set, get_test_set, input_transform
from math import log10
from model import Net
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tv
from torch.utils.data import DataLoader


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, required=True, help="super resolution upscale factor")
parser.add_argument('--full_size', type=int, default=1024, required=True, help="Size of target image")
parser.add_argument('--batchSize', type=int, default=10, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--train', action='store_true', help='run training')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor, opt.full_size)
test_set = get_test_set(opt.upscale_factor, opt.full_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor)
model.to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))



def test(epoch):
    avg_psnr = 0
    model = torch.load("siamese_checkpoints/model_epoch_{}.pth".format(epoch))
    model.to(device)
    lowPred = []
    inputs = []
    targets = []
    with torch.no_grad():
        counter = 1
        for batch in testing_data_loader:
            counter += 1
            inimg, int1, int2, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(
                device)

            lowResult = model(inimg, int1, int2)
            lowPred.append(lowResult)
            inputs.append(inimg)
            targets.append(target)
            mse = criterion(lowResult, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return lowPred, inputs, targets


def checkpoint(epoch):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    model_out_path = "checkpoints/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main():
    if opt.train:
        for epoch in range(1, opt.nEpochs + 1):
            train(epoch)
            checkpoint(epoch)
    else:
        epoch = opt.nEpochs
        lowPred, inputs, targets = test(epoch)
        x = (len(testing_data_loader.dataset))
        if not os.path.exists('dataset/kidney/test/lowprediction{}/'.format(epoch)):
            os.makedirs('dataset/kidney/test/lowprediction{}/'.format(epoch))
        if not os.path.exists('dataset/kidney/test/input{}/'.format(epoch)):
            os.makedirs('dataset/kidney/test/input{}/'.format(epoch))
        if not os.path.exists('dataset/kidney/test/target{}/'.format(epoch)):
            os.makedirs('dataset/kidney/test/target{}/'.format(epoch))
        for i in range(x):
            lowres_fname = (test_set.lowres_filenames[i])
            fname = lowres_fname[27:39]
            filename = 'dataset/kidney/test/lowprediction{}/'.format(epoch) + str(fname)
            in_filename = 'dataset/kidney/test/input{}/'.format(epoch) + str(fname)
            tg_filename = 'dataset/kidney/test/target{}/'.format(epoch) + str(fname)
            print(filename)
            tv.save_image(inputs[i], in_filename)
            tv.save_image(targets[i], tg_filename)
            tv.save_image(lowPred[i], filename)
            low_img = Image.open(in_filename).convert('L')
            low_img = low_img.resize((opt.full_size, opt.full_size), Image.BICUBIC)
            low_img.save(in_filename)


if __name__ == '__main__':
    main()
