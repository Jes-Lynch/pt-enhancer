from __future__ import print_function
import argparse
from data import get_training_set, get_test_set
from math import log10
from model import RNet
from model import Interpolate
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tv
from torch.utils.data import DataLoader


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, required=True, help="Super resolution upscale factor")
parser.add_argument('--full_size', type=int, default=1024, required=True, help="Size of target image")
parser.add_argument('--batchSize', type=int, default=10, help='Training batch size')
parser.add_argument('--testBatchSize', type=int, default=100, help='Testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='Use cuda?')
parser.add_argument('--train', action='store_true', help='Run training')
parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
device = torch.device("cuda" if opt.cuda else "cpu")
full_size = opt.full_size

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor, opt.full_size)
test_set = get_test_set(opt.upscale_factor, opt.full_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

print('===> Building model')
model = RNet(upscale_factor=opt.upscale_factor, full_size=opt.full_size)
model.to(device)
criterion = nn.MSELoss()
#Three optimizers, one for each output
optimizerLow = optim.Adam(model.parameters(), lr=opt.lr)
optimizerInt1 = optim.Adam(model.parameters(), lr=opt.lr)
optimizerInt2 = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    low_loss = 0
    int1_loss = 0
    int2_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        inimg, int1, int2, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        epochloss = 0

        #Run through the model, optimizes for each output, from int2 to int1 and finally the lowest resolution input
        optimizerLow.zero_grad()
        optimizerInt1.zero_grad()
        optimizerInt2.zero_grad()
        int2Result, int1Result, lowResult = model(inimg, int1, int2, target)
        loss = criterion(int2Result, target)
        int2_loss += loss.item()
        epochloss += loss.item()
        loss.backward(retain_graph=True)
        optimizerInt2.step()
        loss = criterion(int1Result, target)
        int1_loss += loss.item()
        epochloss += loss.item()
        loss.backward(retain_graph=True)
        optimizerInt1.step()
        loss = criterion(lowResult, target)
        low_loss += loss.item()
        epochloss += loss.item()
        loss.backward()
        optimizerLow.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), epochloss/3))

    print("===> Epoch {} Complete: Avg. Low Loss: {:.4f}".format(epoch, low_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Int1 Loss: {:.4f}".format(epoch, int1_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Int2 Loss: {:.4f}".format(epoch, int2_loss / len(training_data_loader)))


def test(epoch):
    avg_psnr = 0
    model = torch.load("rrcnn_checkpoints_kidney/model_epoch_{}.pth".format(epoch))
    model.to(device)
    int2Pred = []
    int1Pred = []
    lowPred = []
    inputs = []
    targets = []
    resize_up = Interpolate(size=(full_size, full_size), mode='bilinear')
    with torch.no_grad():
        counter = 1
        for batch in testing_data_loader:
            counter +=1
            inimg, int1, int2, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

            int2Up = resize_up(int2)
            int2Result, int1Result, lowResult = model(inimg, int1, int2, int2Up)
            int2Pred.append(int2Result)
            int1Pred.append(int1Result)
            lowPred.append(lowResult)
            inputs.append(inimg)
            targets.append(target)
            mse = criterion(int2Result, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return int2Pred, int1Pred, lowPred, inputs, targets


def checkpoint(epoch):
    if not os.path.exists('rrcnn_checkpoints_kidney'):
        os.makedirs('rrcnn_checkpoints_kidney')
    model_out_path = "rrcnn_checkpoints_kidney/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main():
    if opt.train:
        for epoch in range(1, opt.nEpochs + 1):
            train(epoch)
            checkpoint(epoch)
    else:
        epoch = opt.nEpochs
        int2Pred, int1Pred, lowPred, inputs, targets = test(epoch)
        x=(len(testing_data_loader.dataset))
        # Print the target image, the reconstructions, and the original input
        if not os.path.exists('dataset/kidney/rrcnn_low_{}/'.format(epoch)):
            os.makedirs('dataset/kidney/rrcnn_low_{}/'.format(epoch))
        if not os.path.exists('dataset/kidney/rrcnn_int1_{}/'.format(epoch)):
            os.makedirs('dataset/kidney/rrcnn_int1_{}/'.format(epoch))
        if not os.path.exists('dataset/kidney/rrcnn_int2_{}/'.format(epoch)):
            os.makedirs('dataset/kidney/rrcnn_int2_{}/'.format(epoch))
        if not os.path.exists('dataset/kidney/rrcnn_input_{}/'.format(epoch)):
            os.makedirs('dataset/kidney/rrcnn_input_{}/'.format(epoch))
        if not os.path.exists('dataset/kidney/rrcnn_target_{}/'.format(epoch)):
            os.makedirs('dataset/kidney/rrcnn_target_{}/'.format(epoch))
        for i in range(x):
            lowres_fname = (test_set.lowres_filenames[i])
            fname = lowres_fname[27:41]
            filename = 'dataset/kidney/rrcnn_low_{}/'.format(epoch) + str(fname)
            i1filename = 'dataset/kidney/rrcnn_int1_{}/'.format(epoch) + str(fname)
            i2filename = 'dataset/kidney/rrcnn_int2_{}/'.format(epoch) + str(fname)
            in_filename = 'dataset/kidney/rrcnn_input_{}/'.format(epoch) + str(fname)
            tg_filename = 'dataset/kidney/rrcnn_target_{}/'.format(epoch) + str(fname)
            print(filename)
            tv.save_image(inputs[i], in_filename)
            tv.save_image(targets[i], tg_filename)
            tv.save_image(lowPred[i], filename)
            tv.save_image(int1Pred[i], i1filename)
            tv.save_image(int2Pred[i], i2filename)
            low_img = Image.open(in_filename).convert('L')
            low_img = low_img.resize((opt.full_size, opt.full_size), Image.BICUBIC)
            low_img.save(in_filename)


if __name__ == '__main__':
    main()
