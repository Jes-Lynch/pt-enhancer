from __future__ import print_function
import argparse
from data import  get_test_set
import numpy as np
import os
from os import makedirs
from os.path import join, exists
from PIL import Image
import torch
import torchvision.utils as tv
from torch.utils.data import DataLoader
from utils import save_output


# SR settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, required=True, help="super resolution upscale factor")
parser.add_argument('--full_size', type=int, default=1024, required=True, help="Size of target image")
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_dir', type=str, help='directory where output images are stored')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()


test_set = get_test_set(opt.upscale_factor, opt.full_size)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=True)
device = torch.device("cuda" if opt.cuda else "cpu")
model_dir = join("singleinput_checkpoints/", opt.model)
model = torch.load(model_dir)
model.to(device)


def main():
    counter = 0
    print("Saving Images")
    if not exists(opt.output_dir):
        makedirs(opt.output_dir)
    with torch.no_grad():
        for batch in testing_data_loader:
            inimg, int1, int2, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            _,_,prediction = model(inimg, int1, int2)
            lowres_fname=(test_set.lowres_filenames[counter])
            fname=lowres_fname[27:40]
            in_filename = opt.output_dir + str(fname)
            out_filename = opt.output_dir + 'out' + str(fname)
            tg_filename = opt.output_dir + 'tg' + str(fname)
            result_filename = 'Result_' + str(fname)
            tv.save_image(inimg, in_filename)
            tv.save_image(prediction, out_filename)
            tv.save_image(target, tg_filename)
            low_img = Image.open(in_filename).convert('L')
            out_img = Image.open(out_filename).convert('L')
            hr_img = Image.open(tg_filename).convert('L')
            out_img = np.asarray(out_img)
            low_img = np.asarray(low_img)
            high_img = np.asarray(hr_img)
            save_output(lr_img=low_img, prediction=out_img, hr_img=high_img,
                        path=os.path.join(opt.output_dir, '%s' % result_filename))
            os.remove(in_filename)
            os.remove(out_filename)
            os.remove(tg_filename)
            counter += 1


if __name__ == '__main__':
    main()