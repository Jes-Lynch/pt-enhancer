from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import glob
import os
from os.path import basename


import numpy as np
from os import listdir, makedirs
from os.path import join, exists
from utils import save_image, save_output

# SR settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_dir', type=str, required=True, help='input image to use')
parser.add_argument('--ref_dir', type=str, required=True, help='high resolution version of input images')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_dir', type=str, help='directory where output images are stored')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

sr_set = opt.input_dir

images = glob.glob(sr_set+"/*.png")

model_dir = join("checkpoints", opt.model)

for image in images:
    filename = os.fsdecode(image)
    print(opt)
    with open(image, 'rb') as file:
        img = Image.open(filename).convert('YCbCr')
    y, cb, cr = img.split()

    model = torch.load(model_dir)
    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    if opt.cuda:
        model = model.cuda()
        input = input.cuda()

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    path = opt.ref_dir
    new_filename = basename(filename)
    print(filename)
    low_img = Image.open(filename)
    hr_img = Image.open(path + new_filename)

    if not exists(opt.output_dir):
        makedirs(opt.output_dir)

    out_img = np.asarray(out_img)
    low_img = np.asarray(low_img)
    high_img = np.asarray(hr_img)

    save_image(image=out_img, path=os.path.join(opt.output_dir, '%s' % basename(filename)))
    save_output(lr_img=low_img, prediction=out_img, hr_img=high_img, path=os.path.join(opt.output_dir, '%s' % basename(filename)))
