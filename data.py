from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        Resize((int(1280/upscale_factor), int(1280/upscale_factor)), interpolation=3),
        CenterCrop(crop_size/upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size, upscale_factor):
    return Compose([
        Resize((1280, 1280), interpolation=3),
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    root_dir = join("dataset", "kidney/images")
    highres_dir = join(root_dir, "highres")
    lowres_dir = join(root_dir, "lowres")
    crop_size = calculate_valid_crop_size(1024, upscale_factor)

    return DatasetFromFolder(lowres_dir, highres_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size, upscale_factor))


def get_test_set(upscale_factor):
    root_dir = join("dataset", "kidney/images")
    highres_dir = join(root_dir, "highres")
    lowres_dir = join(root_dir, "lowres")
    crop_size = calculate_valid_crop_size(1024, upscale_factor)

    return DatasetFromFolder(lowres_dir, highres_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size, upscale_factor))


def get_sr_set():
    root_dir = join("dataset", "kidney/images")
    sr_dir = join(root_dir, "lowres")
    return sr_dir
