from dataset import DatasetFromFolder
from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


#For all of these transformations, we are currently assuming an upscale factor of 8 (256 -> 2048)
#Input final size: 128
def input_transform(crop_size, full_size, upscale_factor):
    return Compose([
        Resize((int(full_size/upscale_factor), int(full_size/upscale_factor)), interpolation=3),
        CenterCrop(int(crop_size/upscale_factor)),
        ToTensor(),
    ])


#Target final size: 1024
def target_transform(crop_size, full_size):
    return Compose([
        Resize((full_size, full_size), interpolation=3),
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor, full_size):
    root_dir = join("dataset", "kidney/train")
    highres_dir = join(root_dir, "highres")
    int2_dir = join(root_dir, "int2")
    int1_dir = join(root_dir, "int1")
    lowres_dir = join(root_dir, "lowres")
    crop_size = calculate_valid_crop_size(full_size, upscale_factor)

    return DatasetFromFolder(lowres_dir, int1_dir, int2_dir, highres_dir,
                            input_transform=input_transform(crop_size, full_size, upscale_factor),
                            target_transform=target_transform(crop_size, full_size))


def get_test_set(upscale_factor, full_size):
    root_dir = join("dataset", "kidney/test")
    highres_dir = join(root_dir, "highres")
    int2_dir = join(root_dir, "int2")
    int1_dir = join(root_dir, "int1")
    lowres_dir = join(root_dir, "lowres")
    crop_size = calculate_valid_crop_size(full_size, upscale_factor)

    return DatasetFromFolder(lowres_dir, int1_dir, int2_dir, highres_dir,
                            input_transform=input_transform(crop_size, full_size, upscale_factor),
                            target_transform=target_transform(crop_size, full_size))