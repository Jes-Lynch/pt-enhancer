import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('L')
    #y, _, _ = img.split()
    y = img
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, lowres_dir, int1_dir, int2_dir, highres_dir, input_transform=None, int1_transform=None, int2_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.highres_filenames = [join(highres_dir, x) for x in listdir(highres_dir) if is_image_file(x)]
        self.int2_filenames = [join(int2_dir, x) for x in listdir(int2_dir) if is_image_file(x)]
        self.int1_filenames = [join(int1_dir, x) for x in listdir(int1_dir) if is_image_file(x)]
        self.lowres_filenames = [join(lowres_dir, x) for x in listdir(lowres_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.int1_transform = int1_transform
        self.int2_transform = int2_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.lowres_filenames[index])
        int1 = load_img(self.int1_filenames[index])
        int2 = load_img(self.int2_filenames[index])
        target = load_img(self.highres_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.int1_transform:
            int1 = self.int1_transform(int1)
        if self.int2_transform:
            int2 = self.int2_transform(int2)
        if self.target_transform:
            target = self.target_transform(target)

        return input, int1, int2, target

    def __len__(self):
        return len(self.highres_filenames)
