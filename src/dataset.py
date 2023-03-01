# Copyright 2023 The Super Scaler Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


class Div2kDataset(Dataset):

    def __init__(self, hf_dataset, transform_input, transform_target):
        self.hf_dataset = hf_dataset
        self.input_transform = transform_input
        self.target_transform = transform_target

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        try:
            img_lr = Image.open(sample['lr'])
            img_lr = self.input_transform(img_lr)

            img_hr = Image.open(sample['hr'])
            img_hr = self.target_transform(img_hr)

            return img_lr, img_hr
        except Exception as exc:
            print(exc)
            return None


def is_image_file(filename):
    supported_file_types = ['.jpg', '.jpeg', '.png', '.gif']
    return any(filename.endswith(ext) for ext in supported_file_types)


class Div2kFromFolder(Dataset):

    def __init__(self, input_dir, target_dir, transform_input,
                 transform_target):
        super().__init__()

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_filenames = [
            x for x in os.listdir(input_dir) if is_image_file(x)
        ]

        self.input_transform = transform_input
        self.target_transform = transform_target

        self.num_samples = len(self.input_filenames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            filename = self.input_filenames[idx]

            img_lr = Image.open(os.path.join(self.input_dir, filename))
            img_lr = self.input_transform(img_lr)

            img_hr = Image.open(os.path.join(self.target_dir, filename))
            img_hr = self.target_transform(img_hr)

            return img_lr, img_hr
        except Exception as exc:
            print(exc)
            return None


def input_transform(size, crop_size, scale, blur):
    if blur:
        return transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.Resize(
                crop_size // scale,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.GaussianBlur(kernel_size=5, sigma=1.0),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size // scale,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def target_transform(size, crop_size):
    return transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def bicubic_transform(img, size):
    return transforms.Resize(
        size, interpolation=transforms.InterpolationMode.BICUBIC)(img)


def get_training_set(dataset, input_size, output_size, crop_size, scale, blur):
    hf_set = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train')
    if dataset == 'div2k':
        return Div2kDataset(hf_set,
                            input_transform(input_size, crop_size, scale, blur),
                            target_transform(output_size, crop_size))

    # TODO: FixMe
    #       Remove hard coded path
    inp_dir = './data/div2kpatch/train/lr'
    tar_dir = './data/div2kpatch/train/hr'

    if dataset == 'div2kpatch':
        return Div2kFromFolder(
            inp_dir, tar_dir, input_transform(input_size, crop_size, scale,
                                              blur),
            target_transform(output_size, crop_size))


def get_validation_set(dataset, input_size, output_size, crop_size, scale,
                       blur):
    hf_set = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation')
    if dataset == 'div2k':
        return Div2kDataset(hf_set,
                            input_transform(input_size, crop_size, scale, blur),
                            target_transform(output_size, crop_size))

    # TODO: FixMe
    #       Remove hard coded path
    inp_dir = './data/div2kpatch/val/lr'
    tar_dir = './data/div2kpatch/val/hr'

    if dataset == 'div2kpatch':
        return Div2kFromFolder(
            inp_dir, tar_dir, input_transform(input_size, crop_size, scale,
                                              blur),
            target_transform(output_size, crop_size))
