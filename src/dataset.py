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


def input_transform(size, crop_size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ])


def target_transform(size, crop_size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ])


def get_training_set(input_size, input_crop_size, output_size,
                     output_crop_size):
    hf_set = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train')
    train_set = Div2kDataset(hf_set, input_transform(input_size,
                                                     input_crop_size),
                             target_transform(output_size, output_crop_size))
    return train_set


def get_validation_set(input_size, input_crop_size, output_size,
                       output_crop_size):
    hf_set = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation')
    val_set = Div2kDataset(hf_set, input_transform(input_size, input_crop_size),
                           target_transform(output_size, output_crop_size))
    return val_set
