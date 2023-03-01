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

from datasets import load_dataset
from torchvision import transforms
from PIL import Image


def build_div2k_patch_dataset(output_dir, size, patch_size, scale):
    data_dir = os.path.join(output_dir, 'div2kpatch')
    train_dir = os.path.join(data_dir, 'train')
    train_lr = os.path.join(train_dir, 'lr')
    train_hr = os.path.join(train_dir, 'hr')

    val_dir = os.path.join(data_dir, 'val')
    val_lr = os.path.join(val_dir, 'lr')
    val_hr = os.path.join(val_dir, 'hr')

    if not os.path.exists(train_lr):
        os.makedirs(train_lr)
    if not os.path.exists(train_hr):
        os.makedirs(train_hr)
    if not os.path.exists(val_lr):
        os.makedirs(val_lr)
    if not os.path.exists(val_hr):
        os.makedirs(val_hr)

    # Training set
    hf_trainset = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train')

    idx = 0
    for img_pair in hf_trainset:
        img_lr = Image.open(img_pair['lr'])
        img_hr = Image.open(img_pair['hr'])

        img_lr = transforms.Resize(size)(img_lr)
        img_hr = transforms.Resize(size)(img_hr)

        for i in range(0, img_lr.height, patch_size):
            for j in range(0, img_lr.width, patch_size):
                # Extract the patch
                patch_hr = img_hr.crop((j, i, j + patch_size, i + patch_size))
                patch_lr = img_lr.crop((j, i, j + patch_size, i + patch_size))

                patch_lr = transforms.Resize(patch_size // scale)(patch_lr)

                # Save the patch
                patch_lr.save(os.path.join(train_lr, f'patch_{idx}.jpg'))
                patch_hr.save(os.path.join(train_hr, f'patch_{idx}.jpg'))

                idx += 1

    # Validation set
    hf_valset = load_dataset('eugenesiow/Div2k',
                             'bicubic_x4',
                             split='validation')

    idx = 0
    for img_pair in hf_valset:
        img_lr = Image.open(img_pair['lr'])
        img_hr = Image.open(img_pair['hr'])

        img_lr = transforms.Resize(size)(img_lr)
        img_hr = transforms.Resize(size)(img_hr)

        for i in range(0, img_lr.height, patch_size):
            for j in range(0, img_lr.width, patch_size):
                # Extract the patch
                patch_hr = img_hr.crop((j, i, j + patch_size, i + patch_size))
                patch_lr = img_lr.crop((j, i, j + patch_size, i + patch_size))

                patch_lr = transforms.Resize(patch_size // scale)(patch_lr)

                # Save the patch
                patch_lr.save(os.path.join(val_lr, f'patch_{idx}.jpg'))
                patch_hr.save(os.path.join(val_hr, f'patch_{idx}.jpg'))

                idx += 1


def main():
    build_div2k_patch_dataset('./data', 1024, 256, 4)


if __name__ == '__main__':
    main()
