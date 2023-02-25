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

import torchvision
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, title=None):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')


def show_batch(data_loader):
    # Get a random batch
    lr_images, hr_images = next(iter(data_loader))

    imshow(torchvision.utils.make_grid(lr_images))
    plt.title('Input LR')
    plt.show()

    imshow(torchvision.utils.make_grid(hr_images))
    plt.title('Target HR')
    plt.show()


def show_images(data_loader, num_images=2):
    images_so_far = 0
    _ = plt.figure(figsize=(8, 8))
    idx = 1

    print('Left: Input LR    Right: Target HR')

    for lr_images, hr_images in data_loader:
        for j in range(lr_images.size()[0]):
            images_so_far += 1

            plt.subplot(num_images, 2, idx)
            imshow(lr_images[j])

            plt.subplot(num_images, 2, idx + 1)
            imshow(hr_images[j])

            idx += 2

            if images_so_far == num_images:
                plt.show()
                return
