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

from dataset import get_training_set, get_validation_set
from train import train_and_evaluate
from models import SRCNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import random


def main():
    seed = 42
    batch_size = 16
    num_workers = 0
    num_epochs = 2

    input_size = 256
    crop_size = 224

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_set = get_training_set(input_size, crop_size)

    val_set = get_validation_set(input_size, crop_size)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(device)

    # Instantiate a neural network model
    net = SRCNN()

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        net = nn.DataParallel(net)

    net = net.to(device)

    # Define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train and evaluate the model
    history, net = train_and_evaluate(net, device, train_loader, val_loader,
                                      criterion, optimizer, scheduler,
                                      num_epochs)


if __name__ == '__main__':
    main()
