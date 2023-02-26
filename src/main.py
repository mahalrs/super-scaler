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

import argparse
import os

import utils
from dataset import get_training_set, get_validation_set
from models import SRCNN
from train import train_and_evaluate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset',
#                     default='div2k',
#                     help='Dataset to use, defined in dataset.py')
parser.add_argument('--exp_dir',
                    default='experiments/exp1_srcnn',
                    help='Directory containing params.json')
parser.add_argument('--seed', default=123, help='Random seed to use')
parser.add_argument('--resume',
                    default=None,
                    help='Optional, path to checkpoint')


def main():
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.exp_dir, 'params.json')
    assert os.path.isfile(
        json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use GPU or MPS if available
    params.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        params.device = torch.device('mps')

    train_set = get_training_set(params.input_size, params.crop_size)
    train_loader = DataLoader(train_set,
                              batch_size=params.batch_size,
                              shuffle=True,
                              num_workers=params.num_workers,
                              pin_memory=True)

    val_set = get_validation_set(params.input_size, params.crop_size)
    val_loader = DataLoader(val_set,
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=params.num_workers,
                            pin_memory=True)

    # Instantiate a neural network model
    net = SRCNN()

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        net = nn.DataParallel(net)

    net = net.to(params.device)

    # Define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=params.step_size,
                                          gamma=params.gamma)

    # Train and evaluate the model
    history, net = train_and_evaluate(net, params.device, train_loader,
                                      val_loader, criterion, optimizer,
                                      scheduler, params.num_epochs,
                                      args.exp_dir, args.resume)


if __name__ == '__main__':
    main()
