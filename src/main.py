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
from models import SRCNN, ESPCN
from train import train

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset',
#                     default='div2k',
#                     help='Dataset to use, defined in dataset.py')
# parser.add_argument('--data_root',
#                     default='./data',
#                     help='Directory to download dataset')
parser.add_argument('--config', help='JSON file containing training arguments')
parser.add_argument('--out_dir',
                    default='.',
                    help='Directory to output images and model checkpoints')
parser.add_argument('--checkpoint_freq',
                    default=1,
                    type=int,
                    help='Frequency to save checkpoints and generate results')
parser.add_argument('--seed', default=123, type=int, help='Random seed to use')
parser.add_argument('--resume',
                    default=None,
                    help='Optional, path to checkpoint')


def main():
    # Load the parameters from json file
    args = parser.parse_args()
    assert os.path.isfile(args.config), f'{args.config} file not found.'
    params = utils.Params(args.config)
    assert args.checkpoint_freq > 0, 'checkpoint frequency must be > 0'

    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use GPU or MPS if available
    params.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        params.device = torch.device('mps')
    print(params.device)

    train_set = get_training_set(params.dataset, params.input_size,
                                 params.output_size, params.crop_size,
                                 params.upscale_factor)
    train_loader = DataLoader(train_set,
                              batch_size=params.batch_size,
                              shuffle=True,
                              num_workers=params.num_workers,
                              pin_memory=True)

    val_set = get_validation_set(params.dataset, params.input_size,
                                 params.output_size, params.crop_size,
                                 params.upscale_factor)
    val_loader = DataLoader(val_set,
                            batch_size=params.valbatch_size,
                            shuffle=True,
                            num_workers=params.num_workers,
                            pin_memory=True)

    # Instantiate a neural network model
    net = None
    if params.model_name == 'srcnn':
        net = SRCNN()
    elif params.model_name == 'espcn':
        net = ESPCN(params.upscale_factor)
    else:
        raise f'{params.model_name} model not implemented.'

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        net = nn.DataParallel(net)

    net = net.to(params.device)

    # Define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # Decay LR by a factor of gamma every milestone
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=params.milestones,
                                               gamma=params.gamma)

    # Train and evaluate the model
    history, net = train(net,
                         params.device,
                         train_loader,
                         criterion,
                         optimizer,
                         scheduler,
                         params.num_epochs,
                         val_loader=val_loader,
                         verbose=1,
                         output_dir=args.out_dir,
                         checkpoint_path=args.resume,
                         checkpoint_freq=args.checkpoint_freq)

    # Save history
    utils.save_history(history, args.out_dir)


if __name__ == '__main__':
    main()
