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

import copy
import os
import time

import utils

import torch
import torch.nn.functional as F

from tqdm import tqdm


def calculate_psnr(batch_pred, batch_gt, max_val=1.0):
    mse = F.mse_loss(batch_pred, batch_gt, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.mean().item()


def training_step(model, device, data_loader, criterion, optimizer, scheduler):
    # switch to train mode
    model.train()

    running_loss = 0.0
    running_psnr = 0.0

    for inputs, targets in tqdm(data_loader, total=len(data_loader)):
        # move data to the same device as model
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # inputs, targets = inputs.to(device), targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # upscale the input images
        outputs = model(inputs)

        # compute the loss based on model output and target
        loss = criterion(outputs, targets)

        # backpropagate the loss
        loss.backward()

        # adjust parameters based on the calculated gradients
        optimizer.step()

        # statistics
        running_loss += loss.item()
        running_psnr += calculate_psnr(outputs, targets)

    scheduler.step()

    # Compute and return average loss and psnr
    avg_loss = running_loss / len(data_loader)
    avg_psnr = running_psnr / len(data_loader)

    return avg_loss, avg_psnr


def evaluate(model, device, data_loader, criterion):
    # switch to evaluate mode
    was_training = model.training
    model.eval()

    running_loss = 0.0
    running_psnr = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, total=len(data_loader)):
            # move data to the same device as model
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # upscale the input images
            outputs = model(inputs)

            # compute the loss based on model output and target
            loss = criterion(outputs, targets)

            # statistics
            running_loss += loss.item()
            running_psnr += calculate_psnr(outputs, targets)

        model.train(mode=was_training)

    # Compute and return average loss and psnr
    avg_loss = running_loss / len(data_loader)
    avg_psnr = running_psnr / len(data_loader)

    return avg_loss, avg_psnr


def train(model, device, train_loader, criterion, optimizer, scheduler,
          num_epochs, output_dir, checkpoint_path):
    history = {'loss': [], 'psnr': []}
    if checkpoint_path:
        history = utils.load_checkpoint(checkpoint_path)
        start_epoch = history['epoch'] + 1

        model.load_state_dict(history['model_state_dict'])
        optimizer.load_state_dict(history['optimizer_state_dict'])
        scheduler.load_state_dict(history['scheduler_state_dict'])

    start = time.time()
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        loss, psnr = training_step(model, device, train_loader, criterion,
                                   optimizer, scheduler)
        history['loss'].append(loss)
        history['psnr'].append(psnr)

        print(f'loss: {loss:.4f} psnr: {psnr}')
        print()

        # Save checkpoint
        history['model_state_dict'] = model.state_dict()
        history['optimizer_state_dict'] = optimizer.state_dict()
        history['scheduler_state_dict'] = scheduler.state_dict()
        history['epoch'] = epoch

        utils.save_checkpoint(history, output_dir)

    end = time.time() - start
    print(f'Finished training: {end // 60:.0f}m {end % 60:.0f}')

    return history, model


def train_and_evaluate(model, device, train_loader, val_loader, criterion,
                       optimizer, scheduler, num_epochs, output_dir,
                       checkpoint_path):
    history = {'loss': [], 'psnr': [], 'val_loss': [], 'val_psnr': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_psnr = 0.0
    start_epoch = 0

    if checkpoint_path:
        history = utils.load_checkpoint(checkpoint_path)
        best_model_wts = history['best_model_state_dict']
        best_psnr = history['best_psnr']
        start_epoch = history['epoch'] + 1

        model.load_state_dict(history['model_state_dict'])
        optimizer.load_state_dict(history['optimizer_state_dict'])
        scheduler.load_state_dict(history['scheduler_state_dict'])

    start = time.time()
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        loss, psnr = training_step(model, device, train_loader, criterion,
                                   optimizer, scheduler)
        val_loss, val_psnr = evaluate(model, device, val_loader, criterion)

        history['loss'].append(loss)
        history['psnr'].append(psnr)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)

        print(
            f'loss: {loss:.4f} psnr: {psnr} val_loss: {val_loss} val_psnr: {val_psnr}'
        )
        print()

        # Save checkpoint
        history['model_state_dict'] = model.state_dict()
        history['optimizer_state_dict'] = optimizer.state_dict()
        history['scheduler_state_dict'] = scheduler.state_dict()
        history['best_model_state_dict'] = best_model_wts
        history['best_psnr'] = best_psnr
        history['epoch'] = epoch

        utils.save_checkpoint(history, output_dir, is_best=val_psnr > best_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_model_wts = copy.deepcopy(model.state_dict())

    end = time.time() - start
    print(f'Finished training: {end // 60:.0f}m {end % 60:.0f}')
    print(f'Best PSNR: {best_psnr:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return history, model
