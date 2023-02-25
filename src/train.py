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

import torch
import torch.nn.functional as F

from tqdm import tqdm

import copy
import time
import shutil
import os


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
          num_epochs):
    history = {'loss': [], 'psnr': []}
    start = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        loss, psnr = training_step(model, device, train_loader, criterion,
                                   optimizer, scheduler)
        history['loss'].append(loss)
        history['psnr'].append(psnr)

        print(f'loss: {loss:.4f} psnr: {psnr}')
        print()
        # save checkpoint

    end = time.time() - start
    print(f'Finished training: {end // 60:.0f}m {end % 60:.0f}')

    return history, model


def train_and_evaluate(model, device, train_loader, val_loader, criterion,
                       optimizer, scheduler, num_epochs):
    history = {'loss': [], 'psnr': [], 'val_loss': [], 'val_psnr': []}
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_psnr = 0.0

    for epoch in range(num_epochs):
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
        # save checkpoint

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_model_wts = copy.deepcopy(model.state_dict())
            # save best model

    end = time.time() - start
    print(f'Finished training: {end // 60:.0f}m {end % 60:.0f}')
    print(f'Best PSNR: {best_psnr:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return history, model


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    p = os.path.join(save_dir, filename)
    torch.save(state, p)
    if is_best:
        shutil.copyfile(p, os.path.join(save_dir, 'best_model.pth'))
