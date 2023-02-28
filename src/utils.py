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

import json
import os
import shutil

import torch


class Params():

    def __init__(self, json_path):
        with open(json_path, encoding='utf-8') as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path, encoding='utf-8') as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def save_checkpoint(state,
                    output_dir,
                    checkpoint='last.pth.tar',
                    is_best=False):
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    filepath = os.path.join(checkpoint_dir, checkpoint)
    if not os.path.exists(checkpoint_dir):
        print(f'Making directory {checkpoint_dir}')
        os.mkdir(checkpoint_dir)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best.pth.tar'))


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise f'File doesn\'t exist {checkpoint_path}'
    return torch.load(checkpoint_path)


def get_results_dir(out_dir):
    results_dir = os.path.join(out_dir, 'results')
    if not os.path.exists(results_dir):
        print(f'Making directory {results_dir}')
        os.mkdir(results_dir)

    return results_dir
