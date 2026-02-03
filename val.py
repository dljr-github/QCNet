# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset, Venti3DDataset
from predictors import QCNet
from transforms import TargetBuilder

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--dataset', type=str, default=None, choices=['argoverse_v2', 'venti3d'])
    args = parser.parse_args()

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)

    # Use command-line dataset if provided, otherwise use checkpoint's dataset
    dataset_name = args.dataset if args.dataset else model.dataset

    dataset_cls = {
        'argoverse_v2': ArgoverseV2Dataset,
        'venti3d': Venti3DDataset,
    }[dataset_name]

    if dataset_name == 'venti3d':
        val_dataset = dataset_cls(
            root=args.root,
            split=args.split,
            transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
            num_historical_steps=model.num_historical_steps,
            num_future_steps=model.num_future_steps,
        )
    else:
        val_dataset = dataset_cls(
            root=args.root,
            split=args.split,
            transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
        )

    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    trainer.validate(model, dataloader)
