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
"""Lightning DataModule for Venti3D MotionDB dataset."""
from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import Venti3DDataset
from transforms import TargetBuilder


class Venti3DDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Venti3D MotionDB dataset.

    This datamodule loads pre-converted QCNet pickle files. Data must be
    converted first using venti3d_converter.py.

    Args:
        root: Root directory containing train/val/test subdirectories with processed/ folders.
        train_batch_size: Batch size for training.
        val_batch_size: Batch size for validation.
        test_batch_size: Batch size for testing.
        shuffle: Whether to shuffle training data.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for data loading.
        persistent_workers: Whether to keep workers alive between epochs.
        dim: Dimensionality (2 or 3). Default 2.
        num_historical_steps: Number of historical time steps. Default 50.
        num_future_steps: Number of future time steps. Default 60.
        train_transform: Transform for training data.
        val_transform: Transform for validation data.
        test_transform: Transform for test data.
    """

    def __init__(
        self,
        root: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        dim: int = 2,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super(Venti3DDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # Dataset parameters
        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

        # Transforms - default to TargetBuilder if not provided
        self.train_transform = train_transform if train_transform is not None else TargetBuilder(num_historical_steps, num_future_steps)
        self.val_transform = val_transform if val_transform is not None else TargetBuilder(num_historical_steps, num_future_steps)
        self.test_transform = test_transform

    def _create_dataset(self, split: str, transform: Optional[Callable]) -> Venti3DDataset:
        """Create a dataset for the given split."""
        return Venti3DDataset(
            root=self.root,
            split=split,
            transform=transform,
            dim=self.dim,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
        )

    def prepare_data(self) -> None:
        # Verify data exists by creating a temporary dataset
        self._create_dataset('train', None)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = self._create_dataset('train', self.train_transform)
        self.val_dataset = self._create_dataset('val', self.val_transform)
        self.test_dataset = self._create_dataset('test', self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
