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
"""Dataset class for Venti3D MotionDB data converted to QCNet format."""
import os
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData

from utils import load_dataset_config


class Venti3DDataset(Dataset):
    """Dataset class for Venti3D MotionDB data (pre-converted to QCNet format).

    This dataset loads pre-processed pickle files that have already been converted
    from MotionDB format to QCNet format using venti3d_converter.py.

    Args:
        root: The root folder of the dataset.
        split: Dataset split: "train" | "val" | "test".
        processed_dir: Optional directory containing processed .pkl files.
            By default, looks for files in root/split/processed/.
        transform: Optional transform function applied to each sample.
        dim: Dimensionality (2 or 3). Default 2.
        num_historical_steps: Number of historical time steps. Default 50.
        num_future_steps: Number of future time steps. Default 60.
    """

    def __init__(self,
                 root: str,
                 split: str,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 2,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60) -> None:
        root = os.path.expanduser(os.path.normpath(root))
        if not os.path.isdir(root):
            os.makedirs(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split

        if processed_dir is None:
            processed_dir = os.path.join(root, split, 'processed')
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))

        self._processed_dir = processed_dir
        if os.path.isdir(self._processed_dir):
            self._processed_file_names = sorted([
                name for name in os.listdir(self._processed_dir)
                if os.path.isfile(os.path.join(self._processed_dir, name))
                and name.endswith(('pkl', 'pickle'))
            ])
        else:
            self._processed_file_names = []

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps

        # Load type definitions from config
        config = load_dataset_config('venti3d')
        self._agent_types = config['agent_types']
        self._polygon_types = config['polygon_types']

        self._agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        self._polygon_is_intersections = [True, False, None]
        self._polygon_is_bidirectionals = [True, False]
        self._point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                             'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                             'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                             'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

        super(Venti3DDataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.split, 'raw')

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def download(self) -> None:
        pass  # Data should already be converted

    def process(self) -> None:
        pass  # Data should already be converted

    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int) -> HeteroData:
        with open(self.processed_paths[idx], 'rb') as handle:
            return HeteroData(pickle.load(handle))

    def _download(self) -> None:
        # No downloading needed for pre-converted data
        pass

    def _process(self) -> None:
        # No processing needed for pre-converted data
        if not self._processed_file_names:
            print(f"Warning: No processed files found in {self._processed_dir}")
            print("Please run venti3d_converter.py to convert MotionDB data first.")
