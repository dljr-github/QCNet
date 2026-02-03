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
"""Configuration loader for dataset-specific type mappings."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def _get_configs_dir() -> Path:
    """Get the configs directory path."""
    # Look for configs directory relative to this file's location
    utils_dir = Path(__file__).parent
    project_root = utils_dir.parent
    configs_dir = project_root / 'configs'

    if not configs_dir.exists():
        raise FileNotFoundError(
            f"Configs directory not found at {configs_dir}. "
            "Please ensure the 'configs' directory exists in the project root."
        )

    return configs_dir


@lru_cache(maxsize=8)
def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Load dataset configuration from YAML file.

    Args:
        dataset_name: Name of the dataset (e.g., 'venti3d', 'argoverse_v2').

    Returns:
        Dictionary containing dataset configuration with keys:
        - agent_types: List of agent type names
        - polygon_types: List of polygon type names
        - agent_type_mapping: Dict mapping source type strings to indices (optional)
        - lane_type_mapping: Dict mapping lane type integers to polygon indices (optional)
        - crosswalk_type: Polygon type index for crosswalks (optional)
        - unknown_type: Polygon type index for unknown lanes (optional)

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required fields are missing.
    """
    configs_dir = _get_configs_dir()
    config_path = configs_dir / f'{dataset_name}.yaml'

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Please create configs/{dataset_name}.yaml with agent_types and polygon_types."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if 'agent_types' not in config:
        raise ValueError(f"Config {config_path} missing required field 'agent_types'")
    if 'polygon_types' not in config:
        raise ValueError(f"Config {config_path} missing required field 'polygon_types'")

    return config


def get_num_agent_types(dataset_name: str) -> int:
    """Get the number of agent types for a dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Number of agent types defined in the config.
    """
    config = load_dataset_config(dataset_name)
    return len(config['agent_types'])


def get_num_polygon_types(dataset_name: str) -> int:
    """Get the number of polygon types for a dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Number of polygon types defined in the config.
    """
    config = load_dataset_config(dataset_name)
    return len(config['polygon_types'])
