"""Generate train/val/test splits for Venti3D dataset with location + time aware grouping.

Creates a splits.json file that maps split names to scenario indices.
Scenarios from the same location (grid cell) AND same recording day are kept
together to prevent spatial data leakage. Scenarios from the same location but
different days can be in different splits (obstacles differ between days).

Features:
- Groups by port (PSA, BNS, KNS) from map_location field
- Clusters by location (grid-based clustering on ego position centroid)
- Groups by recording day (from start_timestamp_ns field)
- Stratified splits: each split contains each port
- Coverage guarantee (optional): ensures each spatial region has val/test representation

Usage:
    python -m datasets.venti3d_split --root /path/to/venti3d_data
    python -m datasets.venti3d_split --root /path/to/venti3d_data --grid_size 200
    python -m datasets.venti3d_split --root /path/to/venti3d_data --heatmap
    python -m datasets.venti3d_split --root /path/to/venti3d_data --coverage
"""
import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


@dataclass
class CoverageStats:
    """Statistics about spatial coverage guarantee."""
    total_spatial_regions: int
    regions_in_train: int
    regions_in_val: int
    regions_in_test: int
    regions_with_val_or_test: int
    regions_needing_reassignment: int
    reassigned_groups: int
    regions_without_train: int  # Regions that have no training data after reassignment


def load_scenario_location_info(
    root: Path,
    manifest: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Load location and time information from all scenarios.

    Args:
        root: Root directory containing scenarios/.
        manifest: List of manifest entries.

    Returns:
        List of dicts with keys: file, idx, port, x, y, date
    """
    location_info = []
    scenarios_dir = root / 'scenarios'

    for entry in manifest:
        file_name = entry['file']
        file_path = scenarios_dir / file_name

        with open(file_path, 'rb') as f:
            scenarios = pickle.load(f)

        for idx, scenario in enumerate(scenarios):
            # Extract port from map_location
            port = scenario.get('map_location', 'UNKNOWN')

            # Extract ego position centroid
            ego_positions = scenario.get('ego_positions', [])
            if len(ego_positions) > 0:
                ego_arr = np.array(ego_positions)
                centroid = ego_arr[:, :2].mean(axis=0)
                x, y = centroid[0], centroid[1]
            else:
                x, y = 0.0, 0.0

            # Extract recording date from timestamp
            timestamp_ns = scenario.get('start_timestamp_ns', 0)
            date = datetime.fromtimestamp(timestamp_ns / 1e9).strftime('%Y-%m-%d')

            location_info.append({
                'file': file_name,
                'idx': idx,
                'port': port,
                'x': x,
                'y': y,
                'date': date,
            })

    return location_info


def assign_grid_groups(
    location_info: List[Dict[str, Any]],
    grid_size: float,
) -> None:
    """Assign location-time group IDs based on grid cells and recording date.

    Modifies location_info in place, adding 'location_group' key.
    Same location on different days = different groups (obstacles differ).

    Args:
        location_info: List of location info dicts.
        grid_size: Size of grid cells in meters.
    """
    for item in location_info:
        grid_x = int(item['x'] // grid_size)
        grid_y = int(item['y'] // grid_size)
        # Include date in group key - same location on different days = different groups
        item['location_group'] = f"{item['port']}_{grid_x}_{grid_y}_{item['date']}"


def compute_spatial_regions(
    location_info: List[Dict[str, Any]],
    grid_size: float,
) -> Dict[str, List[str]]:
    """Compute spatial regions (ignoring date) and map to location groups.

    A spatial region is defined by port and grid cell, ignoring the date.
    Multiple location-time groups can belong to the same spatial region
    (same location, different days).

    Args:
        location_info: List of location info dicts with location_group assigned.
        grid_size: Size of grid cells in meters.

    Returns:
        Dictionary mapping spatial_region -> list of location_group IDs.
    """
    region_to_groups: Dict[str, set] = defaultdict(set)

    for item in location_info:
        grid_x = int(item['x'] // grid_size)
        grid_y = int(item['y'] // grid_size)
        # Spatial region ignores date
        spatial_region = f"{item['port']}_{grid_x}_{grid_y}"
        region_to_groups[spatial_region].add(item['location_group'])

    # Convert sets to sorted lists for determinism
    return {region: sorted(groups) for region, groups in region_to_groups.items()}


def assign_splits(
    location_info: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Assign splits to location groups using deterministic hash.

    Modifies location_info in place, adding 'split' key.

    Args:
        location_info: List of location info dicts with location_group.
        train_ratio: Ratio for training set.
        val_ratio: Ratio for validation set.
    """
    # Group items by location_group
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in location_info:
        groups[item['location_group']].append(item)

    # Assign each group to a split
    for group_id, items in groups.items():
        # Deterministic hash based on location group ID
        h = int(hashlib.md5(group_id.encode()).hexdigest(), 16) % 10000 / 10000

        if h < train_ratio:
            split = 'train'
        elif h < train_ratio + val_ratio:
            split = 'val'
        else:
            split = 'test'

        for item in items:
            item['split'] = split


def assign_splits_with_coverage(
    location_info: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    grid_size: float,
) -> CoverageStats:
    """Assign splits with spatial coverage guarantee.

    First performs hash-based assignment (same as assign_splits), then ensures
    each spatial region has at least one group in val or test splits.

    Args:
        location_info: List of location info dicts with location_group assigned.
        train_ratio: Ratio for training set.
        val_ratio: Ratio for validation set.
        grid_size: Size of grid cells in meters (for computing spatial regions).

    Returns:
        CoverageStats with statistics about the coverage guarantee.
    """
    # Step 1: Initial hash-based assignment (same as assign_splits)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in location_info:
        groups[item['location_group']].append(item)

    group_to_split: Dict[str, str] = {}
    for group_id in groups:
        h = int(hashlib.md5(group_id.encode()).hexdigest(), 16) % 10000 / 10000
        if h < train_ratio:
            group_to_split[group_id] = 'train'
        elif h < train_ratio + val_ratio:
            group_to_split[group_id] = 'val'
        else:
            group_to_split[group_id] = 'test'

    # Step 2: Compute spatial regions
    region_to_groups = compute_spatial_regions(location_info, grid_size)

    # Step 3: Check coverage and reassign if needed
    regions_needing_reassignment = 0
    reassigned_groups = 0

    for region, region_groups in sorted(region_to_groups.items()):
        # Check if any group in this region is in val or test
        has_val_or_test = any(
            group_to_split[g] in ('val', 'test') for g in region_groups
        )

        if not has_val_or_test:
            # All groups in this region are in train - reassign one
            regions_needing_reassignment += 1

            # Use deterministic hash to decide whether to assign to val or test
            region_hash = int(hashlib.md5(region.encode()).hexdigest(), 16) % 2
            target_split = 'val' if region_hash == 0 else 'test'

            # Reassign first group (list is sorted for determinism)
            group_to_reassign = region_groups[0]
            group_to_split[group_to_reassign] = target_split
            reassigned_groups += 1

    # Step 4: Apply splits to location_info
    for item in location_info:
        item['split'] = group_to_split[item['location_group']]

    # Step 5: Compute coverage statistics
    regions_in_train = 0
    regions_in_val = 0
    regions_in_test = 0
    regions_with_val_or_test = 0
    regions_without_train = 0

    for region, region_groups in region_to_groups.items():
        splits_in_region = set(group_to_split[g] for g in region_groups)
        if 'train' in splits_in_region:
            regions_in_train += 1
        else:
            regions_without_train += 1
        if 'val' in splits_in_region:
            regions_in_val += 1
        if 'test' in splits_in_region:
            regions_in_test += 1
        if 'val' in splits_in_region or 'test' in splits_in_region:
            regions_with_val_or_test += 1

    return CoverageStats(
        total_spatial_regions=len(region_to_groups),
        regions_in_train=regions_in_train,
        regions_in_val=regions_in_val,
        regions_in_test=regions_in_test,
        regions_with_val_or_test=regions_with_val_or_test,
        regions_needing_reassignment=regions_needing_reassignment,
        reassigned_groups=reassigned_groups,
        regions_without_train=regions_without_train,
    )


def build_splits_dict(
    location_info: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[int]]]:
    """Build splits dictionary from location info.

    Args:
        location_info: List of location info dicts with split assigned.

    Returns:
        Dictionary mapping split names to file-grouped indices.
    """
    splits: Dict[str, Dict[str, List[int]]] = {'train': {}, 'val': {}, 'test': {}}

    for item in location_info:
        split = item['split']
        file_name = item['file']
        idx = item['idx']

        if file_name not in splits[split]:
            splits[split][file_name] = []
        splits[split][file_name].append(idx)

    # Sort indices within each file for determinism
    for split_name in splits:
        for file_name in splits[split_name]:
            splits[split_name][file_name].sort()

    return splits


def render_heatmaps(
    location_info: List[Dict[str, Any]],
    grid_size: float,
    output_dir: Path,
) -> None:
    """Render heatmap of ego locations for each port.

    Args:
        location_info: List of location info dicts with split assigned.
        grid_size: Grid size used for clustering.
        output_dir: Directory to save heatmap images.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping heatmap generation")
        return

    # Get unique ports
    ports = sorted(set(item['port'] for item in location_info))

    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}

    for port in ports:
        port_items = [item for item in location_info if item['port'] == port]

        if not port_items:
            continue

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot points by split
        for split, color in colors.items():
            items = [i for i in port_items if i['split'] == split]
            if not items:
                continue
            xs = [i['x'] for i in items]
            ys = [i['y'] for i in items]
            ax.scatter(xs, ys, c=color, alpha=0.5, s=20, label=f'{split} ({len(items)})')

        # Calculate grid bounds
        all_x = [i['x'] for i in port_items]
        all_y = [i['y'] for i in port_items]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Add margin
        margin = grid_size
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin

        # Draw grid lines
        grid_x_start = int(x_min // grid_size) * grid_size
        grid_y_start = int(y_min // grid_size) * grid_size

        for gx in np.arange(grid_x_start, x_max + grid_size, grid_size):
            ax.axvline(x=gx, color='gray', alpha=0.3, linewidth=0.5)
        for gy in np.arange(grid_y_start, y_max + grid_size, grid_size):
            ax.axhline(y=gy, color='gray', alpha=0.3, linewidth=0.5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{port} - Ego Location Distribution\n(grid size: {grid_size}m)')
        ax.legend(loc='upper right')

        # Add statistics text
        stats_text = f'Total: {len(port_items)} scenarios'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        output_path = output_dir / f'heatmap_{port}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved heatmap: {output_path}")


def print_coverage_statistics(stats: CoverageStats, grid_size: float) -> None:
    """Print spatial coverage statistics.

    Args:
        stats: CoverageStats from assign_splits_with_coverage.
        grid_size: Grid size used for coverage regions.
    """
    total = stats.total_spatial_regions
    print(f"\nSpatial coverage statistics (grid={grid_size}m):")
    print(f"  Total spatial regions: {total}")
    print(f"  Regions with train data: {stats.regions_in_train} ({100*stats.regions_in_train/total:.1f}%)")
    print(f"  Regions with val data: {stats.regions_in_val} ({100*stats.regions_in_val/total:.1f}%)")
    print(f"  Regions with test data: {stats.regions_in_test} ({100*stats.regions_in_test/total:.1f}%)")
    print(f"  Regions with val OR test: {stats.regions_with_val_or_test} ({100*stats.regions_with_val_or_test/total:.1f}%)")

    if stats.regions_needing_reassignment > 0:
        print(f"\n  Coverage guarantee applied:")
        print(f"    Regions needing reassignment: {stats.regions_needing_reassignment}")
        print(f"    Groups reassigned from train: {stats.reassigned_groups}")

    if stats.regions_without_train > 0:
        print(f"\n  Warning: {stats.regions_without_train} regions have NO training data")
        print(f"    (single-day regions where the only group was reassigned to val/test)")


def generate_splits(
    root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    grid_size: float = 200.0,
    output: str = 'splits.json',
    heatmap: bool = False,
    require_coverage: bool = False,
    coverage_grid_size: Optional[float] = None,
) -> Dict[str, Dict[str, List[int]]]:
    """Generate train/val/test splits for Venti3D dataset.

    Uses location + time grouping to prevent spatial data leakage while
    allowing temporal diversity. Scenarios from the same location (grid cell)
    AND same recording day are assigned to the same split. Scenarios from
    the same location but different days can be in different splits
    (obstacles differ between days).

    When require_coverage is True, ensures each spatial region has at least
    one group in val or test splits for geographic coverage (disabled by default).

    Output format (human-readable, grouped by file):
        {
            "train": {"file1.pkl": [0, 1, 2], "file2.pkl": [0, 1]},
            "val": {"file3.pkl": [0, 1, 2, 3]},
            "test": {...}
        }

    Args:
        root: Root directory containing manifest.pkl and scenarios/.
        train_ratio: Ratio of scenarios for training (default 0.8).
        val_ratio: Ratio of scenarios for validation (default 0.1).
        grid_size: Size of grid cells for location clustering in meters (default 200.0).
        output: Output filename for splits JSON (default 'splits.json').
        heatmap: If True, generate heatmap visualizations.
        require_coverage: If True, ensure each spatial region has val/test coverage.
        coverage_grid_size: Grid size for coverage regions (default: same as grid_size).

    Returns:
        Dictionary mapping split names to file-grouped indices.
    """
    root = Path(root)

    # Load manifest
    manifest_path = root / 'manifest.pkl'
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.pkl not found at {manifest_path}")

    with open(manifest_path, 'rb') as f:
        manifest = pickle.load(f)

    print(f"Loaded manifest with {len(manifest)} entries")

    # Load location info from all scenarios
    print("Loading scenario location info...")
    location_info = load_scenario_location_info(root, manifest)
    print(f"Total scenarios: {len(location_info)}")

    # Print port statistics
    ports = defaultdict(int)
    for item in location_info:
        ports[item['port']] += 1
    print(f"Ports: {dict(ports)}")

    # Assign grid-based location groups
    assign_grid_groups(location_info, grid_size)

    # Count unique dates
    unique_dates = set(item['date'] for item in location_info)
    print(f"Recording dates: {len(unique_dates)} unique days")

    # Count unique location-time groups
    unique_groups = set(item['location_group'] for item in location_info)
    print(f"Location-time groups (grid size={grid_size}m): {len(unique_groups)}")

    # Assign splits
    coverage_stats = None
    if require_coverage:
        effective_coverage_grid = coverage_grid_size if coverage_grid_size is not None else grid_size
        coverage_stats = assign_splits_with_coverage(
            location_info, train_ratio, val_ratio, effective_coverage_grid
        )
    else:
        assign_splits(location_info, train_ratio, val_ratio)

    # Build splits dictionary
    splits = build_splits_dict(location_info)

    # Save to JSON
    output_path = root / output
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    # Print statistics
    print("\nSplit statistics:")
    total = len(location_info)
    for split_name in ['train', 'val', 'test']:
        count = sum(len(indices) for indices in splits[split_name].values())
        file_count = len(splits[split_name])
        print(f"  {split_name}: {count} scenarios ({100*count/total:.1f}%) in {file_count} files")

    # Print per-port statistics
    print("\nPer-port breakdown:")
    for split_name in ['train', 'val', 'test']:
        port_counts = defaultdict(int)
        for item in location_info:
            if item['split'] == split_name:
                port_counts[item['port']] += 1
        port_str = ', '.join(f"{k}: {v}" for k, v in sorted(port_counts.items()))
        print(f"  {split_name}: {port_str}")

    # Print temporal diversity statistics
    print("\nTemporal diversity (unique dates per split):")
    for split_name in ['train', 'val', 'test']:
        split_dates = set(item['date'] for item in location_info if item['split'] == split_name)
        print(f"  {split_name}: {len(split_dates)} unique days")

    # Show locations that appear in multiple splits (different days)
    location_splits: Dict[str, set] = defaultdict(set)
    for item in location_info:
        # Location key without date
        loc_key = f"{item['port']}_{int(item['x'] // grid_size)}_{int(item['y'] // grid_size)}"
        location_splits[loc_key].add(item['split'])
    multi_split_locations = sum(1 for splits in location_splits.values() if len(splits) > 1)
    print(f"\nLocations appearing in multiple splits (different days): {multi_split_locations}")

    # Verify each split contains each port
    print("\nVerifying stratification...")
    all_ok = True
    for split_name in ['train', 'val', 'test']:
        split_ports = set(item['port'] for item in location_info if item['split'] == split_name)
        if split_ports != set(ports.keys()):
            missing = set(ports.keys()) - split_ports
            print(f"  WARNING: {split_name} missing ports: {missing}")
            all_ok = False
    if all_ok:
        print("  All splits contain all ports.")

    # Print coverage statistics if coverage was required
    if coverage_stats is not None:
        effective_coverage_grid = coverage_grid_size if coverage_grid_size is not None else grid_size
        print_coverage_statistics(coverage_stats, effective_coverage_grid)

    print(f"\nSaved splits to {output_path}")

    # Generate heatmaps if requested
    if heatmap:
        print("\nGenerating heatmaps...")
        render_heatmaps(location_info, grid_size, root)

    return splits


def main():
    parser = argparse.ArgumentParser(
        description='Generate train/val/test splits for Venti3D dataset with location + time aware grouping'
    )
    parser.add_argument(
        '--root', type=str, required=True,
        help='Root directory containing manifest.pkl and scenarios/'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Ratio of scenarios for training (default: 0.8)'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.1,
        help='Ratio of scenarios for validation (default: 0.1)'
    )
    parser.add_argument(
        '--grid_size', type=float, default=200.0,
        help='Grid size for location clustering in meters (default: 200.0)'
    )
    parser.add_argument(
        '--output', type=str, default='splits.json',
        help='Output filename for splits JSON (default: splits.json)'
    )
    parser.add_argument(
        '--heatmap', action='store_true',
        help='Generate heatmap visualizations of ego locations'
    )
    parser.add_argument(
        '--coverage', action='store_true',
        help='Enable spatial coverage guarantee (ensures each region has val/test data, but reduces training data)'
    )
    parser.add_argument(
        '--coverage_grid_size', type=float, default=None,
        help='Grid size for coverage regions in meters (default: same as --grid_size)'
    )

    args = parser.parse_args()

    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        parser.error(f"train_ratio + val_ratio must be <= 1.0, got {args.train_ratio + args.val_ratio}")

    generate_splits(
        root=args.root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        grid_size=args.grid_size,
        output=args.output,
        heatmap=args.heatmap,
        require_coverage=args.coverage,
        coverage_grid_size=args.coverage_grid_size,
    )


if __name__ == '__main__':
    main()
