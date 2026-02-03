#!/usr/bin/env python3
"""Convert original QCNet checkpoint to Venti3D-compatible format.

This script remaps agent type and polygon type embeddings from the original
Argoverse V2 checkpoint to the Venti3D format.

Agent Type Embeddings:
    Original AV2 (10 types):
        0: vehicle       (well-trained, high sample count)
        1: pedestrian    (well-trained, high sample count)
        2: motorcyclist
        3: cyclist
        4: bus           (undertrained, low sample count)
        5: static
        6: background
        7: construction  (undertrained, low sample count)
        8: riderless_bicycle
        9: unknown

    Venti3D (6 types):
        0: vehicle
        1: trailer
        2: industrial_vehicle
        3: pedestrian
        4: static
        5: unknown

    Mapping Strategy:
        Uses well-trained vehicle embedding as fallback for undertrained types.
        This provides a better initialization than using undertrained embeddings
        (bus, construction) which may have learned noisy representations due to
        limited training examples. Fine-tuning will adapt the vehicle embedding
        to the specific behaviors of trailer and industrial_vehicle.

Polygon Type Embeddings:
    Original AV2 (4 types):
        0: VEHICLE
        1: BIKE
        2: BUS
        3: PEDESTRIAN

    Venti3D (2 types):
        0: VEHICLE
        1: PEDESTRIAN

    Mapping Strategy:
        Direct mapping for VEHICLE and PEDESTRIAN types.

Usage:
    python scripts/convert_checkpoint.py \
        --source_ckpt path/to/original_qcnet.ckpt \
        --output_ckpt path/to/venti3d_warmstart.ckpt
"""
import argparse
from pathlib import Path

import torch


# Agent type embedding configuration
AGENT_EMBEDDING_KEY = 'encoder.agent_encoder.type_a_emb.weight'
SOURCE_NUM_AGENT_TYPES = 10
TARGET_NUM_AGENT_TYPES = 6

# Mapping from Venti3D target index to original AV2 source index
# Key: target index (Venti3D), Value: source index (original AV2)
# Strategy: Use well-trained embeddings as fallback for undertrained types
AGENT_EMBEDDING_MAPPING = {
    0: 0,   # vehicle <- vehicle (well-trained, direct match)
    1: 0,   # trailer <- vehicle (well-trained fallback, fine-tune adapts)
    2: 0,   # industrial_vehicle <- vehicle (well-trained fallback, fine-tune adapts)
    3: 1,   # pedestrian <- pedestrian (well-trained, direct match)
    4: 5,   # static <- static (direct match)
    5: 9,   # unknown <- unknown (direct match)
}

# Polygon type embedding configuration
POLYGON_EMBEDDING_KEY = 'encoder.map_encoder.type_pl_emb.weight'
SOURCE_NUM_POLYGON_TYPES = 4
TARGET_NUM_POLYGON_TYPES = 2

POLYGON_EMBEDDING_MAPPING = {
    0: 0,   # VEHICLE <- VEHICLE (direct match)
    1: 3,   # PEDESTRIAN <- PEDESTRIAN (direct match)
}


def remap_embeddings(source_weights: torch.Tensor,
                     mapping: dict,
                     target_size: int) -> torch.Tensor:
    """Remap embedding weights from source to target indices.

    Args:
        source_weights: Original embedding weights, shape (source_size, hidden_dim)
        mapping: Dict mapping target_idx -> source_idx
        target_size: Number of types in target embedding

    Returns:
        Remapped embedding weights, shape (target_size, hidden_dim)
    """
    hidden_dim = source_weights.shape[1]
    target_weights = torch.zeros(target_size, hidden_dim, dtype=source_weights.dtype)

    for target_idx, source_idx in mapping.items():
        target_weights[target_idx] = source_weights[source_idx]

    return target_weights


def convert_checkpoint(source_path: Path, output_path: Path) -> None:
    """Convert checkpoint from original AV2 format to Venti3D format.

    Args:
        source_path: Path to source checkpoint file
        output_path: Path to save converted checkpoint
    """
    print(f"Loading source checkpoint: {source_path}")
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats (Lightning vs raw state dict)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        is_lightning = True
    else:
        state_dict = checkpoint
        is_lightning = False

    # -------------------------------------------------------------------------
    # Agent Type Embeddings
    # -------------------------------------------------------------------------
    if AGENT_EMBEDDING_KEY not in state_dict:
        raise KeyError(
            f"Agent embedding key '{AGENT_EMBEDDING_KEY}' not found in checkpoint. "
            f"Available keys: {list(state_dict.keys())[:10]}..."
        )

    agent_source_weights = state_dict[AGENT_EMBEDDING_KEY]
    print(f"\nAgent type embeddings:")
    print(f"  Source shape: {agent_source_weights.shape}")

    if agent_source_weights.shape[0] != SOURCE_NUM_AGENT_TYPES:
        print(f"  Warning: Expected {SOURCE_NUM_AGENT_TYPES} source types, "
              f"found {agent_source_weights.shape[0]}")

    # Remap agent embeddings
    agent_target_weights = remap_embeddings(
        agent_source_weights, AGENT_EMBEDDING_MAPPING, TARGET_NUM_AGENT_TYPES
    )
    print(f"  Target shape: {agent_target_weights.shape}")

    # Verify agent embedding copy
    print("\n  Verification (first 3 values of each type):")
    agent_type_names = ['vehicle', 'trailer', 'industrial_vehicle',
                        'pedestrian', 'static', 'unknown']
    agent_source_names = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist',
                          'bus', 'static', 'background', 'construction',
                          'riderless_bicycle', 'unknown']

    for target_idx, source_idx in AGENT_EMBEDDING_MAPPING.items():
        target_vals = agent_target_weights[target_idx, :3].tolist()
        target_name = agent_type_names[target_idx]
        source_name = agent_source_names[source_idx]
        print(f"    {target_name}[{target_idx}] <- {source_name}[{source_idx}]: "
              f"{[f'{v:.4f}' for v in target_vals]}")
        assert torch.allclose(agent_target_weights[target_idx], agent_source_weights[source_idx]), \
            f"Mismatch for agent type {target_name}"

    state_dict[AGENT_EMBEDDING_KEY] = agent_target_weights

    # -------------------------------------------------------------------------
    # Polygon Type Embeddings
    # -------------------------------------------------------------------------
    if POLYGON_EMBEDDING_KEY not in state_dict:
        raise KeyError(
            f"Polygon embedding key '{POLYGON_EMBEDDING_KEY}' not found in checkpoint. "
            f"Available keys: {list(state_dict.keys())[:10]}..."
        )

    polygon_source_weights = state_dict[POLYGON_EMBEDDING_KEY]
    print(f"\nPolygon type embeddings:")
    print(f"  Source shape: {polygon_source_weights.shape}")

    if polygon_source_weights.shape[0] != SOURCE_NUM_POLYGON_TYPES:
        print(f"  Warning: Expected {SOURCE_NUM_POLYGON_TYPES} source types, "
              f"found {polygon_source_weights.shape[0]}")

    # Remap polygon embeddings
    polygon_target_weights = remap_embeddings(
        polygon_source_weights, POLYGON_EMBEDDING_MAPPING, TARGET_NUM_POLYGON_TYPES
    )
    print(f"  Target shape: {polygon_target_weights.shape}")

    # Verify polygon embedding copy
    print("\n  Verification (first 3 values of each type):")
    polygon_type_names = ['VEHICLE', 'PEDESTRIAN']
    polygon_source_names = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']

    for target_idx, source_idx in POLYGON_EMBEDDING_MAPPING.items():
        target_vals = polygon_target_weights[target_idx, :3].tolist()
        target_name = polygon_type_names[target_idx]
        source_name = polygon_source_names[source_idx]
        print(f"    {target_name}[{target_idx}] <- {source_name}[{source_idx}]: "
              f"{[f'{v:.4f}' for v in target_vals]}")
        assert torch.allclose(polygon_target_weights[target_idx], polygon_source_weights[source_idx]), \
            f"Mismatch for polygon type {target_name}"

    state_dict[POLYGON_EMBEDDING_KEY] = polygon_target_weights

    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"\nSaved converted checkpoint to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  Agent types:   {SOURCE_NUM_AGENT_TYPES} -> {TARGET_NUM_AGENT_TYPES}")
    print(f"  Polygon types: {SOURCE_NUM_POLYGON_TYPES} -> {TARGET_NUM_POLYGON_TYPES}")
    print(f"  Hidden dim:    {agent_target_weights.shape[1]}")
    print(f"  Format:        {'PyTorch Lightning' if is_lightning else 'Raw state dict'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Convert QCNet checkpoint from original AV2 to Venti3D format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--source_ckpt',
        type=Path,
        required=True,
        help='Path to source checkpoint (original QCNet trained on AV2)'
    )
    parser.add_argument(
        '--output_ckpt',
        type=Path,
        required=True,
        help='Path to save converted checkpoint'
    )

    args = parser.parse_args()

    if not args.source_ckpt.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {args.source_ckpt}")

    if args.output_ckpt.exists():
        response = input(f"Output file {args.output_ckpt} exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    convert_checkpoint(args.source_ckpt, args.output_ckpt)


if __name__ == '__main__':
    main()
