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
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder


def measure_runtime(model, dataloader, device, num_warmup=10, num_runs=100):
    """Measure inference runtime performance."""
    model.eval()

    # Reset GPU memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Get a single batch for benchmarking
    batch = next(iter(dataloader)).to(device)

    # Warmup runs
    print(f"Warming up with {num_warmup} runs...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(batch)

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    print(f"Benchmarking with {num_runs} runs...")
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(batch)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    batch_size = batch.num_graphs

    # Get GPU memory stats
    if device.type == 'cuda':
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # MB
    else:
        max_memory_allocated = 0
        max_memory_reserved = 0

    print("\n" + "="*60)
    print("RUNTIME PERFORMANCE")
    print("="*60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print("-"*60)
    print("LATENCY")
    print(f"  Mean: {latencies.mean():.2f} ms")
    print(f"  Std: {latencies.std():.2f} ms")
    print(f"  Min: {latencies.min():.2f} ms")
    print(f"  Max: {latencies.max():.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95: {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99: {np.percentile(latencies, 99):.2f} ms")
    print("-"*60)
    print("THROUGHPUT")
    print(f"  {1000 / latencies.mean():.2f} batches/sec")
    print(f"  {batch_size * 1000 / latencies.mean():.2f} samples/sec ({1000 / latencies.mean():.1f} Hz)")
    print("-"*60)
    print("GPU MEMORY")
    print(f"  Peak allocated: {max_memory_allocated:.2f} MB")
    print(f"  Peak reserved: {max_memory_reserved:.2f} MB")
    print("="*60 + "\n")

    return latencies


def visualize_prediction(model, data, device, save_path=None, has_ground_truth=True):
    """Visualize a single prediction with map, history, and predicted trajectories."""
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    # Extract predictions
    traj_pred = output['traj_refine']  # [num_agents, num_modes, num_future_steps, 2]
    pi = output['pi']  # [num_agents, num_modes]
    pi_softmax = F.softmax(pi, dim=-1)

    # Get the focal agent (first agent marked for prediction)
    eval_mask = data['agent']['predict_mask'][:, model.num_historical_steps - 1]
    focal_idx = torch.where(eval_mask)[0][0].item()

    # Get trajectories for focal agent
    traj_pred_focal = traj_pred[focal_idx].cpu().numpy()  # [num_modes, num_future_steps, 2]
    probs = pi_softmax[focal_idx].cpu().numpy()  # [num_modes]

    # Get historical trajectory
    hist_traj = data['agent']['position'][:, :model.num_historical_steps].cpu().numpy()
    focal_hist = hist_traj[focal_idx]  # [num_historical_steps, 2]

    # Get ground truth future trajectory (only for val split)
    focal_gt = None
    if has_ground_truth:
        gt_traj = data['agent']['position'][:, model.num_historical_steps:].cpu().numpy()
        focal_gt = gt_traj[focal_idx]  # [num_future_steps, 2]

    # Get map data
    map_pos = data['map_polygon']['position'].cpu().numpy()  # [num_points, num_pts_per_pl, 2]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot map lanes
    for i in range(len(map_pos)):
        lane = map_pos[i]
        ax.plot(lane[:, 0], lane[:, 1], 'gray', linewidth=0.5, alpha=0.5)

    # Plot other agents' history
    for i in range(len(hist_traj)):
        if i != focal_idx:
            ax.plot(hist_traj[i, :, 0], hist_traj[i, :, 1],
                   color='blue', linewidth=1, alpha=0.3)
            ax.scatter(hist_traj[i, -1, 0], hist_traj[i, -1, 1],
                      c='blue', s=20, alpha=0.3)

    # Plot focal agent history
    ax.plot(focal_hist[:, 0], focal_hist[:, 1],
           color='blue', linewidth=2, label='History')
    ax.scatter(focal_hist[-1, 0], focal_hist[-1, 1],
              c='blue', s=100, marker='o', zorder=5)

    # Plot ground truth (only if available)
    if focal_gt is not None:
        ax.plot(focal_gt[:, 0], focal_gt[:, 1],
               color='green', linewidth=2, linestyle='--', label='Ground Truth')
        ax.scatter(focal_gt[-1, 0], focal_gt[-1, 1],
                  c='green', s=100, marker='*', zorder=5)

    # Plot predicted trajectories (sorted by probability)
    sorted_idx = np.argsort(probs)[::-1]
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(sorted_idx)))

    for rank, idx in enumerate(sorted_idx):
        traj = traj_pred_focal[idx]
        prob = probs[idx]
        color = colors[len(sorted_idx) - 1 - rank]

        # Connect prediction to current position
        full_traj = np.vstack([focal_hist[-1:], traj])
        ax.plot(full_traj[:, 0], full_traj[:, 1],
               color=color, linewidth=2 - rank * 0.2, alpha=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 1],
                  c=[color], s=50, marker='o', zorder=4)

        # Annotate top predictions
        if rank < 3:
            ax.annotate(f'{prob:.2f}', (traj[-1, 0], traj[-1, 1]),
                       fontsize=8, ha='center', va='bottom')

    # Add legend for predictions
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='blue', linewidth=2, label='History'),
        Line2D([0], [0], color='red', linewidth=2, label='Predictions'),
    ]
    if focal_gt is not None:
        legend_handles.insert(1, Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Ground Truth'))
    ax.legend(handles=legend_handles)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    scenario_id = data['scenario_id'][0] if hasattr(data, 'scenario_id') else 'unknown'
    ax.set_title(f'QCNet Prediction - Scenario: {scenario_id}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()

    return fig


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='QCNet')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_warmup', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--num_vis', type=int, default=5, help='Number of scenarios to visualize')
    parser.add_argument('--save_dir', type=str, default='vis_outputs', help='Directory to save visualizations')
    parser.add_argument('--benchmark_only', action='store_true', help='Only run benchmark, skip visualization')
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model = model.to(device)
    model.eval()

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Load dataset
    print(f"Loading {args.split} dataset from {args.root}...")
    dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](
        root=args.root,
        split=args.split,
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)} scenarios")
    print(f"Dataloader batches: {len(dataloader)}")

    # Measure runtime
    latencies = measure_runtime(
        model, dataloader, device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs
    )

    # Visualize if requested
    if args.visualize and not args.benchmark_only:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)

        has_ground_truth = args.split == 'val'
        print(f"\nGenerating {args.num_vis} visualizations...")
        if not has_ground_truth:
            print("Note: Test set has no ground truth, only predictions will be shown.")

        vis_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        for i, data in enumerate(vis_loader):
            if i >= args.num_vis:
                break

            scenario_id = data['scenario_id'][0]
            save_path = save_dir / f'prediction_{scenario_id}.png'

            print(f"\nVisualizing scenario {i+1}/{args.num_vis}: {scenario_id}")
            visualize_prediction(model, data, device, save_path=save_path, has_ground_truth=has_ground_truth)

    print("\nDone!")


if __name__ == '__main__':
    main()
