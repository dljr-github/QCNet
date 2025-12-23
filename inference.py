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
import warnings
from argparse import ArgumentParser
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder


@dataclass
class OptimizationConfig:
    """Configuration for inference optimizations."""
    use_amp: bool = False
    amp_dtype: str = "float16"

    def validate(self, device: torch.device) -> None:
        """Validate configuration and warn about incompatibilities."""
        if self.use_amp and device.type != 'cuda':
            warnings.warn("AMP requires CUDA device. Disabling AMP.")
            self.use_amp = False


def get_amp_context(config: OptimizationConfig, device: torch.device):
    """Get the appropriate AMP context manager."""
    if not config.use_amp or device.type != 'cuda':
        return nullcontext()

    dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    return torch.amp.autocast(device_type='cuda', dtype=dtype)


def measure_runtime(
    model,
    dataloader,
    device,
    num_warmup=10,
    num_runs=100,
    config: Optional[OptimizationConfig] = None
) -> Dict[str, Any]:
    """
    Measure inference runtime performance with optional AMP optimization.

    Returns:
        Dictionary with latency stats, throughput, memory usage, and config info.
    """
    if config is None:
        config = OptimizationConfig()

    config.validate(device)
    model.eval()

    # Get AMP context
    amp_context = get_amp_context(config, device)

    # Reset GPU memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Get a single batch for benchmarking
    batch = next(iter(dataloader)).to(device)

    # Warmup runs
    print(f"Warming up with {num_warmup} runs...")
    print(f"  AMP: {config.use_amp}" +
          (f" (dtype={config.amp_dtype})" if config.use_amp else ""))

    with torch.no_grad():
        for _ in range(num_warmup):
            with amp_context:
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

            with amp_context:
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

    # Build results
    results = {
        'latencies': latencies,
        'batch_size': batch_size,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'memory_allocated_mb': max_memory_allocated,
        'memory_reserved_mb': max_memory_reserved,
        'config': {
            'use_amp': config.use_amp,
            'amp_dtype': config.amp_dtype if config.use_amp else None,
        }
    }

    # Print results
    print_benchmark_results(results, num_runs)

    return results


def print_benchmark_results(results: Dict[str, Any], num_runs: int) -> None:
    """Print formatted benchmark results."""
    latencies = results['latencies']
    batch_size = results['batch_size']

    print("\n" + "="*60)
    print("RUNTIME PERFORMANCE")
    print("="*60)
    print(f"Device: {results['device']}")
    if results['gpu_name']:
        print(f"GPU: {results['gpu_name']}")
    print(f"Batch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print("-"*60)
    cfg = results['config']
    print(f"AMP: {cfg['use_amp']}" +
          (f" (dtype={cfg['amp_dtype']})" if cfg['use_amp'] else ""))
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
    print(f"  {batch_size * 1000 / latencies.mean():.2f} samples/sec")
    print("-"*60)
    print("GPU MEMORY")
    print(f"  Peak allocated: {results['memory_allocated_mb']:.2f} MB")
    print(f"  Peak reserved: {results['memory_reserved_mb']:.2f} MB")
    print("="*60 + "\n")


def measure_runtime_cached(
    model,
    dataloader,
    device,
    num_warmup: int = 10,
    num_runs: int = 100,
    use_amp: bool = False
) -> Dict[str, Any]:
    """
    Measure inference with cached map encoding.

    This simulates deployment where the map is static and can be pre-computed.
    Only agent encoder + decoder are timed (map encoder runs once).
    """
    model.eval()

    # Get AMP context
    amp_dtype = torch.float16 if use_amp else None
    amp_context = torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()

    # Reset GPU memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Get a single batch
    batch = next(iter(dataloader)).to(device)

    # Pre-compute and cache map encoding (done once, not timed)
    print("Pre-computing map encoding (cached)...")
    with torch.no_grad():
        with amp_context:
            map_enc = model.encoder.map_encoder(batch)

    # Warmup runs (agent encoder + decoder only)
    print(f"Warming up with {num_warmup} runs...")
    print(f"  Map cache: enabled")
    print(f"  AMP: {use_amp}")

    with torch.no_grad():
        for _ in range(num_warmup):
            with amp_context:
                agent_enc = model.encoder.agent_encoder(batch, map_enc)
                scene_enc = {**map_enc, **agent_enc}
                _ = model.decoder(batch, scene_enc)

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs (agent encoder + decoder only)
    print(f"Benchmarking with {num_runs} runs...")
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            with amp_context:
                agent_enc = model.encoder.agent_encoder(batch, map_enc)
                scene_enc = {**map_enc, **agent_enc}
                _ = model.decoder(batch, scene_enc)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    batch_size = batch.num_graphs

    # Get GPU memory stats
    if device.type == 'cuda':
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        max_memory_allocated = 0
        max_memory_reserved = 0

    # Build results
    results = {
        'latencies': latencies,
        'batch_size': batch_size,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'memory_allocated_mb': max_memory_allocated,
        'memory_reserved_mb': max_memory_reserved,
        'config': {
            'use_amp': use_amp,
            'amp_dtype': 'float16' if use_amp else None,
            'use_cache': True,
        }
    }

    # Print results
    print_benchmark_results_cached(results, num_runs)

    return results


def print_benchmark_results_cached(results: Dict[str, Any], num_runs: int) -> None:
    """Print formatted benchmark results for cached inference."""
    latencies = results['latencies']
    batch_size = results['batch_size']

    print("\n" + "="*60)
    print("RUNTIME PERFORMANCE (CACHED MAP)")
    print("="*60)
    print(f"Device: {results['device']}")
    if results['gpu_name']:
        print(f"GPU: {results['gpu_name']}")
    print(f"Batch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print("-"*60)
    cfg = results['config']
    print(f"Map cache: enabled")
    print(f"AMP: {cfg['use_amp']}" +
          (f" (dtype={cfg['amp_dtype']})" if cfg['use_amp'] else ""))
    print("-"*60)
    print("LATENCY (agent encoder + decoder only)")
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
    print(f"  {batch_size * 1000 / latencies.mean():.2f} samples/sec")
    print("-"*60)
    print("GPU MEMORY")
    print(f"  Peak allocated: {results['memory_allocated_mb']:.2f} MB")
    print(f"  Peak reserved: {results['memory_reserved_mb']:.2f} MB")
    print("="*60 + "\n")


def run_comparison_benchmark(
    model,
    dataloader,
    device,
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmarks comparing baseline, AMP, cached, and cached+AMP.

    Returns dictionary mapping config name to results.
    """
    results = {}
    model.eval()

    # Benchmark 1: Baseline (full model, no optimizations)
    print(f"\n{'='*60}")
    print("Running benchmark: baseline")
    print('='*60)
    try:
        results['baseline'] = measure_runtime(
            model, dataloader, device,
            num_warmup=num_warmup,
            num_runs=num_runs,
            config=OptimizationConfig()
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        results['baseline'] = {'error': str(e)}

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Benchmark 2: AMP FP16 (full model with AMP)
    print(f"\n{'='*60}")
    print("Running benchmark: amp_fp16")
    print('='*60)
    try:
        results['amp_fp16'] = measure_runtime(
            model, dataloader, device,
            num_warmup=num_warmup,
            num_runs=num_runs,
            config=OptimizationConfig(use_amp=True, amp_dtype='float16')
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        results['amp_fp16'] = {'error': str(e)}

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Benchmark 3: Cached map encoding (no AMP)
    print(f"\n{'='*60}")
    print("Running benchmark: cached")
    print('='*60)
    try:
        results['cached'] = measure_runtime_cached(
            model, dataloader, device,
            num_warmup=num_warmup,
            num_runs=num_runs,
            use_amp=False
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        results['cached'] = {'error': str(e)}

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Benchmark 4: Cached map encoding + AMP
    print(f"\n{'='*60}")
    print("Running benchmark: cached_amp")
    print('='*60)
    try:
        results['cached_amp'] = measure_runtime_cached(
            model, dataloader, device,
            num_warmup=num_warmup,
            num_runs=num_runs,
            use_amp=True
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        results['cached_amp'] = {'error': str(e)}

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Print comparison summary
    print_comparison_summary(results)

    return results


def print_comparison_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary comparison table."""
    print("\n" + "="*70)
    print("OPTIMIZATION COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Config':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Speedup':<12}")
    print("-"*70)

    baseline_mean = None
    if 'baseline' in results and 'latencies' in results['baseline']:
        baseline_mean = results['baseline']['latencies'].mean()

    for name, result in results.items():
        if 'error' in result:
            print(f"{name:<20} {'ERROR':<12} {'-':<12} {'-':<12}")
        elif 'latencies' in result:
            mean = result['latencies'].mean()
            std = result['latencies'].std()
            speedup = baseline_mean / mean if baseline_mean else 1.0
            print(f"{name:<20} {mean:<12.2f} {std:<12.2f} {speedup:<12.2f}x")

    print("="*70 + "\n")


def run_profiler(
    model,
    dataloader,
    device,
    num_warmup: int = 3,
    num_profile: int = 5,
    use_amp: bool = False
) -> None:
    """
    Profile model inference to show time breakdown by operation.

    Helps verify:
    1. Time spent in graph operations (radius, message passing)
    2. Time spent in attention/MLP layers
    """
    model.eval()

    # Get AMP context
    amp_dtype = torch.float16 if use_amp else None
    amp_context = torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()

    # Get a single batch for profiling
    batch = next(iter(dataloader)).to(device)

    # Warmup runs
    print(f"Warming up with {num_warmup} runs...")
    with torch.no_grad():
        for _ in range(num_warmup):
            with amp_context:
                _ = model(batch)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Profile runs
    print(f"Profiling with {num_profile} runs...")
    print(f"  AMP: {use_amp}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(num_profile):
                with amp_context:
                    _ = model(batch)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

    # Print profiler results
    print("\n" + "="*80)
    print("PROFILER RESULTS")
    print("="*80)

    # Get key averages sorted by CUDA time
    key_averages = prof.key_averages()

    print("\nTop 25 operations by CUDA time:\n")
    print(f"{'Operation':<45} {'CUDA time':>12} {'CPU time':>12} {'Calls':>8}")
    print("-"*80)

    # Sort by CUDA time total
    sorted_ops = sorted(
        key_averages,
        key=lambda x: x.cuda_time_total,
        reverse=True
    )

    for op in sorted_ops[:25]:
        cuda_time = op.cuda_time_total / 1000  # Convert to ms
        cpu_time = op.cpu_time_total / 1000
        name = op.key[:44] if len(op.key) > 44 else op.key
        print(f"{name:<45} {cuda_time:>10.2f}ms {cpu_time:>10.2f}ms {op.count:>8}")

    # Calculate category summaries
    print("\n" + "-"*80)
    print("CATEGORY SUMMARY")
    print("-"*80)

    categories = {
        'Graph Ops (radius, scatter)': ['radius', 'scatter', 'segment', 'gather', 'index'],
        'Attention (mm, bmm, softmax)': ['mm', 'bmm', 'softmax', 'baddbmm'],
        'Linear/MLP': ['linear', 'addmm', 'relu', 'gelu', 'dropout'],
        'GRU/RNN': ['gru', 'rnn', 'lstm'],
        'Normalization': ['layer_norm', 'batch_norm', 'norm'],
        'Embedding': ['embedding', 'fourier'],
        'Memory': ['copy', 'clone', 'contiguous', 'to', 'cat', 'stack'],
    }

    category_times = {cat: 0.0 for cat in categories}
    total_cuda_time = sum(op.cuda_time_total for op in key_averages) / 1000

    for op in key_averages:
        op_name = op.key.lower()
        cuda_time = op.cuda_time_total / 1000

        for cat, keywords in categories.items():
            if any(kw in op_name for kw in keywords):
                category_times[cat] += cuda_time
                break
        else:
            if 'Other' not in category_times:
                category_times['Other'] = 0.0
            category_times['Other'] += cuda_time

    # Sort by time and print
    sorted_cats = sorted(category_times.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Category':<35} {'CUDA time':>12} {'Percentage':>12}")
    print("-"*60)

    for cat, time_ms in sorted_cats:
        if time_ms > 0:
            pct = (time_ms / total_cuda_time * 100) if total_cuda_time > 0 else 0
            print(f"{cat:<35} {time_ms:>10.2f}ms {pct:>10.1f}%")

    print("-"*60)
    print(f"{'Total':<35} {total_cuda_time:>10.2f}ms {100.0:>10.1f}%")
    print("="*80 + "\n")


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

    # Optimization arguments
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable automatic mixed precision (FP16)')
    parser.add_argument('--use_cache', action='store_true',
                        help='Cache map encoding (for fixed location deployment)')
    parser.add_argument('--compare_optimizations', action='store_true',
                        help='Run comparison benchmark (baseline, AMP, cached, cached+AMP)')
    parser.add_argument('--profile', action='store_true',
                        help='Run profiler to show operation time breakdown')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Print PyTorch version and capability info
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(device)}")

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

    # Run profiler, comparison benchmark, or single benchmark
    if args.profile:
        # Run profiler to show operation breakdown
        run_profiler(
            model, dataloader, device,
            num_warmup=args.num_warmup,
            num_profile=min(args.num_runs, 10),  # Limit profile runs
            use_amp=args.use_amp
        )
    elif args.compare_optimizations:
        # Run comparison of all optimization configs
        results = run_comparison_benchmark(
            model, dataloader, device,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs
        )
    elif args.use_cache:
        # Run with cached map encoding
        results = measure_runtime_cached(
            model, dataloader, device,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            use_amp=args.use_amp
        )
    else:
        # Standard benchmark
        opt_config = OptimizationConfig(
            use_amp=args.use_amp,
        )

        results = measure_runtime(
            model, dataloader, device,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            config=opt_config
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
