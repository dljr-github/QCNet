"""
Rolling horizon visualization for QCNet trajectory prediction.

Simulates real-world deployment by running inference continuously at each timestep
as new observations become available. Unlike visualize_video.py which runs inference
once at t=0, this shows how predictions evolve as the prediction window rolls forward.

Usage:
    # By scenario index
    python visualize_scenario.py \
        --root /path/to/dataset \
        --ckpt_path /path/to/checkpoint.ckpt \
        --scenario_index 1 \
        --dataset venti3d --split test

    # By scenario ID
    python visualize_scenario.py \
        --root /path/to/dataset \
        --ckpt_path /path/to/checkpoint.ckpt \
        --scenario_id <scenario_id> \
        --dataset venti3d --split test

    # All scenarios (no --scenario_index or --scenario_id)
    python visualize_scenario.py \
        --root /path/to/dataset \
        --ckpt_path /path/to/checkpoint.ckpt \
        --dataset venti3d --split test
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import subprocess
import shutil
import os

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from predictors import QCNet
from datasets import ArgoverseV2Dataset, Venti3DDataset
from visualization_utils import (
    COLORS, CoordinateTransform,
    draw_agent, draw_map, draw_scale_bar,
    draw_predictions, draw_ground_truth, draw_history_trail, draw_ego_marker
)


def find_scenario_by_id(dataset, scenario_id: str) -> Optional[int]:
    """
    Find the index of a scenario by its ID.

    Args:
        dataset: The dataset to search
        scenario_id: The scenario ID to find

    Returns:
        The index of the scenario, or None if not found
    """
    # For processed datasets, we need to check each scenario
    # This is O(n) but necessary since datasets don't have an index
    for i in range(len(dataset)):
        data = dataset[i]
        if hasattr(data, 'scenario_id'):
            data_id = data.scenario_id[0] if isinstance(data.scenario_id, list) else data.scenario_id
            if data_id == scenario_id:
                return i
    return None


def slice_data_for_timestep(data: HeteroData, current_timestep: int,
                           num_historical_steps: int) -> HeteroData:
    """
    Create a temporal slice of the data for rolling horizon inference.

    At timestep T, we use positions [T - H + 1 : T + 1] as history,
    where H = num_historical_steps.

    Args:
        data: Original scenario data with full timeline
        current_timestep: The current frame index (T)
        num_historical_steps: Number of historical steps the model expects (H)

    Returns:
        A new HeteroData object sliced for inference at timestep T
    """
    H = num_historical_steps
    start_idx = current_timestep - H + 1
    end_idx = current_timestep + 1  # Exclusive

    # Create a new HeteroData object
    sliced = HeteroData()

    # Copy agent data with temporal slicing
    if 'agent' in data.node_types:
        sliced['agent'] = {}

        # Slice temporal fields
        for key in ['position', 'heading', 'velocity', 'valid_mask']:
            if key in data['agent']:
                tensor = data['agent'][key]
                # Assume shape is (num_agents, time_steps, ...) for position/velocity
                # or (num_agents, time_steps) for heading/valid_mask
                sliced['agent'][key] = tensor[:, start_idx:end_idx].clone()

        # Copy non-temporal fields
        for key in ['type', 'category', 'num_nodes']:
            if key in data['agent']:
                sliced['agent'][key] = data['agent'][key].clone() if torch.is_tensor(data['agent'][key]) else data['agent'][key]

        # Rebuild predict_mask for the slice
        # The model uses predict_mask to identify which agents to predict
        # In the sliced data, the "current" time is at index H-1 (last position in history)
        if 'predict_mask' in data['agent']:
            # Original predict_mask shape: (num_agents, num_timesteps)
            # For the slice, we need predict_mask at the new "current" time
            original_mask = data['agent']['predict_mask']
            num_agents = original_mask.shape[0]
            slice_timesteps = end_idx - start_idx

            # Create new predict_mask where only the last timestep matters
            new_predict_mask = torch.zeros(num_agents, slice_timesteps, dtype=torch.bool)
            # Copy relevant portion of original predict_mask
            new_predict_mask[:, :] = original_mask[:, start_idx:end_idx]
            sliced['agent']['predict_mask'] = new_predict_mask

    # Copy map data unchanged (static)
    for node_type in data.node_types:
        if node_type.startswith('map_') and node_type not in sliced.node_types:
            sliced[node_type] = {}
            for key, value in data[node_type].items():
                if torch.is_tensor(value):
                    sliced[node_type][key] = value.clone()
                else:
                    sliced[node_type][key] = value

    # Copy edge data unchanged
    for edge_type in data.edge_types:
        sliced[edge_type] = {}
        for key, value in data[edge_type].items():
            if torch.is_tensor(value):
                sliced[edge_type][key] = value.clone()
            else:
                sliced[edge_type][key] = value

    # Copy scenario metadata
    if hasattr(data, 'scenario_id'):
        sliced.scenario_id = data.scenario_id

    return sliced


def run_rolling_inference(model, data: HeteroData, device: torch.device,
                         scored_indices: np.ndarray) -> List[Optional[Dict]]:
    """
    Run inference at each timestep where full history is available.

    Args:
        model: The QCNet model
        data: Full scenario data
        device: Device to run inference on
        scored_indices: Array of indices for all scored agents (agents needing predictions)

    Returns:
        List of prediction results for each frame, None for frames without predictions.
        Each result contains 'predictions' and 'probs' dicts keyed by agent index.
    """
    model.eval()

    num_historical_steps = model.num_historical_steps
    total_timesteps = data['agent']['position'].shape[1]

    results = []

    # For frames 0 to H-2, we don't have enough history
    # Predictions start at frame H-1 (index num_historical_steps - 1)
    for frame_idx in range(total_timesteps):
        if frame_idx < num_historical_steps - 1:
            # Not enough history yet
            results.append(None)
            continue

        # Slice data for this timestep
        sliced_data = slice_data_for_timestep(data, frame_idx, num_historical_steps)
        sliced_data = sliced_data.to(device)

        with torch.no_grad():
            output = model(sliced_data)

        # Extract predictions for all scored agents
        predictions_dict = {}
        probs_dict = {}

        scales_dict = {}

        for scored_idx in scored_indices:
            predictions_local = output['loc_refine_pos'][scored_idx].cpu().numpy()  # (num_modes, future_steps, 2)
            probs = F.softmax(output['pi'][scored_idx], dim=-1).cpu().numpy()  # (num_modes,)

            # Extract uncertainty scales if available
            if 'scale_refine_pos' in output:
                scales = output['scale_refine_pos'][scored_idx].cpu().numpy()  # (num_modes, future_steps, 2)
            else:
                scales = None

            # Transform predictions to world coordinates
            # Current position and heading are at the last index of the slice (H-1)
            current_pos = sliced_data['agent']['position'][scored_idx, -1, :2].cpu().numpy()
            current_heading = sliced_data['agent']['heading'][scored_idx, -1].cpu().numpy()

            cos_t, sin_t = np.cos(current_heading), np.sin(current_heading)
            rot_mat = np.array([[cos_t, -sin_t],
                               [sin_t, cos_t]])

            predictions_world = np.zeros_like(predictions_local)
            for mode_idx in range(predictions_local.shape[0]):
                rotated = predictions_local[mode_idx] @ rot_mat.T
                predictions_world[mode_idx] = rotated + current_pos

            predictions_dict[scored_idx] = predictions_world
            probs_dict[scored_idx] = probs
            scales_dict[scored_idx] = scales

        results.append({
            'predictions': predictions_dict,
            'probs': probs_dict,
            'scales': scales_dict,
        })

    return results


def run_rolling_inference_batch(model, dataloader, device: torch.device,
                                num_scenarios: int, dataset_type: str) -> List[Dict]:
    """
    Run rolling inference for all scenarios on GPU, store results as numpy arrays.

    This is the GPU-intensive phase that must run sequentially. Each scenario
    requires ~N inference calls where N = total_timesteps - num_historical_steps + 1.

    Args:
        model: The QCNet model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        num_scenarios: Maximum number of scenarios to process

    Returns:
        List of dicts, each containing all data needed to render a scenario
    """
    model.eval()
    inference_results = []
    data_iter = iter(dataloader)

    num_historical_steps = model.num_historical_steps

    for i in range(num_scenarios):
        try:
            data = next(data_iter)
        except StopIteration:
            print(f"  Only {i} scenarios available")
            break

        # Extract common data as numpy arrays (before moving to device)
        positions = data['agent']['position'][:, :, :2].cpu().numpy()
        headings = data['agent']['heading'].cpu().numpy()
        agent_types = data['agent']['type'].cpu().numpy()
        valid_mask = data['agent']['valid_mask'].cpu().numpy()

        # Create visualization mask from actual position data (not vector_repr-modified valid_mask)
        # An agent is visible if its position is not (0, 0) - the default/unset value
        vis_mask = ~((positions[:, :, 0] == 0) & (positions[:, :, 1] == 0))

        # Extract agent IDs for per-agent video naming
        num_agents = positions.shape[0]
        if 'id' in data['agent']:
            raw_ids = data['agent']['id']
            # Handle various formats: tensor, nested list (from batching), or flat list
            if torch.is_tensor(raw_ids):
                agent_ids = raw_ids.cpu().tolist()
            elif isinstance(raw_ids, (list, tuple)):
                # DataLoader may nest the list: [[id1, id2, ...]] for batch_size=1
                if len(raw_ids) == 1 and isinstance(raw_ids[0], (list, tuple)):
                    agent_ids = list(raw_ids[0])
                else:
                    agent_ids = list(raw_ids)
            else:
                agent_ids = [str(i) for i in range(num_agents)]
            # Validate length matches number of agents
            if len(agent_ids) != num_agents:
                print(f"  Warning: agent_ids length ({len(agent_ids)}) != num_agents ({num_agents}), using indices")
                agent_ids = [str(i) for i in range(num_agents)]
        else:
            agent_ids = [str(i) for i in range(num_agents)]

        # Extract bounding box dimensions if available
        if 'length' in data['agent']:
            lengths = data['agent']['length'].cpu().numpy()
        else:
            lengths = None
        if 'width' in data['agent']:
            widths = data['agent']['width'].cpu().numpy()
        else:
            widths = None

        total_timesteps = positions.shape[1]

        # Get focal indices
        agent_categories = data['agent']['category'].cpu().numpy()
        scored_indices = np.where(agent_categories >= 2)[0]
        if len(scored_indices) == 0:
            scored_indices = np.array([0])

        # Get ego (AV) index
        av_index = data['agent'].get('av_index', 0)
        if torch.is_tensor(av_index):
            av_index = av_index.item()

        # Extract map data
        map_cache = {}
        if 'map_point' in data.node_types:
            map_cache['point_pos'] = data['map_point']['position'].cpu().numpy()[:, :2]
            if 'type' in data['map_point']:
                map_cache['point_type'] = data['map_point']['type'].cpu().numpy()
            if 'side' in data['map_point']:
                map_cache['point_side'] = data['map_point']['side'].cpu().numpy()
            if ('map_point', 'to', 'map_polygon') in data.edge_types:
                map_cache['pt_to_pl'] = data['map_point', 'to', 'map_polygon']['edge_index'].cpu().numpy()

        # Get scenario ID
        if hasattr(data, 'scenario_id'):
            scenario_id = data.scenario_id[0] if isinstance(data.scenario_id, list) else data.scenario_id
        else:
            scenario_id = f"scenario_{i:04d}"

        print(f"  Rolling inference {i+1}/{num_scenarios}: {scenario_id} ({total_timesteps} timesteps)")

        # Debug: show vis_mask sums for ALL frames
        print(f"    valid_mask shape: {valid_mask.shape}, vis_mask shape: {vis_mask.shape}")
        vis_sums = [vis_mask[:, t].sum() for t in range(min(20, vis_mask.shape[1]))]
        print(f"    vis_mask sums for frames 0-19: {vis_sums}")
        # Check for any all-zero frames
        zero_frames = [t for t in range(vis_mask.shape[1]) if vis_mask[:, t].sum() == 0]
        if zero_frames:
            print(f"    WARNING: frames with 0 visible agents: {zero_frames[:20]}{'...' if len(zero_frames) > 20 else ''}")

        # Run rolling inference for this scenario
        rolling_predictions = run_rolling_inference(model, data, device, scored_indices)

        inference_results.append({
            'scenario_idx': i,
            'scenario_id': scenario_id,
            'positions': positions,
            'headings': headings,
            'agent_types': agent_types,
            'valid_mask': valid_mask,
            'vis_mask': vis_mask,
            'lengths': lengths,
            'widths': widths,
            'scored_indices': scored_indices,
            'agent_ids': agent_ids,
            'av_index': av_index,
            'map_cache': map_cache,
            'rolling_predictions': rolling_predictions,
            'num_historical_steps': num_historical_steps,
            'total_timesteps': total_timesteps,
            'dataset_type': dataset_type,
        })

    return inference_results


def render_rolling_scenario_worker(args_tuple) -> str:
    """
    CPU-only worker that renders a single rolling scenario to a temp video file.
    No model or CUDA needed - just numpy arrays and OpenCV.

    Args:
        args_tuple: (inference_result, temp_dir, img_size, fps, prediction_horizon, rolling_gt, overlap_threshold)

    Returns:
        Path to the temporary video file
    """
    inference_result, temp_dir, img_size, fps, prediction_horizon, rolling_gt, overlap_threshold = args_tuple

    scenario_idx = inference_result['scenario_idx']
    scenario_id = inference_result['scenario_id']
    positions = inference_result['positions']
    headings = inference_result['headings']
    agent_types = inference_result['agent_types']
    valid_mask = inference_result['valid_mask']
    vis_mask = inference_result['vis_mask']
    lengths = inference_result.get('lengths')
    widths = inference_result.get('widths')
    scored_indices = inference_result['scored_indices']
    av_index = inference_result.get('av_index', 0)
    map_cache = inference_result['map_cache']
    rolling_predictions = inference_result['rolling_predictions']
    num_historical_steps = inference_result['num_historical_steps']
    total_timesteps = inference_result['total_timesteps']
    dataset_type = inference_result.get('dataset_type', 'venti3d')

    num_agents = positions.shape[0]

    # Compute ground truth for all scored agents
    gt_positions = {idx: positions[idx, num_historical_steps:] for idx in scored_indices}
    gt_valid = {idx: valid_mask[idx, num_historical_steps:] for idx in scored_indices}

    # Dynamic camera configuration
    MIN_VIEW_SIZE = 40.0   # Minimum view size (meters)
    MAX_VIEW_SIZE = 150.0  # Maximum view size (meters)
    PADDING = 20.0         # Padding around agents (meters)
    SMOOTH_ALPHA = 0.15    # Smoothing factor (0-1, higher = faster response)

    # Initialize smoothed camera values (will be set from first frame with valid agents)
    smoothed_center = None
    smoothed_view_size = None

    # Create temp video file
    # Use render_job_idx if available (per-agent mode), else scenario_idx
    job_idx = inference_result.get('render_job_idx', scenario_idx)
    focal_agent_id = inference_result.get('focal_agent_id')
    if focal_agent_id is not None:
        temp_path = os.path.join(temp_dir, f"temp_{scenario_idx:04d}_{focal_agent_id}.mp4")
    else:
        temp_path = os.path.join(temp_dir, f"temp_{scenario_idx:04d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_path, fourcc, fps, img_size)

    scenario_label = f"Rolling Horizon: {scenario_id}"

    for frame_idx in range(total_timesteps):

        # Compute dynamic camera center and view size from FOCAL agents only
        tracked_positions = []
        for scored_idx in scored_indices:
            if vis_mask[scored_idx, frame_idx]:
                tracked_positions.append(positions[scored_idx, frame_idx])

        if tracked_positions:
            tracked_positions = np.array(tracked_positions)

            # Target center: mean of valid tracked agents
            target_center = tracked_positions.mean(axis=0)

            # Target view size: bounding box of tracked agents + padding
            x_range = tracked_positions[:, 0].max() - tracked_positions[:, 0].min()
            y_range = tracked_positions[:, 1].max() - tracked_positions[:, 1].min()
            target_view_size = max(x_range, y_range) + 2 * PADDING
            target_view_size = np.clip(target_view_size, MIN_VIEW_SIZE, MAX_VIEW_SIZE)

            # Initialize or apply exponential smoothing
            if smoothed_center is None:
                smoothed_center = target_center
                smoothed_view_size = target_view_size
            else:
                smoothed_center = SMOOTH_ALPHA * target_center + (1 - SMOOTH_ALPHA) * smoothed_center
                smoothed_view_size = SMOOTH_ALPHA * target_view_size + (1 - SMOOTH_ALPHA) * smoothed_view_size

        # Use smoothed values for view bounds (fallback to defaults if no valid agents yet)
        if smoothed_center is not None:
            x_center, y_center = smoothed_center
            view_size = smoothed_view_size
        else:
            x_center, y_center = 0.0, 0.0
            view_size = MIN_VIEW_SIZE

        x_min = x_center - view_size / 2
        x_max = x_center + view_size / 2
        y_min = y_center - view_size / 2
        y_max = y_center + view_size / 2
        transform = CoordinateTransform((x_min, y_min, x_max, y_max), img_size)

        img = np.full((img_size[1], img_size[0], 3), COLORS['background'], dtype=np.uint8)

        # Draw map
        draw_map(img, transform, map_cache)

        # Draw all agents (use vis_mask for visibility, not vector_repr-modified valid_mask)
        for agent_idx in range(num_agents):
            if not vis_mask[agent_idx, frame_idx]:
                continue

            x, y = positions[agent_idx, frame_idx]
            heading = headings[agent_idx, frame_idx]
            agent_type = agent_types[agent_idx]
            is_focal = agent_idx in scored_indices

            # Get per-agent, per-frame dimensions if available
            agent_length = lengths[agent_idx, frame_idx] if lengths is not None else None
            agent_width = widths[agent_idx, frame_idx] if widths is not None else None

            draw_agent(img, transform, x, y, heading, agent_type, is_focal=is_focal,
                      length=agent_length, width=agent_width, dataset_type=dataset_type)

            # Draw history trail on top of agent for agents with predictions
            if is_focal and frame_idx > 0:
                draw_history_trail(img, transform, positions[agent_idx],
                                  valid_mask[agent_idx], frame_idx)

        # Draw ego marker on top of the ego vehicle
        if vis_mask[av_index, frame_idx]:
            ego_x, ego_y = positions[av_index, frame_idx]
            draw_ego_marker(img, transform, ego_x, ego_y)

        # Draw ground truth for all scored agents (only after history is built)
        if frame_idx >= num_historical_steps - 1:
            for scored_idx in scored_indices:
                if rolling_gt:
                    # Rolling GT: show only next prediction_horizon steps from current frame
                    gt_start = frame_idx - num_historical_steps + 1
                    gt_end = min(gt_start + prediction_horizon, len(gt_positions[scored_idx]))
                    rolling_gt_pos = gt_positions[scored_idx][gt_start:gt_end]
                    rolling_gt_valid = gt_valid[scored_idx][gt_start:gt_end]
                    draw_ground_truth(img, transform, rolling_gt_pos, rolling_gt_valid)
                else:
                    # Complete GT: show all remaining future positions
                    draw_ground_truth(img, transform, gt_positions[scored_idx], gt_valid[scored_idx])

        # Draw predictions for all scored agents if available
        pred_result = rolling_predictions[frame_idx]
        if pred_result is not None:
            for scored_idx in scored_indices:
                scales = pred_result['scales'].get(scored_idx) if pred_result.get('scales') else None
                draw_predictions(img, transform,
                               pred_result['predictions'][scored_idx],
                               pred_result['probs'][scored_idx],
                               prediction_horizon,
                               overlap_threshold=overlap_threshold,
                               scales=scales)

        # Draw scale bar
        draw_scale_bar(img, transform, img_size, bar_length_m=10.0)

        # Determine phase
        if frame_idx < num_historical_steps - 1:
            phase = "BUILDING HISTORY"
            phase_color = COLORS['building_history']
            time_label = f"t = {frame_idx - num_historical_steps + 1}"
        else:
            phase = "ROLLING PREDICTION"
            phase_color = COLORS['rolling_prediction']
            if frame_idx < num_historical_steps:
                time_label = f"t = {frame_idx - num_historical_steps + 1}"
            else:
                time_label = f"t = +{frame_idx - num_historical_steps + 1}"

        visible_agents = vis_mask[:, frame_idx].sum()

        # Debug: print what's being shown for first few frames
        if frame_idx < 5:
            print(f"    [Render] frame={frame_idx}, visible_agents={visible_agents}, vis_mask[:, {frame_idx}].sum()={vis_mask[:, frame_idx].sum()}")

        # Draw info bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img_size[0], 45), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        cv2.rectangle(img, (0, 0), (5, 45), phase_color, -1)

        info_text = f"{phase}  |  Frame {frame_idx + 1}/{total_timesteps}  |  {time_label}  |  Agents: {visible_agents}"
        cv2.putText(img, info_text,
                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1, cv2.LINE_AA)

        # Draw scenario label at bottom
        if scenario_label:
            label_y = img_size[1] - 15
            cv2.putText(img, scenario_label,
                       (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)

        # Draw legend (fixed height with uncertainty always shown)
        legend_x = 10
        legend_y = 55
        legend_width = 150
        item_spacing = 22
        legend_height = 139  # Fixed: History, Pred, Hi/Lo, Ground Truth, Ego, Uncertainty

        overlay = img.copy()
        cv2.rectangle(overlay, (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        item_y = legend_y + 18

        # History trail
        cv2.line(img, (legend_x + 5, item_y), (legend_x + 20, item_y),
                COLORS['history_trail'], 2, cv2.LINE_AA)
        cv2.putText(img, "History", (legend_x + 25, item_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)

        # Prediction with confidence gradient (hot-to-cold: red->blue)
        item_y += item_spacing
        prediction_colors = [
            (0, 0, 255),      # Red (high confidence)
            (0, 128, 255),    # Orange
            (0, 255, 255),    # Yellow
            (0, 255, 128),    # Yellow-green
            (255, 255, 0),    # Cyan
            (255, 128, 0),    # Blue (low confidence)
        ]
        gradient_width = 60
        segment_width = gradient_width // len(prediction_colors)
        for i, color in enumerate(prediction_colors):
            x1 = legend_x + 5 + i * segment_width
            x2 = x1 + segment_width
            cv2.rectangle(img, (x1, item_y - 4), (x2, item_y + 4), color, -1)
        cv2.putText(img, "Pred", (legend_x + 70, item_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)

        # Confidence labels
        item_y += item_spacing - 6
        cv2.putText(img, "Hi", (legend_x + 5, item_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "Lo", (legend_x + 48, item_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        # Ground truth
        item_y += item_spacing - 4
        cv2.drawMarker(img, (legend_x + 12, item_y), COLORS['ground_truth'],
                      cv2.MARKER_STAR, 10, 2)
        cv2.putText(img, "Ground Truth", (legend_x + 25, item_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)

        # Ego marker
        item_y += item_spacing
        cv2.circle(img, (legend_x + 12, item_y), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Ego", (legend_x + 25, item_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)

        # Uncertainty ellipse (always shown) - matches draw_confidence_ellipse colors
        item_y += item_spacing
        cv2.ellipse(img, (legend_x + 12, item_y), (8, 5), 0, 0, 360, (200, 100, 200), -1, cv2.LINE_AA)
        cv2.ellipse(img, (legend_x + 12, item_y), (8, 5), 0, 0, 360, (150, 50, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "Uncertainty", (legend_x + 25, item_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)

        video_writer.write(img)

    video_writer.release()
    return temp_path


def concatenate_videos_ffmpeg(temp_video_paths: List[str], output_path: str) -> bool:
    """
    Concatenate temporary videos into a single output using ffmpeg.
    Uses stream copy (no re-encoding) for speed.

    Returns True on success, False on failure.
    """
    if not temp_video_paths:
        return False

    # Create a file list for ffmpeg concat
    temp_dir = os.path.dirname(temp_video_paths[0])
    filelist_path = os.path.join(temp_dir, "filelist.txt")

    with open(filelist_path, 'w') as f:
        for path in sorted(temp_video_paths):
            f.write(f"file '{path}'\n")

    # Run ffmpeg to concatenate
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', filelist_path,
        '-c', 'copy',
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("  Error: ffmpeg not found. Please install ffmpeg.")
        return False


def process_rolling_scenarios_parallel(model, dataloader, device: torch.device,
                                        output_path: Path, num_scenarios: int,
                                        num_workers: int, img_size: Tuple[int, int],
                                        fps: int, prediction_horizon: int,
                                        rolling_gt: bool, keep_temp: bool,
                                        single_video: bool = True,
                                        overlap_threshold: float = 0.5,
                                        all_scored_agents: bool = True,
                                        dataset_type: str = 'venti3d') -> List[str]:
    """
    Process rolling scenarios in parallel using the two-phase pipeline:
    1. Batch rolling inference on GPU (sequential due to CUDA)
    2. Parallel rendering on CPU
    3. Optionally concatenate with ffmpeg

    Args:
        model: The QCNet model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        output_path: Path for the output video (or output directory if single_video=False)
        num_scenarios: Number of scenarios to process
        num_workers: Number of parallel rendering workers
        img_size: (width, height) of output frames
        fps: Frames per second
        prediction_horizon: How many future steps to show in predictions
        rolling_gt: If True, show ground truth as rolling trajectory
        keep_temp: Keep temporary files for debugging
        single_video: If True, concatenate all into one video; if False, output individual videos
        overlap_threshold: Distance threshold in meters for filtering overlapping predictions
        all_scored_agents: If True, show all scored agents in each video; if False, one video per focal agent

    Returns:
        List of output video paths
    """
    print(f"\n=== Phase 1: Running rolling inference on {num_scenarios} scenarios ===")
    inference_results = run_rolling_inference_batch(model, dataloader, device, num_scenarios, dataset_type)

    if not inference_results:
        print("No scenarios to process")
        return []

    # Expand inference results into render jobs (one per scored agent if not all_scored_agents)
    if all_scored_agents:
        render_jobs = inference_results
    else:
        render_jobs = []
        for result in inference_results:
            for i, scored_idx in enumerate(result['scored_indices']):
                job = result.copy()
                job['scored_indices'] = np.array([scored_idx])
                job['focal_agent_id'] = result['agent_ids'][scored_idx]
                # Create unique job index for temp file naming
                job['render_job_idx'] = len(render_jobs)
                render_jobs.append(job)

    print(f"\n=== Phase 2: Rendering {len(render_jobs)} videos with {num_workers} workers ===")

    # Create temp directory (or output directory if not combining)
    if single_video:
        temp_dir = tempfile.mkdtemp(prefix="qcnet_rolling_")
        print(f"  Temp directory: {temp_dir}")
    else:
        # Output directly to the output directory
        output_dir = output_path if output_path.is_dir() or output_path.suffix == '' else output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = str(output_dir)
        print(f"  Output directory: {temp_dir}")

    try:
        # Prepare arguments for workers
        worker_args = [
            (job, temp_dir, img_size, fps, prediction_horizon, rolling_gt, overlap_threshold)
            for job in render_jobs
        ]

        temp_video_paths = []
        # Track render_job_idx -> temp_path mapping for non-single-video mode
        job_to_path = {}
        completed_count = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(render_rolling_scenario_worker, args): i
                      for i, args in enumerate(worker_args)}

            for future in as_completed(futures):
                job_idx = futures[future]
                try:
                    temp_path = future.result()
                    temp_video_paths.append(temp_path)
                    job_to_path[job_idx] = temp_path
                    completed_count += 1
                    print(f"  Rendered video {completed_count}/{len(render_jobs)}")
                except Exception as e:
                    print(f"  Error rendering video {job_idx}: {e}")

        if single_video:
            print(f"\n=== Phase 3: Concatenating {len(temp_video_paths)} videos ===")

            if concatenate_videos_ffmpeg(sorted(temp_video_paths), str(output_path)):
                print(f"  Combined video saved to: {output_path}")
                return [str(output_path)]
            else:
                print("  Failed to concatenate videos with ffmpeg")
                # Fallback: keep individual videos
                print("  Keeping individual temp videos as fallback")
                keep_temp = True
                return temp_video_paths
        else:
            # Rename temp files to proper names based on scenario_id and focal_agent_id
            output_paths = []
            for i, job in enumerate(render_jobs):
                if i not in job_to_path:
                    continue  # Skip if rendering failed
                scenario_id = job['scenario_id']
                focal_agent_id = job.get('focal_agent_id')
                temp_path = job_to_path[i]
                if focal_agent_id is not None:
                    final_path = os.path.join(temp_dir, f"{scenario_id}_agent_{focal_agent_id}_rolling.mp4")
                else:
                    final_path = os.path.join(temp_dir, f"{scenario_id}_rolling.mp4")
                if temp_path != final_path:
                    shutil.move(temp_path, final_path)
                output_paths.append(final_path)
            print(f"\n{len(output_paths)} videos saved to: {temp_dir}")
            return output_paths

    finally:
        if single_video and not keep_temp:
            print(f"  Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif single_video and keep_temp:
            print(f"  Keeping temp directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate rolling horizon visualization for QCNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # By scenario index (1-based)
  python visualize_scenario.py \\
      --root /path/to/dataset \\
      --ckpt_path /path/to/checkpoint.ckpt \\
      --scenario_index 1 \\
      --dataset venti3d --split test

  # By scenario ID
  python visualize_scenario.py \\
      --root /path/to/dataset \\
      --ckpt_path /path/to/checkpoint.ckpt \\
      --scenario_id <scenario_id> \\
      --dataset venti3d --split test

  # All scenarios (outputs to ./videos/ by default)
  python visualize_scenario.py \\
      --root /path/to/dataset \\
      --ckpt_path /path/to/checkpoint.ckpt \\
      --dataset venti3d --split test

  # Parallel processing for multiple scenarios
  python visualize_scenario.py \\
      --root /path/to/dataset \\
      --ckpt_path /path/to/checkpoint.ckpt \\
      --dataset venti3d --split test \\
      --num-workers 4 --single-video
        """
    )
    parser.add_argument('--root', type=str, required=True,
                       help='Path to dataset root')
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='Path to model checkpoint')

    # Mutually exclusive scenario selection (optional - if neither provided, render all)
    scenario_group = parser.add_mutually_exclusive_group(required=False)
    scenario_group.add_argument('--scenario_id', type=str,
                               help='Scenario ID to visualize')
    scenario_group.add_argument('--scenario_index', type=int,
                               help='Scenario number to visualize (1-based)')

    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: videos/{scenario_id}_rolling.mp4)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for output video')
    parser.add_argument('--width', type=int, default=1920,
                       help='Video width in pixels')
    parser.add_argument('--height', type=int, default=1080,
                       help='Video height in pixels')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--dataset', type=str, default='argoverse_v2',
                       choices=['argoverse_v2', 'venti3d'],
                       help='Dataset type to use')
    parser.add_argument('--prediction_horizon', type=int, default=60,
                       help='How many future steps to show in predictions (60 = 6s at 10Hz)')
    parser.add_argument('--rolling-gt', action='store_true',
                       help='Show ground truth as rolling trajectory (like predictions) '
                            'instead of complete future trajectory')
    parser.add_argument('--overlap-threshold', type=float, default=0.5,
                       help='Distance threshold in meters for filtering overlapping predictions (default: 0.5)')
    parser.add_argument('--all-scored-agents', action='store_true',
                       help='Show all scored agents in a single video (default: one video per focal agent)')

    # Parallel processing arguments
    parser.add_argument('--max-scenarios', type=int, default=None,
                       help='Maximum number of scenarios to render (only used when no specific scenario is specified)')
    parser.add_argument('--num-workers', type=int, default=max(1, os.cpu_count() // 2),
                       help='Number of parallel workers for rendering (default: half of CPU cores)')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary files for debugging')
    parser.add_argument('--single-video', action='store_true',
                       help='Combine all scenarios into a single video file (uses ffmpeg)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {args.ckpt_path}...")
    model = QCNet.load_from_checkpoint(args.ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()

    print(f"Loading {args.split} split from {args.root} ({args.dataset})...")
    # Note: We don't use TargetBuilder here because visualization doesn't need
    # the 'target' field, and TargetBuilder assumes fixed future length matching
    # the model. This allows visualizing scenarios with arbitrary lengths.
    if args.dataset == 'argoverse_v2':
        dataset = ArgoverseV2Dataset(
            root=args.root,
            split=args.split,
            transform=None
        )
    elif args.dataset == 'venti3d':
        dataset = Venti3DDataset(
            root=args.root,
            split=args.split,
            transform=None
        )

    print(f"Dataset size: {len(dataset)} scenarios")

    img_size = (args.width, args.height)
    num_workers = args.num_workers

    # Determine which scenarios to process
    if args.scenario_id is not None:
        # Single scenario by ID
        print(f"Searching for scenario ID: {args.scenario_id}...")
        scenario_idx = find_scenario_by_id(dataset, args.scenario_id)
        if scenario_idx is None:
            print(f"Error: Scenario ID '{args.scenario_id}' not found in dataset")
            return
        print(f"Found at index {scenario_idx + 1}")
        scenario_indices = [scenario_idx]
    elif args.scenario_index is not None:
        # Single scenario by index
        scenario_idx = args.scenario_index - 1  # Convert 1-based input to 0-based
        if scenario_idx < 0 or scenario_idx >= len(dataset):
            print(f"Error: Scenario index {args.scenario_index} out of range (1-{len(dataset)})")
            return
        scenario_indices = [scenario_idx]
    else:
        # No scenario specified - render all (or up to max-scenarios)
        if args.max_scenarios is not None:
            num_to_render = min(args.max_scenarios, len(dataset))
            print(f"No scenario specified, rendering first {num_to_render} of {len(dataset)} scenarios...")
            scenario_indices = list(range(num_to_render))
        else:
            print(f"No scenario specified, rendering all {len(dataset)} scenarios...")
            scenario_indices = list(range(len(dataset)))

    total_scenarios = len(scenario_indices)

    # Determine if output should be single video or individual videos
    # Single video when: --single-video flag OR (only 1 scenario AND showing all scored agents together)
    # When per-agent mode (not all_scored_agents), always output individual videos unless --single-video
    single_video = args.single_video or (total_scenarios == 1 and args.all_scored_agents)

    # Create a subset dataloader if processing specific scenarios
    from torch.utils.data import Subset
    if scenario_indices != list(range(len(dataset))):
        subset = Subset(dataset, scenario_indices)
        dataloader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    # Set up output path based on single_video flag
    if single_video:
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path('./videos')
            output_dir.mkdir(parents=True, exist_ok=True)
            if total_scenarios == 1:
                # For single scenario, get the scenario_id for the filename
                data = dataset[scenario_indices[0]]
                if hasattr(data, 'scenario_id'):
                    scenario_id = data.scenario_id[0] if isinstance(data.scenario_id, list) else data.scenario_id
                else:
                    scenario_id = f"scenario_{scenario_indices[0]:04d}"
                output_path = output_dir / f"{scenario_id}_rolling.mp4"
            else:
                output_path = output_dir / "combined_rolling.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Output directory for individual videos
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path('./videos')
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {total_scenarios} scenario(s) with {num_workers} workers")
    output_paths = process_rolling_scenarios_parallel(
        model, dataloader, device, output_path,
        num_scenarios=total_scenarios,
        num_workers=num_workers,
        img_size=img_size,
        fps=args.fps,
        prediction_horizon=args.prediction_horizon,
        rolling_gt=args.rolling_gt,
        keep_temp=args.keep_temp,
        single_video=single_video,
        overlap_threshold=args.overlap_threshold,
        all_scored_agents=args.all_scored_agents,
        dataset_type=args.dataset
    )
    if single_video:
        print(f"\nDone! Video saved to {output_path}")
    else:
        print(f"\nDone! {len(output_paths)} videos saved to {output_path}")


if __name__ == '__main__':
    main()
