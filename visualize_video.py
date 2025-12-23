#!/usr/bin/env python3
"""
Video visualization for QCNet trajectory prediction using OpenCV.

Generates animated videos showing:
- Lane graph (map polygons and boundary points)
- All agents moving through time
- Trajectory predictions for focal agents with mode probabilities
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch_geometric.loader import DataLoader

from predictors import QCNet
from datasets import ArgoverseV2Dataset
from transforms import TargetBuilder


# Agent type names for reference
AGENT_TYPES = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus',
               'static', 'background', 'construction', 'riderless_bicycle', 'unknown']

# Colors in BGR format for OpenCV
COLORS = {
    'focal_agent': (60, 76, 231),       # Red
    'other_vehicle': (219, 152, 52),    # Blue
    'pedestrian': (156, 188, 26),       # Cyan/Teal
    'cyclist': (18, 156, 243),          # Orange
    'lane_boundary': (141, 140, 127),   # Gray
    'centerline': (166, 165, 149),      # Light gray
    'crosswalk': (39, 196, 241),        # Yellow
    'prediction': (34, 126, 230),       # Orange
    'ground_truth': (96, 174, 39),      # Green
    'history_trail': (219, 152, 52),    # Blue
    'background': (80, 62, 44),         # Dark blue-gray
    'text': (255, 255, 255),            # White
}


class CoordinateTransform:
    """Transforms world coordinates to image coordinates."""

    def __init__(self, bounds: Tuple[float, float, float, float],
                 img_size: Tuple[int, int], margin: int = 50):
        """
        Args:
            bounds: (x_min, y_min, x_max, y_max) in world coordinates
            img_size: (width, height) of output image
            margin: Pixel margin around the scene
        """
        self.x_min, self.y_min, self.x_max, self.y_max = bounds
        self.img_width, self.img_height = img_size
        self.margin = margin

        # Calculate scale to fit scene in image with margin
        scene_width = self.x_max - self.x_min
        scene_height = self.y_max - self.y_min

        available_width = self.img_width - 2 * margin
        available_height = self.img_height - 2 * margin

        self.scale = min(available_width / scene_width,
                        available_height / scene_height)

        # Center offset
        self.offset_x = margin + (available_width - scene_width * self.scale) / 2
        self.offset_y = margin + (available_height - scene_height * self.scale) / 2

    def world_to_image(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to image pixel coordinates."""
        img_x = int((x - self.x_min) * self.scale + self.offset_x)
        img_y = int((self.y_max - y) * self.scale + self.offset_y)  # Flip y-axis
        return img_x, img_y

    def world_to_image_array(self, points: np.ndarray) -> np.ndarray:
        """Convert array of world coordinates to image coordinates."""
        img_x = ((points[:, 0] - self.x_min) * self.scale + self.offset_x).astype(np.int32)
        img_y = ((self.y_max - points[:, 1]) * self.scale + self.offset_y).astype(np.int32)
        return np.stack([img_x, img_y], axis=1)

    def scale_length(self, length: float) -> int:
        """Convert a world-space length to pixel length."""
        return max(1, int(length * self.scale))


def draw_agent(img: np.ndarray, transform: CoordinateTransform,
               x: float, y: float, heading: float, agent_type: int,
               is_focal: bool = False, alpha: float = 1.0) -> None:
    """
    Draw an agent at the given position with heading.
    """
    # Get image coordinates
    cx, cy = transform.world_to_image(x, y)

    if agent_type in [0, 4]:  # vehicle or bus
        # Vehicle dimensions
        if agent_type == 4:  # bus
            length = transform.scale_length(12.0)
            width = transform.scale_length(2.5)
        else:  # regular vehicle
            length = transform.scale_length(4.5)
            width = transform.scale_length(2.0)

        color = COLORS['focal_agent'] if is_focal else COLORS['other_vehicle']

        # Create rotated rectangle
        # Note: heading is in radians, cv2.boxPoints expects degrees
        angle_deg = -np.degrees(heading)  # Negative because y-axis is flipped
        rect = ((cx, cy), (length, width), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        if is_focal:
            cv2.fillPoly(img, [box], color)
            cv2.polylines(img, [box], True, (255, 255, 255), 2)
        else:
            cv2.polylines(img, [box], True, color, 2)

        # Draw heading indicator
        front_x = int(cx + (length/2) * np.cos(-heading))
        front_y = int(cy + (length/2) * np.sin(-heading))
        indicator_color = (255, 255, 255) if is_focal else color
        cv2.line(img, (cx, cy), (front_x, front_y), indicator_color, 2)

    elif agent_type == 1:  # pedestrian
        color = COLORS['focal_agent'] if is_focal else COLORS['pedestrian']
        radius = transform.scale_length(0.5)
        radius = max(3, radius)

        if is_focal:
            cv2.circle(img, (cx, cy), radius, color, -1)
            cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
        else:
            cv2.circle(img, (cx, cy), radius, color, 2)

    elif agent_type in [2, 3, 8]:  # motorcyclist, cyclist, riderless_bicycle
        color = COLORS['focal_agent'] if is_focal else COLORS['cyclist']
        length = transform.scale_length(2.0)
        width = transform.scale_length(0.8)

        angle_deg = -np.degrees(heading)
        rect = ((cx, cy), (length, width), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        if is_focal:
            cv2.fillPoly(img, [box], color)
        else:
            cv2.polylines(img, [box], True, color, 2)

    else:  # other types
        color = COLORS['focal_agent'] if is_focal else (149, 165, 166)
        cv2.circle(img, (cx, cy), 4, color, -1)


def draw_map(img: np.ndarray, transform: CoordinateTransform,
             map_cache: Dict) -> None:
    """
    Draw lane graph from cached map data.
    """
    if 'point_pos' not in map_cache:
        return

    point_pos = map_cache['point_pos']
    point_type = map_cache.get('point_type', None)

    # Convert all points to image coordinates
    img_points = transform.world_to_image_array(point_pos)

    # Draw points
    if point_type is not None:
        # Crosswalk points (type 15)
        crosswalk_mask = point_type == 15
        for i in np.where(crosswalk_mask)[0]:
            cv2.circle(img, tuple(img_points[i]), 2, COLORS['crosswalk'], -1)

        # Regular lane points
        lane_mask = ~crosswalk_mask
        for i in np.where(lane_mask)[0]:
            cv2.circle(img, tuple(img_points[i]), 1, COLORS['lane_boundary'], -1)
    else:
        for i in range(len(img_points)):
            cv2.circle(img, tuple(img_points[i]), 1, COLORS['lane_boundary'], -1)


def draw_predictions(img: np.ndarray, transform: CoordinateTransform,
                    predictions: np.ndarray, probs: np.ndarray,
                    future_idx: int, top_k: int = 6) -> None:
    """
    Draw trajectory predictions with probability-based styling.
    Predictions should already be in world coordinates.
    """
    sorted_idx = np.argsort(probs)[::-1][:top_k]

    for rank, idx in enumerate(sorted_idx):
        traj = predictions[idx]  # Already in world coordinates
        prob = probs[idx]

        # Only draw up to current future timestep + lookahead
        show_steps = min(future_idx + 20, len(traj))
        if show_steps <= 1:
            continue

        traj_show = traj[:show_steps]

        # Convert to image coordinates
        img_points = transform.world_to_image_array(traj_show)

        # Style based on rank
        thickness = max(1, 3 - rank)
        alpha = max(0.3, 0.9 - rank * 0.1)

        # Blend color with probability
        base_color = np.array(COLORS['prediction'], dtype=np.float32)
        color = tuple(int(c * (0.5 + 0.5 * prob)) for c in base_color)

        # Draw trajectory line
        cv2.polylines(img, [img_points], False, color, thickness, cv2.LINE_AA)

        # Draw endpoint for top predictions
        if rank < 3:
            end_point = tuple(img_points[-1])
            cv2.circle(img, end_point, 6 - rank, color, -1)
            cv2.circle(img, end_point, 6 - rank, (255, 255, 255), 1)

            # Draw probability label
            label = f'{prob:.0%}'
            cv2.putText(img, label, (end_point[0] + 8, end_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_ground_truth(img: np.ndarray, transform: CoordinateTransform,
                     gt_traj: np.ndarray, future_idx: int) -> None:
    """
    Draw ground truth trajectory up to current timestep.
    Ground truth should already be in world coordinates.
    """
    if future_idx < 0 or gt_traj is None:
        return

    show_steps = min(future_idx + 1, len(gt_traj))
    if show_steps <= 1:
        return

    # Ground truth is already in absolute coordinates
    traj_show = gt_traj[:show_steps]

    # Convert to image coordinates
    img_points = transform.world_to_image_array(traj_show)

    # Draw dashed line effect
    color = COLORS['ground_truth']
    for i in range(0, len(img_points) - 1, 2):
        end_idx = min(i + 2, len(img_points))
        cv2.polylines(img, [img_points[i:end_idx]], False, color, 2, cv2.LINE_AA)

    # Draw endpoint star
    end_point = tuple(img_points[-1])
    cv2.drawMarker(img, end_point, color, cv2.MARKER_STAR, 15, 2)


def draw_history_trail(img: np.ndarray, transform: CoordinateTransform,
                      positions: np.ndarray, valid_mask: np.ndarray,
                      frame_idx: int, trail_length: int = 20) -> None:
    """
    Draw fading history trail for an agent.
    """
    start_idx = max(0, frame_idx - trail_length)
    end_idx = frame_idx + 1

    if end_idx <= start_idx:
        return

    # Get valid positions in trail range
    trail_mask = valid_mask[start_idx:end_idx]
    trail = positions[start_idx:end_idx]

    if not trail_mask.any():
        return

    # Convert to image coordinates
    img_points = transform.world_to_image_array(trail)

    # Draw with fading
    color = COLORS['history_trail']
    for i in range(len(trail) - 1):
        if not (trail_mask[i] and trail_mask[i + 1]):
            continue

        # Fade alpha based on position in trail
        alpha = 0.3 + 0.7 * (i / max(1, len(trail) - 1))
        faded_color = tuple(int(c * alpha) for c in color)

        cv2.line(img, tuple(img_points[i]), tuple(img_points[i + 1]),
                faded_color, 2, cv2.LINE_AA)


def generate_video(model, data, device, output_path: Path,
                  fps: int = 10, img_size: Tuple[int, int] = (1280, 720),
                  has_ground_truth: bool = True,
                  all_agents_focal: bool = False) -> None:
    """
    Generate video visualization for a scenario using OpenCV.
    """
    model.eval()
    data = data.to(device)

    num_historical_steps = model.num_historical_steps
    num_future_steps = model.num_future_steps
    total_frames = num_historical_steps + num_future_steps

    # Run inference
    with torch.no_grad():
        output = model(data)

    # Extract data to CPU
    positions = data['agent']['position'][:, :, :2].cpu().numpy()
    headings = data['agent']['heading'].cpu().numpy()
    agent_types = data['agent']['type'].cpu().numpy()
    valid_mask = data['agent']['valid_mask'].cpu().numpy()

    predictions_local = output['loc_refine_pos'].cpu().numpy()  # In agent-local frame
    probs = F.softmax(output['pi'], dim=-1).cpu().numpy()

    # Find focal agents
    if all_agents_focal:
        # All agents are focal - show predictions for everyone
        focal_indices = np.arange(positions.shape[0])
    else:
        # Only scored/evaluated agents
        eval_mask = data['agent']['predict_mask'][:, num_historical_steps - 1].cpu().numpy()
        focal_indices = np.where(eval_mask)[0]

        if len(focal_indices) == 0:
            print("  Warning: No focal agents found, using first agent")
            focal_indices = np.array([0])

    # Current positions and headings at prediction time (last historical step)
    current_positions = positions[:, num_historical_steps - 1]
    current_headings = headings[:, num_historical_steps - 1]

    # Transform predictions from agent-local to world coordinates
    # predictions_local is in agent frame, need to rotate by heading and add position
    predictions = np.zeros_like(predictions_local)
    for agent_idx in range(len(predictions_local)):
        theta = current_headings[agent_idx]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotation matrix: agent-local to world
        rot_mat = np.array([[cos_t, -sin_t],
                           [sin_t, cos_t]])
        # Transform each mode and timestep
        pred_local = predictions_local[agent_idx]  # [num_modes, num_steps, 2]
        for mode_idx in range(pred_local.shape[0]):
            # Rotate and translate
            rotated = pred_local[mode_idx] @ rot_mat.T
            predictions[agent_idx, mode_idx] = rotated + current_positions[agent_idx]

    # Get ground truth for focal agents
    focal_gt = {}
    if has_ground_truth:
        for focal_idx in focal_indices:
            focal_gt[focal_idx] = positions[focal_idx, num_historical_steps:]

    # Cache map data
    map_cache = {}
    if 'map_point' in data.node_types:
        map_cache['point_pos'] = data['map_point']['position'].cpu().numpy()[:, :2]
        if 'type' in data['map_point']:
            map_cache['point_type'] = data['map_point']['type'].cpu().numpy()

    # Calculate scene bounds - center on focal agent with 200m x 200m view
    focal_idx = focal_indices[0]

    # Get focal agent's center position (middle of trajectory)
    focal_positions = positions[focal_idx]
    focal_valid = valid_mask[focal_idx]
    focal_valid_pos = focal_positions[focal_valid]

    if len(focal_valid_pos) > 0:
        # Center on the middle of the focal agent's trajectory
        x_center = focal_valid_pos[:, 0].mean()
        y_center = focal_valid_pos[:, 1].mean()
    else:
        x_center, y_center = 0, 0

    # Fixed 80m x 80m view
    view_size = 80.0
    x_min = x_center - view_size / 2
    x_max = x_center + view_size / 2
    y_min = y_center - view_size / 2
    y_max = y_center + view_size / 2

    # Create coordinate transform
    transform = CoordinateTransform((x_min, y_min, x_max, y_max), img_size)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, img_size)

    num_agents = positions.shape[0]

    for frame_idx in range(total_frames):
        # Create blank frame
        img = np.full((img_size[1], img_size[0], 3), COLORS['background'], dtype=np.uint8)

        # Draw map
        draw_map(img, transform, map_cache)

        # Draw all agents
        for agent_idx in range(num_agents):
            if not valid_mask[agent_idx, frame_idx]:
                continue

            x, y = positions[agent_idx, frame_idx]
            heading = headings[agent_idx, frame_idx]
            agent_type = agent_types[agent_idx]
            is_focal = agent_idx in focal_indices

            # Draw history trail for focal agents
            if is_focal and frame_idx > 0:
                draw_history_trail(img, transform, positions[agent_idx],
                                  valid_mask[agent_idx], frame_idx)

            # Draw agent
            draw_agent(img, transform, x, y, heading, agent_type, is_focal=is_focal)

        # Draw predictions and ground truth after historical period
        if frame_idx >= num_historical_steps:
            future_idx = frame_idx - num_historical_steps

            for focal_idx in focal_indices:
                # Draw predictions (already in world coordinates)
                draw_predictions(img, transform, predictions[focal_idx],
                               probs[focal_idx], future_idx)

                # Draw ground truth
                if focal_idx in focal_gt:
                    draw_ground_truth(img, transform, focal_gt[focal_idx],
                                     future_idx)

        # Add frame info overlay
        if frame_idx < num_historical_steps:
            time_label = f"History: t = {frame_idx - num_historical_steps + 1}"
            phase = "HISTORY"
            phase_color = COLORS['history_trail']
        else:
            time_label = f"Future: t = +{frame_idx - num_historical_steps + 1}"
            phase = "PREDICTION"
            phase_color = COLORS['prediction']

        # Draw info bar at top
        cv2.rectangle(img, (0, 0), (img_size[0], 40), (30, 30, 30), -1)
        cv2.putText(img, f"{phase} | Frame {frame_idx + 1}/{total_frames} | {time_label}",
                   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2, cv2.LINE_AA)

        # Draw legend
        legend_y = 60
        cv2.circle(img, (20, legend_y), 6, COLORS['focal_agent'], -1)
        cv2.putText(img, "Focal Agent", (35, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)

        cv2.circle(img, (20, legend_y + 25), 6, COLORS['prediction'], -1)
        cv2.putText(img, "Prediction", (35, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)

        if has_ground_truth:
            cv2.circle(img, (20, legend_y + 50), 6, COLORS['ground_truth'], -1)
            cv2.putText(img, "Ground Truth", (35, legend_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)

        # Write frame
        video_writer.write(img)

    video_writer.release()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate video visualizations for QCNet')
    parser.add_argument('--root', type=str, required=True,
                       help='Path to Argoverse 2 dataset root')
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./videos',
                       help='Directory to save videos')
    parser.add_argument('--num_videos', type=int, default=5,
                       help='Number of videos to generate')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for output video')
    parser.add_argument('--width', type=int, default=1280,
                       help='Video width in pixels')
    parser.add_argument('--height', type=int, default=720,
                       help='Video height in pixels')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--all_agents_focal', action='store_true',
                       help='Show predictions for all agents, not just scored agents')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model = QCNet.load_from_checkpoint(args.ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Load dataset
    print(f"Loading {args.split} split from {args.root}...")
    dataset = ArgoverseV2Dataset(
        root=args.root,
        split=args.split,
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    has_ground_truth = args.split in ['train', 'val']
    print(f"Dataset size: {len(dataset)} scenarios")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.width, args.height)

    # Generate videos
    print(f"\nGenerating {args.num_videos} videos...")
    data_iter = iter(dataloader)

    for i in range(args.num_videos):
        try:
            data = next(data_iter)
        except StopIteration:
            print(f"  Only {i} scenarios available in {args.split} split")
            break

        # Get scenario ID if available
        if hasattr(data, 'scenario_id'):
            scenario_id = data.scenario_id[0] if isinstance(data.scenario_id, list) else data.scenario_id
        else:
            scenario_id = f"scenario_{i:04d}"

        print(f"\nScenario {i+1}/{args.num_videos}: {scenario_id}")

        output_path = output_dir / f"{scenario_id}.mp4"
        generate_video(model, data, device, output_path,
                      fps=args.fps, img_size=img_size,
                      has_ground_truth=has_ground_truth,
                      all_agents_focal=args.all_agents_focal)

    print(f"\nDone! Videos saved to {output_dir}")


if __name__ == '__main__':
    main()
