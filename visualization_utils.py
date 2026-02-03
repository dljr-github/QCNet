#!/usr/bin/env python3
"""
Shared visualization utilities for QCNet trajectory prediction.

Contains:
- Constants (AGENT_TYPES, COLORS)
- CoordinateTransform class for world-to-image conversion
- Drawing functions for agents, maps, predictions, and ground truth
"""

import numpy as np
import cv2
from functools import lru_cache
from typing import Dict, Optional, Tuple

from utils.config import load_dataset_config


# Agent type names for reference (consolidated Venti3D types)
# Index: 0=vehicle, 1=trailer, 2=industrial_vehicle, 3=pedestrian, 4=unknown
AGENT_TYPES = ['vehicle', 'trailer', 'industrial_vehicle', 'pedestrian', 'unknown']

# Default agent rendering config (fallback if not in YAML)
# Format: {type_index: (shape, default_length_m, default_width_m)}
# shape: 'box', 'small_box', 'circle', 'tiny_circle'
_DEFAULT_AGENT_RENDER_CONFIG = {
    0: ('box', 4.5, 2.0),           # vehicle
    1: ('box', 12.0, 2.5),          # trailer
    2: ('small_box', 2.0, 0.8),     # industrial_vehicle
    3: ('circle', None, 1.0),       # pedestrian
    4: ('tiny_circle', None, None), # static
    5: ('tiny_circle', None, None), # unknown
}


@lru_cache(maxsize=8)
def get_agent_render_config(dataset_type: str) -> Dict[int, Tuple[str, Optional[float], Optional[float]]]:
    """Get agent rendering config for a dataset.

    Loads from YAML config file, falls back to defaults if not present.

    Args:
        dataset_type: Dataset type ('venti3d', 'argoverse_v2', etc.)

    Returns:
        Dict mapping agent type index to (shape, length, width) tuple.
    """
    try:
        config = load_dataset_config(dataset_type)
        yaml_config = config.get('agent_render_config', {})

        if not yaml_config:
            return _DEFAULT_AGENT_RENDER_CONFIG.copy()

        # Convert YAML format to tuple format
        result = {}
        for type_idx, settings in yaml_config.items():
            shape = settings.get('shape', 'tiny_circle')
            length = settings.get('length')
            width = settings.get('width')
            result[int(type_idx)] = (shape, length, width)

        return result

    except FileNotFoundError:
        return _DEFAULT_AGENT_RENDER_CONFIG.copy()

# Colors in BGR format for OpenCV (av2-api inspired professional scheme)
COLORS = {
    'focal_agent': (91, 165, 236),      # #ECA25B Orange
    'other_vehicle': (239, 232, 211),   # #D3E8EF Light blue
    'pedestrian': (239, 232, 211),      # Light blue (same)
    'cyclist': (239, 232, 211),         # Light blue
    'lane_segment': (224, 224, 224),    # #E0E0E0 Light gray fill
    'lane_boundary': (180, 180, 180),   # Gray for boundaries
    'centerline': (200, 200, 200),      # Lighter gray dashed
    'crosswalk': (180, 200, 220),       # Light beige
    'prediction': (91, 165, 236),       # Orange (match focal)
    'ground_truth': (96, 174, 39),      # Green
    'history_trail': (255, 0, 255),     # Magenta - distinct from predictions
    'background': (50, 50, 50),         # Dark gray
    'text': (255, 255, 255),            # White
    'building_history': (128, 128, 128),  # Gray for building history phase
    'rolling_prediction': (0, 165, 255),  # Orange for rolling prediction phase
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


def draw_confidence_ellipse(img: np.ndarray, transform: CoordinateTransform,
                             center_x: float, center_y: float,
                             scale_x: float, scale_y: float,
                             heading: float,
                             alpha: float = 0.3, scale_factor: float = 2.0) -> None:
    """
    Draw a confidence ellipse at the given position.

    Args:
        img: Image to draw on
        transform: Coordinate transform
        center_x, center_y: World position of ellipse center
        scale_x, scale_y: Laplace scale parameters (uncertainty) in x and y
        heading: Rotation angle in radians
        alpha: Transparency (0-1)
        scale_factor: Multiplier for scale values to determine ellipse size
    """
    cx, cy = transform.world_to_image(center_x, center_y)

    # Convert Laplace scale to ellipse axes (scale * factor gives reasonable visual size)
    # The Laplace distribution has 95% of mass within ~3*scale
    axis_x = transform.scale_length(scale_x * scale_factor)
    axis_y = transform.scale_length(scale_y * scale_factor)

    # Ensure minimum visible size
    axis_x = max(2, axis_x)
    axis_y = max(2, axis_y)

    # Convert heading to degrees (negative because y-axis is flipped)
    angle_deg = -np.degrees(heading)

    # Fixed uncertainty colors (distinct from all other visualization colors)
    fill_color = (200, 100, 200)    # Light purple (BGR)
    outline_color = (150, 50, 150)  # Darker purple (BGR)

    # Draw filled ellipse with transparency using overlay
    overlay = img.copy()
    cv2.ellipse(overlay, (cx, cy), (axis_x, axis_y), angle_deg, 0, 360, fill_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw solid outline for definition
    cv2.ellipse(img, (cx, cy), (axis_x, axis_y), angle_deg, 0, 360, outline_color, 1, cv2.LINE_AA)


def draw_arrow_head(img: np.ndarray, tip_x: int, tip_y: int,
                    heading: float, size: int, color: tuple) -> None:
    """Draw an arrow head at the given position pointing in the heading direction."""
    wing_angle = np.pi / 6
    wing_length = size

    left_angle = heading + np.pi + wing_angle
    left_x = int(tip_x + wing_length * np.cos(-left_angle))
    left_y = int(tip_y + wing_length * np.sin(-left_angle))

    right_angle = heading + np.pi - wing_angle
    right_x = int(tip_x + wing_length * np.cos(-right_angle))
    right_y = int(tip_y + wing_length * np.sin(-right_angle))

    pts = np.array([[tip_x, tip_y], [left_x, left_y], [right_x, right_y]], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def _draw_box_agent(img: np.ndarray, transform: CoordinateTransform,
                    cx: int, cy: int, heading: float,
                    length_m: float, width_m: float,
                    color: Tuple[int, int, int], is_focal: bool) -> None:
    """Draw a box-shaped agent (vehicles, buses, trailers)."""
    length_px = transform.scale_length(length_m)
    width_px = transform.scale_length(width_m)

    angle_deg = -np.degrees(heading)  # Negative because y-axis is flipped
    rect = ((cx, cy), (length_px, width_px), angle_deg)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.fillPoly(img, [box], color)
    outline_color = (255, 255, 255) if is_focal else tuple(int(c * 0.7) for c in color)
    cv2.polylines(img, [box], True, outline_color, 2, cv2.LINE_AA)

    front_x = int(cx + (length_px/2 - 3) * np.cos(-heading))
    front_y = int(cy + (length_px/2 - 3) * np.sin(-heading))
    arrow_color = (255, 255, 255) if is_focal else outline_color
    arrow_size = max(6, int(length_px / 4))
    draw_arrow_head(img, front_x, front_y, heading, arrow_size, arrow_color)


def _draw_small_box_agent(img: np.ndarray, transform: CoordinateTransform,
                          cx: int, cy: int, heading: float,
                          length_m: float, width_m: float,
                          color: Tuple[int, int, int], is_focal: bool) -> None:
    """Draw a small box-shaped agent (cyclists, motorcyclists, industrial equipment)."""
    length_px = transform.scale_length(length_m)
    width_px = transform.scale_length(width_m)

    angle_deg = -np.degrees(heading)
    rect = ((cx, cy), (length_px, width_px), angle_deg)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.fillPoly(img, [box], color)
    outline_color = (255, 255, 255) if is_focal else tuple(int(c * 0.7) for c in color)
    cv2.polylines(img, [box], True, outline_color, 1, cv2.LINE_AA)

    front_x = int(cx + (length_px/2 - 2) * np.cos(-heading))
    front_y = int(cy + (length_px/2 - 2) * np.sin(-heading))
    draw_arrow_head(img, front_x, front_y, heading, 4, outline_color)


def _draw_circle_agent(img: np.ndarray, transform: CoordinateTransform,
                       cx: int, cy: int, radius_m: float,
                       color: Tuple[int, int, int], is_focal: bool) -> None:
    """Draw a circle-shaped agent (pedestrians)."""
    radius = transform.scale_length(radius_m)
    radius = max(3, radius)

    cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    outline_color = (255, 255, 255) if is_focal else tuple(int(c * 0.7) for c in color)
    cv2.circle(img, (cx, cy), radius, outline_color, 1, cv2.LINE_AA)


def _draw_tiny_circle_agent(img: np.ndarray, cx: int, cy: int,
                            color: Tuple[int, int, int], is_focal: bool) -> None:
    """Draw a tiny circle agent (unknown/static types)."""
    cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, (100, 100, 100), 1, cv2.LINE_AA)


def draw_agent(img: np.ndarray, transform: CoordinateTransform,
               x: float, y: float, heading: float, agent_type: int,
               is_focal: bool = False, alpha: float = 1.0,
               length: Optional[float] = None, width: Optional[float] = None,
               dataset_type: Optional[str] = None) -> None:
    """
    Draw an agent at the given position with heading.
    Uses anti-aliased drawing for cleaner visuals.

    Args:
        img: Image to draw on
        transform: Coordinate transform
        x, y: World position
        heading: Heading angle in radians
        agent_type: Agent type index (dataset-specific mapping)
        is_focal: Whether this is the focal agent
        alpha: Transparency (unused currently)
        length: Optional actual length in meters (uses config default if None)
        width: Optional actual width in meters (uses config default if None)
        dataset_type: Dataset type ('venti3d', 'argoverse_v2') for shape lookup
    """
    cx, cy = transform.world_to_image(x, y)

    # Look up render config for this agent type from YAML config
    config = get_agent_render_config(dataset_type or 'venti3d')
    shape, default_length, default_width = config.get(agent_type, ('tiny_circle', None, None))

    # Determine color based on focal status
    color = COLORS['focal_agent'] if is_focal else COLORS['other_vehicle']

    if shape == 'box':
        length_m = length if (length is not None and length > 0) else default_length
        width_m = width if (width is not None and width > 0) else default_width
        _draw_box_agent(img, transform, cx, cy, heading, length_m, width_m, color, is_focal)

    elif shape == 'small_box':
        length_m = length if (length is not None and length > 0) else default_length
        width_m = width if (width is not None and width > 0) else default_width
        _draw_small_box_agent(img, transform, cx, cy, heading, length_m, width_m, color, is_focal)

    elif shape == 'circle':
        radius_m = (width / 2) if (width is not None and width > 0) else (default_width / 2 if default_width else 0.5)
        _draw_circle_agent(img, transform, cx, cy, radius_m, color, is_focal)

    else:  # tiny_circle (unknown/static types)
        _draw_tiny_circle_agent(img, cx, cy, color, is_focal)


def draw_dashed_line(img: np.ndarray, pts: np.ndarray, color: tuple,
                     thickness: int = 1, dash_length: int = 8) -> None:
    """Draw a dashed polyline."""
    for i in range(0, len(pts) - 1, 2):
        end_idx = min(i + 1, len(pts) - 1)
        cv2.line(img, tuple(pts[i]), tuple(pts[end_idx]), color, thickness, cv2.LINE_AA)


def draw_trajectory_arrow_head(img: np.ndarray, start: Tuple[int, int], end: Tuple[int, int],
                               color: Tuple[int, int, int], size: int = 10, thickness: int = 2) -> None:
    """Draw an arrow head at 'end' pointing in direction from 'start' to 'end'."""
    # Calculate direction vector
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx*dx + dy*dy)
    if length < 1:
        return

    # Normalize
    dx, dy = dx/length, dy/length

    # Arrow head points (two lines from end point going backward at angles)
    angle = np.pi / 6  # 30 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Left wing
    lx = end[0] - size * (dx * cos_a + dy * sin_a)
    ly = end[1] - size * (dy * cos_a - dx * sin_a)

    # Right wing
    rx = end[0] - size * (dx * cos_a - dy * sin_a)
    ry = end[1] - size * (dy * cos_a + dx * sin_a)

    cv2.line(img, end, (int(lx), int(ly)), color, thickness, cv2.LINE_AA)
    cv2.line(img, end, (int(rx), int(ry)), color, thickness, cv2.LINE_AA)


def _interpolate_point_at_distance(points: np.ndarray, cum_dist: np.ndarray,
                                    target_dist: float) -> Tuple[int, int]:
    """Find the point along a polyline at the given cumulative distance."""
    # Find segment containing target distance
    idx = np.searchsorted(cum_dist, target_dist, side='right') - 1
    idx = max(0, min(idx, len(points) - 2))

    # Interpolate within segment
    seg_start_dist = cum_dist[idx]
    seg_length = cum_dist[idx + 1] - seg_start_dist

    if seg_length < 1e-6:
        return (int(points[idx][0]), int(points[idx][1]))

    t = (target_dist - seg_start_dist) / seg_length
    t = max(0.0, min(1.0, t))

    x = points[idx][0] + t * (points[idx + 1][0] - points[idx][0])
    y = points[idx][1] + t * (points[idx + 1][1] - points[idx][1])

    return (int(x), int(y))


def draw_dotted_polyline(img: np.ndarray, points: np.ndarray, color: Tuple[int, int, int],
                         thickness: int = 2, dash_length: int = 6, gap_length: int = 6) -> None:
    """Draw a dotted/dashed polyline along the entire path."""
    if len(points) < 2:
        return

    # Calculate cumulative distances along the path
    diffs = np.diff(points, axis=0).astype(np.float64)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cum_dist[-1]

    if total_length < 1:
        return

    # Draw dashes along the path
    pattern_length = dash_length + gap_length
    current_dist = 0.0

    while current_dist < total_length:
        # Dash start and end distances
        dash_start = current_dist
        dash_end = min(current_dist + dash_length, total_length)

        # Find points along path at these distances
        start_pt = _interpolate_point_at_distance(points, cum_dist, dash_start)
        end_pt = _interpolate_point_at_distance(points, cum_dist, dash_end)

        cv2.line(img, start_pt, end_pt, color, thickness, cv2.LINE_AA)

        current_dist += pattern_length


def draw_map(img: np.ndarray, transform: CoordinateTransform,
             map_cache: Dict) -> None:
    """
    Draw lane graph with proper lane boundaries.
    Uses connected polylines when polygon groupings are available.
    """
    if 'point_pos' not in map_cache:
        return

    point_pos = map_cache['point_pos']
    point_type = map_cache.get('point_type')
    point_side = map_cache.get('point_side')
    pt_to_pl = map_cache.get('pt_to_pl')

    if pt_to_pl is not None and point_side is not None:
        polygon_ids = np.unique(pt_to_pl[1])

        for pl_id in polygon_ids:
            mask = pt_to_pl[1] == pl_id
            pt_indices = pt_to_pl[0][mask]

            if len(pt_indices) == 0:
                continue

            # Draw boundaries by side: 0=LEFT, 1=RIGHT, 2=CENTER
            for side, color, is_dashed in [
                (0, COLORS['lane_boundary'], False),  # LEFT - solid
                (1, COLORS['lane_boundary'], False),  # RIGHT - solid
                (2, COLORS['centerline'], True),      # CENTER - dashed
            ]:
                side_mask = point_side[pt_indices] == side
                if not side_mask.any():
                    continue

                pts = point_pos[pt_indices[side_mask]]
                if len(pts) < 2:
                    continue

                # Sort points by distance along the lane (use cumulative distance)
                if len(pts) > 2:
                    diffs = np.diff(pts, axis=0)
                    dists = np.sqrt((diffs ** 2).sum(axis=1))
                    cum_dists = np.concatenate([[0], np.cumsum(dists)])
                    sort_idx = np.argsort(cum_dists)
                    pts = pts[sort_idx]

                img_pts = transform.world_to_image_array(pts)

                if is_dashed:
                    draw_dashed_line(img, img_pts, color, 1)
                else:
                    cv2.polylines(img, [img_pts], False, color, 1, cv2.LINE_AA)
    else:
        img_points = transform.world_to_image_array(point_pos)

        if point_type is not None:
            crosswalk_mask = point_type == 15
            for i in np.where(crosswalk_mask)[0]:
                cv2.circle(img, tuple(img_points[i]), 2, COLORS['crosswalk'], -1)

            lane_mask = ~crosswalk_mask
            for i in np.where(lane_mask)[0]:
                cv2.circle(img, tuple(img_points[i]), 1, COLORS['lane_boundary'], -1)
        else:
            for i in range(len(img_points)):
                cv2.circle(img, tuple(img_points[i]), 1, COLORS['lane_boundary'], -1)


def draw_scale_bar(img: np.ndarray, transform: CoordinateTransform,
                   img_size: Tuple[int, int], bar_length_m: float = 10.0) -> None:
    """
    Draw a scale bar in the bottom-right corner.

    Args:
        img: Image to draw on
        transform: Coordinate transform
        img_size: (width, height) of image
        bar_length_m: Length of scale bar in meters
    """
    bar_length_px = transform.scale_length(bar_length_m)

    padding = 20
    bar_height = 6
    x_end = img_size[0] - padding
    x_start = x_end - bar_length_px
    y = img_size[1] - padding - 20

    cv2.rectangle(img, (x_start - 2, y - 2), (x_end + 2, y + bar_height + 2),
                 (30, 30, 30), -1)
    cv2.rectangle(img, (x_start, y), (x_end, y + bar_height),
                 (255, 255, 255), -1)

    cv2.line(img, (x_start, y - 3), (x_start, y + bar_height + 3),
            (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (x_end, y - 3), (x_end, y + bar_height + 3),
            (255, 255, 255), 1, cv2.LINE_AA)

    label = f'{int(bar_length_m)}m'
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x = x_start + (bar_length_px - text_w) // 2
    text_y = y - 6
    cv2.putText(img, label, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def trajectories_overlap(traj1: np.ndarray, traj2: np.ndarray,
                         threshold: float = 2.0) -> bool:
    """
    Check if two trajectories overlap significantly.

    Args:
        traj1, traj2: Trajectory arrays of shape (num_steps, 2)
        threshold: Distance threshold in meters for overlap detection

    Returns:
        True if trajectories overlap (mean point distance < threshold)
    """
    # Use the shorter length for comparison
    min_len = min(len(traj1), len(traj2))
    if min_len == 0:
        return False

    # Compute mean distance between corresponding points
    distances = np.linalg.norm(traj1[:min_len] - traj2[:min_len], axis=1)
    mean_dist = np.mean(distances)

    return mean_dist < threshold


def draw_predictions(img: np.ndarray, transform: CoordinateTransform,
                    predictions: np.ndarray, probs: np.ndarray,
                    horizon: int, top_k: int = 6,
                    filter_overlaps: bool = True,
                    overlap_threshold: float = 0.5,
                    scales: Optional[np.ndarray] = None) -> None:
    """
    Draw trajectory predictions with probability-based styling.
    Predictions should already be in world coordinates.
    Uses a hot-to-cold color scheme where hot (red) = high confidence,
    cold (blue) = low confidence.

    Uncertainty ellipses are always drawn for the top prediction (rank 0) when
    scales are provided.

    Args:
        img: Image to draw on
        transform: Coordinate transform
        predictions: Shape (num_modes, future_steps, 2) in world coordinates
        probs: Shape (num_modes,) probabilities for each mode
        horizon: Number of future steps to display
        top_k: Number of top predictions to show
        filter_overlaps: If True, skip low-confidence predictions that overlap
                        with higher-confidence ones
        overlap_threshold: Distance threshold in meters for overlap detection
        scales: Optional shape (num_modes, future_steps, 2) uncertainty scales (Laplace)
    """
    sorted_idx = np.argsort(probs)[::-1][:top_k]

    # Hot-to-cold color scheme (BGR): red (hot/high confidence) to blue (cold/low confidence)
    prediction_colors = [
        (0, 0, 255),      # Rank 0: Red (hottest - highest confidence)
        (0, 128, 255),    # Rank 1: Orange
        (0, 255, 255),    # Rank 2: Yellow
        (0, 255, 128),    # Rank 3: Yellow-green
        (255, 255, 0),    # Rank 4: Cyan
        (255, 128, 0),    # Rank 5: Blue (coldest - lowest confidence)
    ]

    # Filter overlapping trajectories if enabled
    if filter_overlaps:
        accepted_indices = []
        accepted_trajs = []
        for idx in sorted_idx:
            traj = predictions[idx]
            show_steps = min(horizon, len(traj))
            if show_steps <= 1:
                continue
            traj_show = traj[:show_steps]

            # Check if this trajectory overlaps with any already-accepted one
            overlaps = False
            for accepted_traj in accepted_trajs:
                if trajectories_overlap(traj_show, accepted_traj, overlap_threshold):
                    overlaps = True
                    break

            if not overlaps:
                accepted_indices.append(idx)
                accepted_trajs.append(traj_show)

        # Create mapping from accepted index to original rank
        draw_list = [(rank, idx) for rank, idx in enumerate(sorted_idx) if idx in accepted_indices]
    else:
        draw_list = list(enumerate(sorted_idx))

    # Draw lower-ranked predictions first (so top prediction is on top)
    for rank, idx in reversed(draw_list):
        traj = predictions[idx]  # Already in world coordinates

        show_steps = min(horizon, len(traj))
        if show_steps <= 1:
            continue

        traj_show = traj[:show_steps]
        img_points = transform.world_to_image_array(traj_show)

        color = prediction_colors[min(rank, 5)]

        # Get scales for this mode if available
        mode_scales = scales[idx][:show_steps] if scales is not None else None

        # Draw confidence ellipses for top prediction only (rank 0)
        if rank == 0 and mode_scales is not None:
            # Draw ellipses at every 3rd timestep for smooth coverage
            for step_idx in range(0, show_steps, 3):
                # Compute heading from trajectory direction
                if step_idx < show_steps - 1:
                    dx = traj_show[step_idx + 1, 0] - traj_show[step_idx, 0]
                    dy = traj_show[step_idx + 1, 1] - traj_show[step_idx, 1]
                    heading = np.arctan2(dy, dx)
                elif step_idx > 0:
                    dx = traj_show[step_idx, 0] - traj_show[step_idx - 1, 0]
                    dy = traj_show[step_idx, 1] - traj_show[step_idx - 1, 1]
                    heading = np.arctan2(dy, dx)
                else:
                    heading = 0.0

                scale_x, scale_y = mode_scales[step_idx]
                draw_confidence_ellipse(
                    img, transform,
                    traj_show[step_idx, 0], traj_show[step_idx, 1],
                    scale_x, scale_y, heading, alpha=0.7, scale_factor=1.0
                )

        # Draw dotted line with consistent 2px thickness
        draw_dotted_polyline(img, img_points, color, thickness=2)

        # Draw arrow head for ALL predictions
        # Use a point further back for stable direction (avoid jitter from close points)
        if len(img_points) >= 2:
            end_point = tuple(img_points[-1])
            # Find a point at least min_distance pixels back for stable direction
            min_distance = 15
            start = None
            for i in range(len(img_points) - 2, -1, -1):
                dx = img_points[-1][0] - img_points[i][0]
                dy = img_points[-1][1] - img_points[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist >= min_distance:
                    start = tuple(img_points[i])
                    break
            # Fallback to first point if trajectory is too short
            if start is None:
                start = tuple(img_points[0])
            draw_trajectory_arrow_head(img, start, end_point, color, size=12, thickness=2)

        # Probability labels only for top 3
        if rank < 3:
            if len(img_points) < 2:
                end_point = tuple(img_points[-1])
            prob = probs[idx]
            label_offset_y = (rank - 1) * 12
            label_x = end_point[0] + 14
            label_y = end_point[1] + label_offset_y

            label = f'{prob:.0%}'
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (label_x - 1, label_y - text_h - 2),
                         (label_x + text_w + 1, label_y + 2), (30, 30, 30), -1)
            cv2.putText(img, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_ground_truth(img: np.ndarray, transform: CoordinateTransform,
                     gt_traj: np.ndarray, valid_mask: np.ndarray,
                     future_idx: Optional[int] = None,
                     debug: bool = False) -> None:
    """
    Draw ground truth trajectory.
    Ground truth should already be in world coordinates.

    Args:
        img: Image to draw on
        transform: Coordinate transform
        gt_traj: Ground truth trajectory positions
        valid_mask: Valid mask for each timestep
        future_idx: If provided, only show up to this timestep (for progressive reveal).
                   If None, show entire trajectory.
        debug: Enable debug print statements
    """
    if debug:
        print(f"  [GT DEBUG] future_idx={future_idx}, gt_traj shape={gt_traj.shape if gt_traj is not None else None}, valid_mask sum={valid_mask.sum() if valid_mask is not None else None}")

    if gt_traj is None or len(gt_traj) == 0:
        if debug:
            print(f"  [GT DEBUG] Early return: gt_traj is None or empty")
        return

    # Determine how much of the trajectory to show
    if future_idx is not None:
        if future_idx < 0:
            if debug:
                print(f"  [GT DEBUG] Early return: future_idx={future_idx} < 0")
            return
        show_steps = min(future_idx + 1, len(gt_traj))
        if show_steps <= 1:
            if debug:
                print(f"  [GT DEBUG] Early return: show_steps={show_steps}")
            return
        traj_show = gt_traj[:show_steps]
        mask_show = valid_mask[:show_steps]
    else:
        # Show entire trajectory
        traj_show = gt_traj
        mask_show = valid_mask

    if not mask_show.any():
        if debug:
            print(f"  [GT DEBUG] Early return: no valid points in mask_show (len={len(mask_show)})")
        return

    if debug:
        print(f"  [GT DEBUG] Drawing {mask_show.sum()} valid points out of {len(mask_show)}")

    img_points = transform.world_to_image_array(traj_show)
    color = COLORS['ground_truth']

    valid_indices = np.where(mask_show)[0]
    if len(valid_indices) >= 2:
        valid_points = img_points[valid_indices]
        cv2.polylines(img, [valid_points], False, color, 3, cv2.LINE_AA)

    for i in valid_indices:
        cv2.circle(img, tuple(img_points[i]), 3, color, -1, cv2.LINE_AA)

    if len(valid_indices) > 0:
        last_valid = valid_indices[-1]
        end_point = tuple(img_points[last_valid])
        cv2.drawMarker(img, end_point, color, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)


def draw_ego_marker(img: np.ndarray, transform: CoordinateTransform,
                    x: float, y: float, radius_m: float = 3.0,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2) -> None:
    """
    Draw a distinct circle outline marker around the ego vehicle position.

    Args:
        img: Image to draw on
        transform: Coordinate transform
        x, y: World position of ego vehicle
        radius_m: Radius of the marker circle in meters
        color: BGR color for the marker (default white)
        thickness: Line thickness for the circle outline
    """
    cx, cy = transform.world_to_image(x, y)
    radius_px = transform.scale_length(radius_m)
    cv2.circle(img, (cx, cy), radius_px, color, thickness, cv2.LINE_AA)


def draw_history_trail(img: np.ndarray, transform: CoordinateTransform,
                      positions: np.ndarray, valid_mask: np.ndarray,
                      frame_idx: int, trail_length: int = 20) -> None:
    """
    Draw fading history trail for an agent.
    Uses smooth alpha gradient from old (faint) to new (bright).
    """
    start_idx = max(0, frame_idx - trail_length)
    end_idx = frame_idx + 1

    if end_idx <= start_idx:
        return

    trail_mask = valid_mask[start_idx:end_idx]
    trail = positions[start_idx:end_idx]

    if not trail_mask.any():
        return

    # Get all valid indices
    valid_indices = np.where(trail_mask)[0]
    if len(valid_indices) < 2:
        return

    img_points = transform.world_to_image_array(trail)
    color = np.array(COLORS['history_trail'], dtype=np.float32)
    total_points = len(valid_indices)

    # Draw line segments between consecutive valid points
    for j in range(len(valid_indices) - 1):
        i = valid_indices[j]
        i_next = valid_indices[j + 1]

        # Calculate alpha based on position in trail (older = fainter)
        t = j / max(1, total_points - 1)
        alpha = 0.2 + 0.8 * (t ** 0.7)
        faded_color = tuple(int(c * alpha) for c in color)

        cv2.line(img, tuple(img_points[i]), tuple(img_points[i_next]),
                faded_color, 1, cv2.LINE_AA)
