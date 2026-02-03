"""Converter for Venti3D MotionDB format to QCNet preprocessed format."""
import argparse
import json
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from utils import safe_list_index
from utils import side_to_directed_lineseg
from utils import load_dataset_config

# QCNet point types
POINT_TYPES = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
               'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
               'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
               'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']

POINT_SIDES = ['LEFT', 'RIGHT', 'CENTER']

# Polygon to polygon relationship types
POLYGON_TO_POLYGON_TYPES = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']


class Venti3DToQCNetConverter:
    """Converts Venti3D MotionDB scenarios to QCNet preprocessed format.

    Args:
        motion_dir: Directory containing MotionDB pickle files.
        output_dir: Directory to save converted QCNet pickle files.
        dim: Dimensionality (2 or 3). Default 2 for XY only.
        num_historical_steps: Number of historical time steps. Default 50.
        num_future_steps: Number of future time steps. Default 60.
        predict_unseen_agents: If False, filter out agents unseen during history.
        vector_repr: If True, time step t is valid only when t and t-1 are valid.
    """

    def __init__(self,
                 motion_dir: str,
                 output_dir: str,
                 dim: int = 2,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True) -> None:
        self.motion_dir = Path(motion_dir)
        self.output_dir = Path(output_dir)
        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr

        # Load type mappings from config
        config = load_dataset_config('venti3d')
        self._agent_types = config['agent_types']
        self._polygon_types = config['polygon_types']
        self._agent_type_mapping = config.get('agent_type_mapping', {})
        self._lane_type_mapping = {int(k): v for k, v in config.get('lane_type_mapping', {}).items()}
        self._crosswalk_type = config.get('crosswalk_type', 1)
        self._unknown_type = config.get('unknown_type', 0)

        self._agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        self._polygon_is_intersections = [True, False, None]
        self._polygon_is_bidirectionals = [True, False]
        self._point_types = POINT_TYPES
        self._point_sides = POINT_SIDES
        self._polygon_to_polygon_types = POLYGON_TO_POLYGON_TYPES

    def convert_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single MotionDB scenario to QCNet format.

        Args:
            scenario: MotionDB scenario dictionary.

        Returns:
            QCNet-compatible dictionary.
        """
        data = {}
        data['scenario_id'] = scenario.get('scenario_id', 'unknown')
        data['city'] = scenario.get('map_location', 'venti3d')
        data['agent'] = self._convert_agents(scenario)

        # Convert map features
        map_data = self._convert_map_features(scenario)
        data.update(map_data)

        return data

    def _convert_agents(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent tensors from MotionDB tracks.

        Ego vehicle is stored separately from tracks in MotionDB and must be
        added as a synthetic track.

        Args:
            scenario: MotionDB scenario dictionary.

        Returns:
            Dictionary with agent features in QCNet format.
        """
        tracks = scenario.get('tracks', [])

        # objects_of_interest contains indices into the tracks array, not track IDs
        # Convert indices to track_ids for comparison later
        objects_of_interest_indices = scenario.get('objects_of_interest', [])
        objects_of_interest = set(
            tracks[i].get('track_id')
            for i in objects_of_interest_indices
            if i < len(tracks) and tracks[i].get('track_id') is not None
        )

        # Ego vehicle data (stored separately from tracks in MotionDB)
        ego_positions = scenario.get('ego_positions', [])
        ego_headings = scenario.get('ego_headings', [])
        ego_velocities = scenario.get('ego_velocities', [])
        has_ego = len(ego_positions) >= self.num_steps

        # Build agent ID list: ego first (if exists), then other tracks
        # Skip 'ignore' type objects entirely
        agent_ids: List[Any] = []
        if has_ego:
            agent_ids.append('AV')  # Ego vehicle ID

        for track in tracks:
            track_id = track.get('track_id')
            obj_type = track.get('object_type', '')
            if track_id is not None and obj_type.lower() != 'ignore':
                agent_ids.append(track_id)

        if not self.predict_unseen_agents:
            # Filter to agents seen in historical steps (keep ego always)
            visible_agent_ids = set()
            if has_ego:
                visible_agent_ids.add('AV')

            for track in tracks:
                track_id = track.get('track_id')
                states = track.get('states', [])
                for t, state in enumerate(states):
                    if t < self.num_historical_steps and state.get('valid', False):
                        visible_agent_ids.add(track_id)
                        break
            agent_ids = [aid for aid in agent_ids if aid in visible_agent_ids]

        num_agents = len(agent_ids)
        if num_agents == 0:
            return self._empty_agent_data()

        # Ego is always at index 0 if present
        av_idx = 0 if has_ego else -1

        # Initialize tensors
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        length = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        width = torch.zeros(num_agents, self.num_steps, dtype=torch.float)

        # Build track lookup
        track_lookup = {t.get('track_id'): t for t in tracks}

        for agent_idx, track_id in enumerate(agent_ids):
            agent_id[agent_idx] = str(track_id)

            if track_id == 'AV':
                # Process ego vehicle (synthetic track from ego_* fields)
                agent_type[agent_idx] = 0  # vehicle
                agent_category[agent_idx] = 1  # UNSCORED_TRACK (ego is not predicted)

                for t in range(min(self.num_steps, len(ego_positions))):
                    valid_mask[agent_idx, t] = True

                    # Position
                    pos = ego_positions[t]
                    position[agent_idx, t, 0] = pos[0]
                    position[agent_idx, t, 1] = pos[1]
                    if self.dim == 3 and len(pos) > 2:
                        position[agent_idx, t, 2] = pos[2]

                    # Heading
                    if t < len(ego_headings):
                        heading[agent_idx, t] = ego_headings[t]

                    # Velocity
                    if t < len(ego_velocities):
                        vel = ego_velocities[t]
                        velocity[agent_idx, t, 0] = vel[0]
                        velocity[agent_idx, t, 1] = vel[1]
                        if self.dim == 3 and len(vel) > 2:
                            velocity[agent_idx, t, 2] = vel[2]
            else:
                # Process regular track
                track = track_lookup.get(track_id)
                if track is None:
                    continue

                # Get object type
                obj_type = track.get('object_type', 'unknown')
                if isinstance(obj_type, str):
                    agent_type[agent_idx] = self._agent_type_mapping.get(
                        obj_type.lower(), len(self._agent_types) - 1)
                else:
                    agent_type[agent_idx] = len(self._agent_types) - 1  # unknown

                # Process states (category will be set after determining validity)
                states = track.get('states', [])
                for t, state in enumerate(states):
                    if t >= self.num_steps:
                        break

                    is_valid = state.get('valid', True)
                    if is_valid:
                        valid_mask[agent_idx, t] = True

                        # Position
                        x = state.get('center_x', state.get('x', 0.0))
                        y = state.get('center_y', state.get('y', 0.0))
                        position[agent_idx, t, 0] = x
                        position[agent_idx, t, 1] = y
                        if self.dim == 3:
                            z = state.get('center_z', state.get('z', 0.0))
                            position[agent_idx, t, 2] = z

                        # Heading
                        heading[agent_idx, t] = state.get('heading', 0.0)

                        # Velocity
                        vx = state.get('velocity_x', state.get('vx', 0.0))
                        vy = state.get('velocity_y', state.get('vy', 0.0))
                        velocity[agent_idx, t, 0] = vx
                        velocity[agent_idx, t, 1] = vy
                        if self.dim == 3:
                            vz = state.get('velocity_z', state.get('vz', 0.0))
                            velocity[agent_idx, t, 2] = vz

                        # Bounding box dimensions
                        length[agent_idx, t] = state.get('length', 4.5)
                        width[agent_idx, t] = state.get('width', 2.0)

            # Current valid mask (valid at last historical step)
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]

            # Set category based on objects_of_interest (track_ids) and validity
            # SCORED_TRACK: agents whose track_id is in objects_of_interest with valid data at current time
            # UNSCORED_TRACK: all other agents (context only, not evaluated)
            if track_id in objects_of_interest and current_valid_mask[agent_idx]:
                agent_category[agent_idx] = 2  # SCORED_TRACK
            else:
                agent_category[agent_idx] = 1  # UNSCORED_TRACK

            # Predict mask - only future frames for agents valid at current time
            predict_mask[agent_idx, :] = valid_mask[agent_idx, :]
            if self.vector_repr:
                # A time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1:self.num_historical_steps] = (
                    valid_mask[agent_idx, :self.num_historical_steps - 1] &
                    valid_mask[agent_idx, 1:self.num_historical_steps])
                valid_mask[agent_idx, 0] = False

            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

        # Select focal track from scored agents with high observability
        # FOCAL_TRACK (3) is a single agent per scenario selected as the primary prediction target
        # Minimum valid future points required to be focal candidate
        # Argoverse requires full observability (60/60); we use 50/60 (83%) as threshold
        MIN_FUTURE_POINTS = 50

        scored_agents = [
            (idx, predict_mask[idx, self.num_historical_steps:].sum().item())
            for idx in range(num_agents)
            if agent_category[idx] == 2  # SCORED_TRACK
        ]
        # Filter to agents with high observability (like Argoverse's full-observation requirement)
        focal_candidates = [(idx, pts) for idx, pts in scored_agents if pts >= MIN_FUTURE_POINTS]
        if focal_candidates:
            # Random selection among qualified candidates
            focal_idx = random.choice(focal_candidates)[0]
            agent_category[focal_idx] = 3  # Upgrade to FOCAL_TRACK
        elif scored_agents:
            # Fallback: if no agent meets threshold, pick best available
            focal_idx = max(scored_agents, key=lambda x: x[1])[0]
            agent_category[focal_idx] = 3

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': agent_id,
            'type': agent_type,
            'category': agent_category,
            'position': position,
            'heading': heading,
            'velocity': velocity,
            'length': length,
            'width': width,
        }

    def _empty_agent_data(self) -> Dict[str, Any]:
        """Return empty agent data structure."""
        return {
            'num_nodes': 0,
            'av_index': 0,
            'valid_mask': torch.zeros(0, self.num_steps, dtype=torch.bool),
            'predict_mask': torch.zeros(0, self.num_steps, dtype=torch.bool),
            'id': [],
            'type': torch.zeros(0, dtype=torch.uint8),
            'category': torch.zeros(0, dtype=torch.uint8),
            'position': torch.zeros(0, self.num_steps, self.dim, dtype=torch.float),
            'heading': torch.zeros(0, self.num_steps, dtype=torch.float),
            'velocity': torch.zeros(0, self.num_steps, self.dim, dtype=torch.float),
            'length': torch.zeros(0, self.num_steps, dtype=torch.float),
            'width': torch.zeros(0, self.num_steps, dtype=torch.float),
        }

    def _convert_map_features(self, scenario: Dict[str, Any]) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        """Convert lane and crosswalk data to QCNet map format.

        Args:
            scenario: MotionDB scenario dictionary.

        Returns:
            Dictionary with map_polygon, map_point, and edge data.
        """
        # MotionDB uses list-based indexing for lane data
        # lane_ids: list of lane IDs (e.g., [58, 62, 63, ...])
        # lane_centerlines: list indexed by position (0, 1, 2...), not by lane_id
        # lane_connectivity: dict keyed by lane_id with {'next': [], 'prev': []}
        lane_ids = scenario.get('lane_ids', [])
        lane_centerlines = scenario.get('lane_centerlines', [])
        lane_types = scenario.get('lane_types', [])
        lane_left_edges = scenario.get('lane_left_edges', [])
        lane_right_edges = scenario.get('lane_right_edges', [])
        lane_connectivity = scenario.get('lane_connectivity', {})
        lane_adjacent_ids = scenario.get('lane_adjacent_ids', [])
        crosswalk_polygons = scenario.get('crosswalk_polygons', [])
        crosswalk_ids = scenario.get('crosswalk_ids', [])

        # Build lane_id to index mapping for connectivity lookups
        lane_id_to_idx = {lid: idx for idx, lid in enumerate(lane_ids)}

        num_lane_segments = len(lane_centerlines)
        num_crosswalks = len(crosswalk_polygons)
        num_polygons = num_lane_segments + num_crosswalks * 2  # Crosswalks have forward/backward

        if num_polygons == 0:
            return self._empty_map_data()

        # Initialize polygon tensors
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_bidirectional = torch.zeros(num_polygons, dtype=torch.uint8)

        # Point data stored per polygon
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        # Process lane segments (using list indexing)
        for lane_idx, centerline_data in enumerate(lane_centerlines):
            # Convert centerline to tensor
            if isinstance(centerline_data, np.ndarray):
                centerline = torch.from_numpy(centerline_data).float()
            elif isinstance(centerline_data, list):
                centerline = torch.tensor(centerline_data, dtype=torch.float)
            else:
                centerline = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float)

            if centerline.dim() == 1:
                centerline = centerline.unsqueeze(0)

            # Ensure at least 2 points
            if centerline.size(0) < 2:
                centerline = torch.cat([centerline, centerline + torch.tensor([[1, 0, 0]])], dim=0)

            # Ensure 3D (add z=0 if needed)
            if centerline.size(1) == 2:
                centerline = torch.cat([centerline, torch.zeros(centerline.size(0), 1)], dim=1)

            # Set polygon position and orientation
            polygon_position[lane_idx] = centerline[0, :self.dim]
            polygon_orientation[lane_idx] = torch.atan2(
                centerline[1, 1] - centerline[0, 1],
                centerline[1, 0] - centerline[0, 0])
            polygon_height[lane_idx] = centerline[1, 2] - centerline[0, 2]

            # Lane type (list indexed by position)
            lane_type = lane_types[lane_idx] if lane_idx < len(lane_types) else 0
            polygon_type[lane_idx] = self._lane_type_mapping.get(lane_type, self._unknown_type)

            # Intersection status: default to False, will be updated based on connectivity
            polygon_is_intersection[lane_idx] = 1  # False (not intersection)

            # Bidirectional status from source data
            lane_bidirectionals = scenario.get('lane_bidirectional', [])
            is_bidir = lane_bidirectionals[lane_idx] if lane_idx < len(lane_bidirectionals) else False
            polygon_is_bidirectional[lane_idx] = 0 if is_bidir else 1  # 0=True, 1=False

            # Get boundaries (list indexed by position)
            left_boundary_data = lane_left_edges[lane_idx] if lane_idx < len(lane_left_edges) else None
            right_boundary_data = lane_right_edges[lane_idx] if lane_idx < len(lane_right_edges) else None

            # Convert boundaries to tensors
            if left_boundary_data is not None:
                if isinstance(left_boundary_data, np.ndarray):
                    left_boundary = torch.from_numpy(left_boundary_data).float()
                else:
                    left_boundary = torch.tensor(left_boundary_data, dtype=torch.float)
            else:
                # Create default boundary offset from centerline
                left_boundary = centerline.clone()
                left_boundary[:, 1] += 1.5  # Offset left

            if right_boundary_data is not None:
                if isinstance(right_boundary_data, np.ndarray):
                    right_boundary = torch.from_numpy(right_boundary_data).float()
                else:
                    right_boundary = torch.tensor(right_boundary_data, dtype=torch.float)
            else:
                right_boundary = centerline.clone()
                right_boundary[:, 1] -= 1.5  # Offset right

            # Ensure 3D
            if left_boundary.size(1) == 2:
                left_boundary = torch.cat([left_boundary, torch.zeros(left_boundary.size(0), 1)], dim=1)
            if right_boundary.size(1) == 2:
                right_boundary = torch.cat([right_boundary, torch.zeros(right_boundary.size(0), 1)], dim=1)

            # Build point data
            point_position[lane_idx] = torch.cat([
                left_boundary[:-1, :self.dim],
                right_boundary[:-1, :self.dim],
                centerline[:-1, :self.dim]], dim=0)

            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]

            point_orientation[lane_idx] = torch.cat([
                torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)

            point_magnitude[lane_idx] = torch.norm(torch.cat([
                left_vectors[:, :2],
                right_vectors[:, :2],
                center_vectors[:, :2]], dim=0), p=2, dim=-1)

            point_height[lane_idx] = torch.cat([
                left_vectors[:, 2],
                right_vectors[:, 2],
                center_vectors[:, 2]], dim=0)

            # Point types - use CENTERLINE for all (can be enhanced with boundary markings)
            center_type_idx = self._point_types.index('CENTERLINE')
            none_type_idx = self._point_types.index('NONE')
            point_type[lane_idx] = torch.cat([
                torch.full((len(left_vectors),), none_type_idx, dtype=torch.uint8),
                torch.full((len(right_vectors),), none_type_idx, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type_idx, dtype=torch.uint8)], dim=0)

            # Point sides
            point_side[lane_idx] = torch.cat([
                torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        # Process crosswalks (list indexed by position)
        for cw_idx, corners_data in enumerate(crosswalk_polygons):
            polygon_idx = num_lane_segments + cw_idx
            polygon_idx_reverse = num_lane_segments + num_crosswalks + cw_idx
            if isinstance(corners_data, np.ndarray):
                corners = torch.from_numpy(corners_data).float()
            else:
                corners = torch.tensor(corners_data, dtype=torch.float)

            # Ensure 3D
            if corners.size(1) == 2:
                corners = torch.cat([corners, torch.zeros(corners.size(0), 1)], dim=1)

            # Extract two edges (assuming corners are ordered)
            if corners.size(0) >= 4:
                edge1 = corners[[0, 1]]
                edge2 = corners[[3, 2]]  # Reversed for direction
            else:
                # Fallback: create simple edges
                edge1 = corners[:2] if corners.size(0) >= 2 else corners
                edge2 = edge1.clone()

            # Compute start and end positions (midpoints of edges)
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2

            # Forward polygon
            polygon_position[polygon_idx] = start_position[:self.dim]
            polygon_orientation[polygon_idx] = torch.atan2(
                (end_position - start_position)[1],
                (end_position - start_position)[0])
            polygon_height[polygon_idx] = end_position[2] - start_position[2]
            polygon_type[polygon_idx] = self._crosswalk_type
            polygon_is_intersection[polygon_idx] = 2  # None/Unknown (match Argoverse)
            polygon_is_bidirectional[polygon_idx] = 0  # True (bidirectional)

            # Backward polygon
            polygon_position[polygon_idx_reverse] = end_position[:self.dim]
            polygon_orientation[polygon_idx_reverse] = torch.atan2(
                (start_position - end_position)[1],
                (start_position - end_position)[0])
            polygon_height[polygon_idx_reverse] = start_position[2] - end_position[2]
            polygon_type[polygon_idx_reverse] = self._crosswalk_type
            polygon_is_intersection[polygon_idx_reverse] = 2  # None/Unknown (match Argoverse)
            polygon_is_bidirectional[polygon_idx_reverse] = 0  # True (bidirectional)

            # Determine left/right boundaries based on side
            mid_edge1 = (edge1[0] + edge1[-1]) / 2
            if side_to_directed_lineseg(mid_edge1, start_position, end_position) == 'LEFT':
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1

            # Compute centerline
            num_centerline_points = max(2, math.ceil(
                torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1)
            t_vals = torch.linspace(0, 1, int(num_centerline_points))
            centerline = start_position.unsqueeze(0) + t_vals.unsqueeze(1) * (end_position - start_position).unsqueeze(0)

            # Forward crosswalk points
            point_position[polygon_idx] = torch.cat([
                left_boundary[:-1, :self.dim] if left_boundary.size(0) > 1 else left_boundary[:, :self.dim],
                right_boundary[:-1, :self.dim] if right_boundary.size(0) > 1 else right_boundary[:, :self.dim],
                centerline[:-1, :self.dim]], dim=0)

            left_vectors = left_boundary[1:] - left_boundary[:-1] if left_boundary.size(0) > 1 else left_boundary
            right_vectors = right_boundary[1:] - right_boundary[:-1] if right_boundary.size(0) > 1 else right_boundary
            center_vectors = centerline[1:] - centerline[:-1]

            # Handle edge case of single point boundaries
            if left_vectors.dim() == 1 or left_vectors.size(0) == 0:
                left_vectors = center_vectors[:1].clone()
            if right_vectors.dim() == 1 or right_vectors.size(0) == 0:
                right_vectors = center_vectors[:1].clone()

            point_orientation[polygon_idx] = torch.cat([
                torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)

            point_magnitude[polygon_idx] = torch.norm(torch.cat([
                left_vectors[:, :2],
                right_vectors[:, :2],
                center_vectors[:, :2]], dim=0), p=2, dim=-1)

            point_height[polygon_idx] = torch.cat([
                left_vectors[:, 2],
                right_vectors[:, 2],
                center_vectors[:, 2]], dim=0)

            crosswalk_type_idx = self._point_types.index('CROSSWALK')
            center_type_idx = self._point_types.index('CENTERLINE')
            point_type[polygon_idx] = torch.cat([
                torch.full((len(left_vectors),), crosswalk_type_idx, dtype=torch.uint8),
                torch.full((len(right_vectors),), crosswalk_type_idx, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type_idx, dtype=torch.uint8)], dim=0)

            point_side[polygon_idx] = torch.cat([
                torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

            # Backward crosswalk points (reversed)
            point_position[polygon_idx_reverse] = torch.cat([
                right_boundary.flip(dims=[0])[:-1, :self.dim] if right_boundary.size(0) > 1 else right_boundary[:, :self.dim],
                left_boundary.flip(dims=[0])[:-1, :self.dim] if left_boundary.size(0) > 1 else left_boundary[:, :self.dim],
                centerline.flip(dims=[0])[:-1, :self.dim]], dim=0)

            point_orientation[polygon_idx_reverse] = torch.cat([
                torch.atan2(-right_vectors.flip(dims=[0])[:, 1], -right_vectors.flip(dims=[0])[:, 0]),
                torch.atan2(-left_vectors.flip(dims=[0])[:, 1], -left_vectors.flip(dims=[0])[:, 0]),
                torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)

            point_magnitude[polygon_idx_reverse] = torch.norm(torch.cat([
                -right_vectors.flip(dims=[0])[:, :2],
                -left_vectors.flip(dims=[0])[:, :2],
                -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)

            point_height[polygon_idx_reverse] = torch.cat([
                -right_vectors.flip(dims=[0])[:, 2],
                -left_vectors.flip(dims=[0])[:, 2],
                -center_vectors.flip(dims=[0])[:, 2]], dim=0)

            point_type[polygon_idx_reverse] = torch.cat([
                torch.full((len(right_vectors),), crosswalk_type_idx, dtype=torch.uint8),
                torch.full((len(left_vectors),), crosswalk_type_idx, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type_idx, dtype=torch.uint8)], dim=0)

            point_side[polygon_idx_reverse] = torch.cat([
                torch.full((len(right_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(left_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        # Build point-to-polygon edges
        num_points = torch.tensor([p.size(0) for p in point_position if p is not None], dtype=torch.long)
        if num_points.numel() > 0:
            point_to_polygon_edge_index = torch.stack([
                torch.arange(num_points.sum(), dtype=torch.long),
                torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        else:
            point_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)

        # Build polygon-to-polygon edges (lane connectivity)
        # lane_connectivity is a dict keyed by lane_id (not index)
        # lane_adjacent_ids is a list indexed by position
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []

        for lane_idx, lane_id in enumerate(lane_ids):
            # Connectivity is keyed by lane_id
            connectivity = lane_connectivity.get(lane_id, {})

            # Predecessors
            preds = connectivity.get('predecessors', connectivity.get('prev', []))
            # Successors (read early for intersection inference)
            succs = connectivity.get('successors', connectivity.get('next', []))

            # Refine intersection status based on connectivity:
            # Multiple successors or predecessors indicate intersection
            if len(preds) > 1 or len(succs) > 1:
                polygon_is_intersection[lane_idx] = 0  # True (is intersection)
            if preds:
                pred_inds = []
                for pred_id in preds:
                    pred_idx = lane_id_to_idx.get(pred_id)
                    if pred_idx is not None:
                        pred_inds.append(pred_idx)
                if pred_inds:
                    polygon_to_polygon_edge_index.append(
                        torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                     torch.full((len(pred_inds),), lane_idx, dtype=torch.long)], dim=0))
                    polygon_to_polygon_type.append(
                        torch.full((len(pred_inds),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))

            # Successors (already read above for intersection inference)
            if succs:
                succ_inds = []
                for succ_id in succs:
                    succ_idx = lane_id_to_idx.get(succ_id)
                    if succ_idx is not None:
                        succ_inds.append(succ_idx)
                if succ_inds:
                    polygon_to_polygon_edge_index.append(
                        torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                     torch.full((len(succ_inds),), lane_idx, dtype=torch.long)], dim=0))
                    polygon_to_polygon_type.append(
                        torch.full((len(succ_inds),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))

            # Adjacent lanes (list indexed by position, each item is a list of adjacent lane_ids)
            adjacents = lane_adjacent_ids[lane_idx] if lane_idx < len(lane_adjacent_ids) else []
            if adjacents and isinstance(adjacents, list):
                for adj_id in adjacents:
                    adj_idx = lane_id_to_idx.get(adj_id)
                    if adj_idx is not None:
                        # Use NONE for generic adjacency (MotionDB doesn't specify left/right)
                        polygon_to_polygon_edge_index.append(
                            torch.tensor([[adj_idx], [lane_idx]], dtype=torch.long))
                        polygon_to_polygon_type.append(
                            torch.tensor([self._polygon_to_polygon_types.index('NONE')], dtype=torch.uint8))

        if polygon_to_polygon_edge_index:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        # Build output map data
        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }

        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        if self.dim == 3:
            map_data['map_polygon']['height'] = polygon_height
        map_data['map_polygon']['type'] = polygon_type
        map_data['map_polygon']['is_intersection'] = polygon_is_intersection
        map_data['map_polygon']['is_bidirectional'] = polygon_is_bidirectional

        # Concatenate point data
        valid_point_positions = [p for p in point_position if p is not None]
        if valid_point_positions:
            map_data['map_point']['num_nodes'] = sum(p.size(0) for p in valid_point_positions)
            map_data['map_point']['position'] = torch.cat(valid_point_positions, dim=0)
            map_data['map_point']['orientation'] = torch.cat([o for o in point_orientation if o is not None], dim=0)
            map_data['map_point']['magnitude'] = torch.cat([m for m in point_magnitude if m is not None], dim=0)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.cat([h for h in point_height if h is not None], dim=0)
            map_data['map_point']['type'] = torch.cat([t for t in point_type if t is not None], dim=0)
            map_data['map_point']['side'] = torch.cat([s for s in point_side if s is not None], dim=0)
        else:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float).reshape(0, self.dim)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)

        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data

    def _empty_map_data(self) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        """Return empty map data structure."""
        return {
            'map_polygon': {
                'num_nodes': 0,
                'position': torch.tensor([], dtype=torch.float).reshape(0, self.dim),
                'orientation': torch.tensor([], dtype=torch.float),
                'type': torch.tensor([], dtype=torch.uint8),
                'is_intersection': torch.tensor([], dtype=torch.uint8),
                'is_bidirectional': torch.tensor([], dtype=torch.uint8),
            },
            'map_point': {
                'num_nodes': 0,
                'position': torch.tensor([], dtype=torch.float).reshape(0, self.dim),
                'orientation': torch.tensor([], dtype=torch.float),
                'magnitude': torch.tensor([], dtype=torch.float),
                'type': torch.tensor([], dtype=torch.uint8),
                'side': torch.tensor([], dtype=torch.uint8),
            },
            ('map_point', 'to', 'map_polygon'): {
                'edge_index': torch.tensor([[], []], dtype=torch.long),
            },
            ('map_polygon', 'to', 'map_polygon'): {
                'edge_index': torch.tensor([[], []], dtype=torch.long),
                'type': torch.tensor([], dtype=torch.uint8),
            },
        }

    def convert_all(self, splits_file: str = 'splits.json') -> None:
        """Process all scenarios and save as individual pickle files.

        Reads split assignments from splits.json (generated by venti3d_split.py)
        and converts each split's scenarios to QCNet format.

        Args:
            splits_file: Path to splits JSON file (relative to motion_dir).
        """
        # Load splits from JSON (generated by venti3d_split.py)
        splits_path = self.motion_dir / splits_file
        if not splits_path.exists():
            raise FileNotFoundError(
                f"splits.json not found at {splits_path}. "
                f"Run: python -m datasets.venti3d_split --root {self.motion_dir}"
            )

        with open(splits_path, 'r') as f:
            splits = json.load(f)

        print(f"Loaded splits from {splits_path}")

        # MotionDB stores scenarios in scenarios/ subdirectory
        scenarios_dir = self.motion_dir / 'scenarios'
        if not scenarios_dir.exists():
            raise FileNotFoundError(f"Scenarios directory not found at {scenarios_dir}")

        # Convert each split
        for split_name in ['train', 'val', 'test']:
            if split_name not in splits or not splits[split_name]:
                print(f"No scenarios for {split_name}")
                continue

            # Load scenarios for this split
            scenarios: List[Tuple[str, int, Dict[str, Any]]] = []
            file_scenarios_cache: Dict[str, List] = {}

            print(f"\nLoading {split_name} scenarios...")
            for file_name, indices in tqdm(splits[split_name].items(), desc=f"Loading {split_name}"):
                file_path = scenarios_dir / file_name

                # Cache file contents to avoid reloading
                if file_name not in file_scenarios_cache:
                    try:
                        with open(file_path, 'rb') as f:
                            file_scenarios_cache[file_name] = pickle.load(f)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue

                file_scenarios = file_scenarios_cache[file_name]
                file_stem = file_name.replace('.pkl', '').replace('.pickle', '')

                for idx in indices:
                    if idx < len(file_scenarios):
                        scenarios.append((file_stem, idx, file_scenarios[idx]))
                    else:
                        print(f"Warning: Index {idx} out of range for {file_name} (len={len(file_scenarios)})")

            print(f"Loaded {len(scenarios)} scenarios for {split_name}")
            self._convert_scenarios(scenarios, split_name)

            # Clear cache after each split to free memory
            file_scenarios_cache.clear()

        # Print summary
        total = sum(
            sum(len(indices) for indices in splits[split_name].values())
            for split_name in ['train', 'val', 'test']
            if split_name in splits
        )
        print(f"\nConversion complete. Total scenarios: {total}")

    def _convert_scenarios(self, scenarios: List[Tuple[str, int, Dict[str, Any]]], split: str) -> None:
        """Convert a list of scenarios and save to the specified split directory.

        Args:
            scenarios: List of (file_stem, index, scenario_dict) tuples.
            split: Target split directory ('train', 'val', or 'test').
        """
        if not scenarios:
            print(f"No scenarios for {split}")
            return

        output_dir = self.output_dir / split / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Converting {len(scenarios)} scenarios to {split}...")

        for file_stem, idx, scenario in tqdm(scenarios, desc=f"Converting {split}"):
            try:
                data = self.convert_scenario(scenario)
                out_name = f"{file_stem}_{idx}.pkl"
                with open(output_dir / out_name, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                print(f"Error processing {file_stem}_{idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Conversion complete for {split}. Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Venti3D MotionDB to QCNet format',
        epilog='''
Example workflow:
  1. Generate splits:  python -m datasets.venti3d_split --root /path/to/data
  2. Convert data:     python -m datasets.venti3d_converter --motion_dir /path/to/data --output_dir /path/to/data
  3. Train:            python train_qcnet.py --root /path/to/data --dataset venti3d
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--motion_dir', type=str, required=True,
                        help='Input directory containing MotionDB pickle files (with scenarios/ subdir and splits.json)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for converted QCNet pickle files')
    parser.add_argument('--splits_file', type=str, default='splits.json',
                        help='Path to splits JSON file relative to motion_dir (default: splits.json)')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='Dimensionality (2 for XY, 3 for XYZ)')
    parser.add_argument('--num_historical_steps', type=int, default=50,
                        help='Number of historical time steps')
    parser.add_argument('--num_future_steps', type=int, default=60,
                        help='Number of future time steps')
    parser.add_argument('--predict_unseen_agents', action='store_true',
                        help='Include agents unseen during historical steps')
    parser.add_argument('--no_vector_repr', action='store_true',
                        help='Disable vector representation (valid requires t and t-1)')

    args = parser.parse_args()

    converter = Venti3DToQCNetConverter(
        motion_dir=args.motion_dir,
        output_dir=args.output_dir,
        dim=args.dim,
        num_historical_steps=args.num_historical_steps,
        num_future_steps=args.num_future_steps,
        predict_unseen_agents=args.predict_unseen_agents,
        vector_repr=not args.no_vector_repr,
    )

    converter.convert_all(splits_file=args.splits_file)


if __name__ == '__main__':
    main()
