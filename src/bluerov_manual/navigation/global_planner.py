from dataclasses import dataclass
from typing import List, Sequence, Tuple

import math
import random

from .obstacles import Obstacle


@dataclass
class GlobalPlannerConfig:
    grid_resolution: float = 0.5
    fixed_depth: float | None = None
    bounds_min: Tuple[float, float, float] = (-10.0, -10.0, -5.0)
    bounds_max: Tuple[float, float, float] = (10.0, 10.0, 5.0)
    step_size: float = 1.0
    max_iters: int = 400
    goal_sample_rate: float = 0.1
    neighbor_radius: float = 2.0
    goal_tolerance: float = 1.0


@dataclass
class _Node:
    pos: Tuple[float, float, float]
    parent: int | None
    cost: float


class GlobalPlanner:
    def __init__(self, cfg: GlobalPlannerConfig):
        self.cfg = cfg

    def plan(
        self,
        start_w: Sequence[float],
        goal_w: Sequence[float],
        obstacles: Sequence[Obstacle] | None = None,
    ) -> List[Tuple[float, float, float]]:
        """Return a coarse list of waypoints in world frame."""
        obstacles = list(obstacles or [])
        start = (float(start_w[0]), float(start_w[1]), float(start_w[2]))
        if self.cfg.fixed_depth is not None:
            goal = (float(goal_w[0]), float(goal_w[1]), float(self.cfg.fixed_depth))
        else:
            goal = (float(goal_w[0]), float(goal_w[1]), float(goal_w[2]))

        nodes = [_Node(pos=start, parent=None, cost=0.0)]
        goal_idx = None

        for _ in range(self.cfg.max_iters):
            sample = self._sample(goal)
            nearest_idx = self._nearest(nodes, sample)
            new_pos = self._steer(nodes[nearest_idx].pos, sample)
            if not self._collision_free(nodes[nearest_idx].pos, new_pos, obstacles):
                continue

            new_cost = nodes[nearest_idx].cost + self._dist(nodes[nearest_idx].pos, new_pos)
            new_node = _Node(pos=new_pos, parent=nearest_idx, cost=new_cost)

            neighbor_idxs = self._near(nodes, new_pos)
            best_parent = nearest_idx
            best_cost = new_cost
            for idx in neighbor_idxs:
                cand_cost = nodes[idx].cost + self._dist(nodes[idx].pos, new_pos)
                if cand_cost < best_cost and self._collision_free(nodes[idx].pos, new_pos, obstacles):
                    best_parent = idx
                    best_cost = cand_cost
            new_node.parent = best_parent
            new_node.cost = best_cost
            nodes.append(new_node)
            new_idx = len(nodes) - 1

            for idx in neighbor_idxs:
                cand_cost = new_node.cost + self._dist(nodes[idx].pos, new_pos)
                if cand_cost < nodes[idx].cost and self._collision_free(nodes[idx].pos, new_pos, obstacles):
                    nodes[idx].parent = new_idx
                    nodes[idx].cost = cand_cost

            if self._dist(new_pos, goal) <= self.cfg.goal_tolerance:
                goal_idx = new_idx
                break

        if goal_idx is None:
            return [goal]
        return self._extract_path(nodes, goal_idx, goal)

    def _sample(self, goal: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if random.random() < self.cfg.goal_sample_rate:
            return goal
        return (
            random.uniform(self.cfg.bounds_min[0], self.cfg.bounds_max[0]),
            random.uniform(self.cfg.bounds_min[1], self.cfg.bounds_max[1]),
            random.uniform(self.cfg.bounds_min[2], self.cfg.bounds_max[2]),
        )

    def _nearest(self, nodes: List[_Node], point: Tuple[float, float, float]) -> int:
        dists = [self._dist(n.pos, point) for n in nodes]
        return int(min(range(len(dists)), key=dists.__getitem__))

    def _near(self, nodes: List[_Node], point: Tuple[float, float, float]) -> List[int]:
        return [
            i
            for i, n in enumerate(nodes)
            if self._dist(n.pos, point) <= self.cfg.neighbor_radius
        ]

    def _steer(
        self, from_pos: Tuple[float, float, float], to_pos: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        d = self._dist(from_pos, to_pos)
        if d <= self.cfg.step_size:
            return to_pos
        scale = self.cfg.step_size / d
        return (
            from_pos[0] + (to_pos[0] - from_pos[0]) * scale,
            from_pos[1] + (to_pos[1] - from_pos[1]) * scale,
            from_pos[2] + (to_pos[2] - from_pos[2]) * scale,
        )

    @staticmethod
    def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _collision_free(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        obstacles: Sequence[Obstacle],
    ) -> bool:
        for obs in obstacles:
            if self._segment_sphere_intersect(a, b, obs.position_w, obs.radius):
                return False
        return True

    @staticmethod
    def _segment_sphere_intersect(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        center: Tuple[float, float, float],
        radius: float,
    ) -> bool:
        ax, ay, az = a
        bx, by, bz = b
        cx, cy, cz = center
        abx, aby, abz = (bx - ax), (by - ay), (bz - az)
        acx, acy, acz = (cx - ax), (cy - ay), (cz - az)
        ab_len2 = abx * abx + aby * aby + abz * abz
        if ab_len2 == 0.0:
            dist2 = (ax - cx) ** 2 + (ay - cy) ** 2 + (az - cz) ** 2
            return dist2 <= radius * radius
        t = max(0.0, min(1.0, (acx * abx + acy * aby + acz * abz) / ab_len2))
        closest = (ax + abx * t, ay + aby * t, az + abz * t)
        dist2 = (closest[0] - cx) ** 2 + (closest[1] - cy) ** 2 + (closest[2] - cz) ** 2
        return dist2 <= radius * radius

    def _extract_path(
        self, nodes: List[_Node], goal_idx: int, goal: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        path = [goal]
        idx = goal_idx
        while idx is not None:
            node = nodes[idx]
            path.append(node.pos)
            idx = node.parent
        path.reverse()
        return path
