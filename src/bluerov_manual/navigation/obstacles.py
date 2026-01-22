from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Obstacle:
    position_w: Tuple[float, float, float]
    radius: float = 0.5
    velocity_w: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ObstacleFieldConfig:
    max_obstacles: int = 0


class ObstacleField:
    def __init__(self, cfg: ObstacleFieldConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._obstacles: List[Obstacle] = []

    def update(self, dt: float) -> None:
        return None

    def update_from_positions(
        self, positions_w: torch.Tensor, radius: float | torch.Tensor = 0.5
    ) -> None:
        self._obstacles = []
        if positions_w.numel() == 0:
            return
        if isinstance(radius, torch.Tensor):
            radii = radius.flatten().tolist()
        else:
            radii = [float(radius)] * positions_w.shape[0]
        for pos, r in zip(positions_w.tolist(), radii):
            self._obstacles.append(Obstacle(tuple(pos), float(r)))

    def get_obstacles(self) -> List[Obstacle]:
        return list(self._obstacles)
