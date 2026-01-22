from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass
class Obstacle:
    position_w: Tuple[float, float, float]
    velocity_w: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ObstacleFieldConfig:
    max_obstacles: int = 0


class ObstacleField:
    def __init__(self, cfg: ObstacleFieldConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def update(self, dt: float) -> None:
        return None

    def get_obstacles(self) -> List[Obstacle]:
        return []
