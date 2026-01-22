from dataclasses import dataclass
from typing import Dict

import torch

from .local_policy import BehaviorAction


@dataclass
class BehaviorFusionConfig:
    slow_scale: float = 0.3


class BehaviorFusion:
    def __init__(self, cfg: BehaviorFusionConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def to_local_target(
        self,
        behavior: torch.Tensor,
        waypoint_w: torch.Tensor,
        pos_w: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Convert a behavior into a local target (world-frame position)."""
        target_pos = waypoint_w.clone()
        return {"target_pos_w": target_pos, "behavior": behavior}
