from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import torch


class BehaviorAction(IntEnum):
    FOLLOW_PATH = 0
    AVOID_LEFT = 1
    AVOID_RIGHT = 2
    SLOW = 3
    STOP = 4
    REQUEST_REPLAN = 5


@dataclass
class LocalPolicyConfig:
    min_range_stop: float = 0.5


class LocalPolicy:
    def __init__(self, cfg: LocalPolicyConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def decide(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-env discrete behavior."""
        num_envs = obs["ranges"].shape[0]
        return torch.full((num_envs,), int(BehaviorAction.FOLLOW_PATH), device=self.device)
