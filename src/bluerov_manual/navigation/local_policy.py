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
    min_range_replan: float = 0.8
    min_range_avoid: float = 1.2


class LocalPolicy:
    def __init__(self, cfg: LocalPolicyConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def decide(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-env discrete behavior (rule-based stub)."""
        ranges = obs["ranges"]
        num_envs, num_rays = ranges.shape
        min_ranges, min_idx = torch.min(ranges, dim=1)

        behavior = torch.full(
            (num_envs,), int(BehaviorAction.FOLLOW_PATH), device=self.device
        )
        behavior = torch.where(
            min_ranges < self.cfg.min_range_replan,
            torch.tensor(int(BehaviorAction.REQUEST_REPLAN), device=self.device),
            behavior,
        )
        behavior = torch.where(
            min_ranges < self.cfg.min_range_avoid,
            torch.tensor(int(BehaviorAction.AVOID_LEFT), device=self.device),
            behavior,
        )
        behavior = torch.where(
            min_ranges < self.cfg.min_range_stop,
            torch.tensor(int(BehaviorAction.STOP), device=self.device),
            behavior,
        )

        left_mask = min_idx < (num_rays // 2)
        behavior = torch.where(
            (behavior == int(BehaviorAction.AVOID_LEFT)) & left_mask,
            torch.tensor(int(BehaviorAction.AVOID_RIGHT), device=self.device),
            behavior,
        )
        return behavior
