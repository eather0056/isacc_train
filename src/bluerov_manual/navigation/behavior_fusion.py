from dataclasses import dataclass
from typing import Dict

import torch

from .local_policy import BehaviorAction


@dataclass
class BehaviorFusionConfig:
    slow_scale: float = 0.3
    avoidance_offset: float = 1.0
    max_speed: float = 0.3
    max_vert_speed: float = 0.1


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
        direction = waypoint_w - pos_w
        norm = torch.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
        dir_unit = direction / norm
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(dir_unit)
        left = torch.cross(up, dir_unit, dim=-1)

        is_left = behavior == int(BehaviorAction.AVOID_LEFT)
        is_right = behavior == int(BehaviorAction.AVOID_RIGHT)
        target_pos = torch.where(
            is_left.unsqueeze(-1),
            target_pos + left * self.cfg.avoidance_offset,
            target_pos,
        )
        target_pos = torch.where(
            is_right.unsqueeze(-1),
            target_pos - left * self.cfg.avoidance_offset,
            target_pos,
        )

        # Compute direction to the (possibly modified) target_pos
        target_direction = target_pos - pos_w
        target_norm = torch.norm(target_direction, dim=-1, keepdim=True).clamp_min(1e-6)
        target_dir_unit = target_direction / target_norm

        target_vel = target_dir_unit * self.cfg.max_speed
        target_vel = torch.where(
            (behavior == int(BehaviorAction.SLOW)).unsqueeze(-1),
            target_vel * self.cfg.slow_scale,
            target_vel,
        )
        target_vel = torch.where(
            (behavior == int(BehaviorAction.STOP)).unsqueeze(-1),
            torch.zeros_like(target_vel),
            target_vel,
        )
        target_vel[..., 2] = torch.clamp(
            target_vel[..., 2], -self.cfg.max_vert_speed, self.cfg.max_vert_speed
        )
        target_pos = torch.where(
            (behavior == int(BehaviorAction.STOP)).unsqueeze(-1),
            pos_w,
            target_pos,
        )
        target_yaw = torch.atan2(target_dir_unit[..., 1], target_dir_unit[..., 0])
        return {
            "target_pos_w": target_pos,
            "target_vel_w": target_vel,
            "target_yaw": target_yaw,
            "behavior": behavior,
        }
