from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class SonarConfig:
    num_rays: int = 16
    fov_deg: float = 180.0
    max_range: float = 10.0
    include_angle_encoding: bool = True
    min_range_filter: float = 0.2


class Sonar:
    def __init__(self, cfg: SonarConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        angles = torch.linspace(
            -0.5 * torch.deg2rad(torch.tensor(self.cfg.fov_deg)),
            0.5 * torch.deg2rad(torch.tensor(self.cfg.fov_deg)),
            steps=self.cfg.num_rays,
        )
        self.angles = angles.to(self.device)
        try:
            from omni.physx import get_physx_scene_query_interface  # type: ignore
        except Exception:
            get_physx_scene_query_interface = None
        self._scene_query = (
            get_physx_scene_query_interface() if get_physx_scene_query_interface else None
        )

    def scan(self, pos_w: torch.Tensor, rot_w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """PhysX raycast scan (fallbacks to max-range if unavailable)."""
        num_envs = pos_w.shape[0]
        ranges = torch.full(
            (num_envs, self.cfg.num_rays),
            float(self.cfg.max_range),
            device=self.device,
        )
        if self._scene_query is not None:
            from pxr import Gf
            from marinegym.utils.torch import quat_rotate

            angles = self.angles.to(pos_w.device)
            dir_body = torch.stack(
                [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
            )
            dir_body = dir_body.unsqueeze(0).expand(num_envs, -1, -1)
            rot_expanded = rot_w.unsqueeze(1).expand(-1, self.cfg.num_rays, -1)
            dir_world = quat_rotate(rot_expanded, dir_body).detach()
            origins = pos_w.detach().cpu().tolist()
            dirs = dir_world.cpu().tolist()

            for env_idx in range(num_envs):
                origin = Gf.Vec3f(*origins[env_idx])
                for ray_idx in range(self.cfg.num_rays):
                    direction = Gf.Vec3f(*dirs[env_idx][ray_idx])
                    hit = self._scene_query.raycast_closest(
                        origin, direction, float(self.cfg.max_range)
                    )
                    if hit.get("hit", False):
                        dist = float(hit.get("distance", self.cfg.max_range))
                        if dist >= self.cfg.min_range_filter:
                            ranges[env_idx, ray_idx] = dist
        obs = {"ranges": ranges}
        if self.cfg.include_angle_encoding:
            obs["angles"] = self.angles.expand(num_envs, -1)
            obs["angles_sin"] = torch.sin(obs["angles"])
            obs["angles_cos"] = torch.cos(obs["angles"])
        return obs
