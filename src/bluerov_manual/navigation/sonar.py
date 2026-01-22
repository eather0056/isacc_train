from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class SonarConfig:
    num_rays: int = 16
    fov_deg: float = 180.0
    max_range: float = 10.0


class Sonar:
    def __init__(self, cfg: SonarConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def scan(self, pos_w: torch.Tensor, rot_w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Stub scan: returns max-range readings."""
        num_envs = pos_w.shape[0]
        ranges = torch.full(
            (num_envs, self.cfg.num_rays),
            float(self.cfg.max_range),
            device=self.device,
        )
        return {"ranges": ranges}
