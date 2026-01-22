from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class GlobalPlannerConfig:
    grid_resolution: float = 0.5
    fixed_depth: float | None = None


class GlobalPlanner:
    def __init__(self, cfg: GlobalPlannerConfig):
        self.cfg = cfg

    def plan(
        self,
        start_w: Sequence[float],
        goal_w: Sequence[float],
        obstacles: Sequence[object] | None = None,
    ) -> List[Tuple[float, float, float]]:
        """Return a coarse list of waypoints in world frame."""
        if self.cfg.fixed_depth is not None:
            goal = (float(goal_w[0]), float(goal_w[1]), float(self.cfg.fixed_depth))
        else:
            goal = (float(goal_w[0]), float(goal_w[1]), float(goal_w[2]))
        return [goal]
