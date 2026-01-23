import os
import omni.isaac.core.utils.prims as prim_utils
import marinegym.utils.kit as kit_utils

from marinegym.envs.single.hover import Hover  # IMPORTANT


class HoverTank(Hover):
    """
    MarineGym Hover + tank USD loaded into /World/envs/env_0/tank
    """

    def _design_scene(self):
        # Build normal Hover scene (target + BlueROV + ground plane)
        global_paths = super()._design_scene()

        # Read tank path from cfg.robot.tank_usd_path (your config layout)
        if not hasattr(self.cfg, "task") or "tank_usd_path" not in self.cfg.task:
            raise RuntimeError("Missing cfg.task.tank_usd_path in your Hydra config.")
        tank_usd = os.path.abspath(os.path.expanduser(str(self.cfg.task.tank_usd_path)))

        if not os.path.exists(tank_usd):
            raise FileNotFoundError(f"Tank USD not found: {tank_usd}")

        tank_prim_path = "/World/envs/env_0/tank"

        if not prim_utils.is_prim_path_valid(tank_prim_path):
            tank_prim = prim_utils.create_prim(
                prim_path=tank_prim_path,
                usd_path=tank_usd,
                translation=(0.0, 0.0, 0.0),
            )

            # Static + collidable
            kit_utils.set_nested_collision_properties(
                tank_prim.GetPath(),
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            )
            kit_utils.set_nested_rigid_body_properties(
                tank_prim.GetPath(),
                disable_gravity=True,
                rigid_body_enabled=True,
            )

        return global_paths
