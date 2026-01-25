import os
import omni.isaac.core.utils.prims as prim_utils
import marinegym.utils.kit as kit_utils
from omni.isaac.core.utils.extensions import enable_extension

from marinegym.envs.single.hover import Hover  # IMPORTANT


class HoverTank(Hover):
    """
    MarineGym Hover + tank USD loaded into /World/envs/env_0/tank
    """

    def _design_scene(self):
        # Enable PhysX for raycast
        enable_extension("omni.physx")
        # Build normal Hover scene (target + BlueROV + ground plane)
        global_paths = super()._design_scene()
