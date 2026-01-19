# src/bluerov_manual/underwater_vehicle.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import yaml

from .t200 import T200
from .hydro6dof import Hydro6DOF, HydroParams


@dataclass
class UnderwaterVehicleAssets:
    usd_path: str
    param_yaml: str


class UnderwaterVehicle:
    def __init__(
        self,
        assets: UnderwaterVehicleAssets,
        device: torch.device,
        num_envs: int,
        envs_prim_paths: list[str],          # /World/envs/env_0, ...
        envs_positions: torch.Tensor,        # (num_envs, 3) on device
        dt: float,
        name: str = "bluerov",
    ):
        self.assets = assets
        self.device = device
        self.num_envs = int(num_envs)
        self.envs_prim_paths = envs_prim_paths
        self.envs_positions = envs_positions
        self.dt = float(dt)
        self.name = name

        # Load params
        with open(self.assets.param_yaml, "r") as f:
            self.params = yaml.safe_load(f)

        rotor_cfg = self.params["rotor_configuration"]
        self.num_rotors = int(rotor_cfg["num_rotors"])

        # Disturbance buffers (num_envs, 6)
        self.flow_vels = torch.zeros(self.num_envs, 6, device=self.device)
        self.max_flow_vel = torch.zeros_like(self.flow_vels)
        self.flow_noise_scale = torch.zeros_like(self.flow_vels)

        # Thrusters + hydro
        self.thrusters = T200(rotor_cfg, dt=self.dt, device=self.device)

        hydro = self.params["hydro_coef"]
        self.hydro = Hydro6DOF(
            HydroParams(
                volume=float(self.params["volume"]),
                coBM=float(self.params["coBM"]),
                added_mass=torch.tensor(hydro["added_mass"], dtype=torch.float32, device=self.device),
                linear_damping=torch.tensor(hydro["linear_damping"], dtype=torch.float32, device=self.device),
                quadratic_damping=torch.tensor(hydro["quadratic_damping"], dtype=torch.float32, device=self.device),
                water_density=997.0,
                gravity=9.81,
            ),
            dt=self.dt,
            device=self.device,
        )

        # Views (created lazily)
        self.base_link = None
        self.rotors_view = None

        # Cached kinematics (num_envs, *)
        self.pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.rot = torch.zeros(self.num_envs, 4, device=self.device)      # wxyz
        self.vel_w = torch.zeros(self.num_envs, 6, device=self.device)    # lin+ang
        self.vel_b = torch.zeros_like(self.vel_w)

        self.heading = torch.zeros(self.num_envs, 3, device=self.device)
        self.up = torch.zeros(self.num_envs, 3, device=self.device)

        # Actions
        self.throttle = torch.zeros(self.num_envs, self.num_rotors, device=self.device)
        self._last_throttle = torch.zeros_like(self.throttle)
        self.throttle_difference = torch.zeros(self.num_envs, device=self.device)

        # Force accumulators
        self._thrusts_local = torch.zeros(self.num_envs, self.num_rotors, 3, device=self.device)
        self._forces_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._torques_local = torch.zeros(self.num_envs, 3, device=self.device)

        # Mass/inertia
        self.masses = torch.zeros(self.num_envs, 1, device=self.device)
        self.inertias_diag = torch.zeros(self.num_envs, 3, device=self.device)

    # ------------------------------------------------------------
    # Isaac/Omni-dependent methods (must be called after app init)
    # ------------------------------------------------------------

    def spawn(self, env_local_translation=(0.0, 0.0, 2.0)) -> None:
        """
        References the robot USD under each env root.
        Call AFTER env cloning created /World/envs/env_i.
        """
        # Lazy import omni AFTER SimulationApp exists
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.prims import XFormPrim

        for i, env_path in enumerate(self.envs_prim_paths):
            robot_root = f"{env_path}/{self.name}"
            if prim_utils.is_prim_path_valid(robot_root):
                continue

            prim_utils.define_prim(robot_root, "Xform")
            prim_utils.add_reference_to_stage(self.assets.usd_path, robot_root)

            # env-local + env offset
            wpos = (torch.tensor(env_local_translation, dtype=torch.float32)
                    + self.envs_positions[i].detach().cpu()).tolist()

            XFormPrim(robot_root).set_world_pose(position=wpos)

    def initialize_views(self) -> None:
        """
        Create batched views for base_link and rotors.
        Must be called AFTER spawn() and AFTER sim.reset().
        """
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.prims import RigidPrimView

        base_expr = f"/World/envs/env_*/{self.name}/base_link"
        rotor_expr = f"/World/envs/env_*/{self.name}/rotor_*"

        if not prim_utils.is_prim_path_valid("/World/envs"):
            raise RuntimeError("Envs root missing. Did you create /World/envs and clone envs?")

        self.base_link = RigidPrimView(
            prim_paths_expr=base_expr,
            name=f"{self.name}_base",
            reset_xform_properties=False,
            shape=(self.num_envs, 1),
        )
        self.base_link.initialize()

        self.rotors_view = RigidPrimView(
            prim_paths_expr=rotor_expr,
            name=f"{self.name}_rotors",
            reset_xform_properties=False,
            shape=(self.num_envs, self.num_rotors),
        )
        self.rotors_view.initialize()

        masses = self.base_link.get_masses().reshape(self.num_envs, 1)
        self.masses[:] = torch.as_tensor(masses, dtype=torch.float32, device=self.device)

        inertias = self.base_link.get_inertias().reshape(self.num_envs, 3, 3)
        self.inertias_diag[:] = inertias.diagonal(0, -2, -1)

    # ------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(self.device)

        self._forces_local[env_ids] = 0
        self._torques_local[env_ids] = 0
        self._thrusts_local[env_ids] = 0

        self.throttle[env_ids] = 0
        self._last_throttle[env_ids] = 0
        self.throttle_difference[env_ids] = 0

        # Flow init
        self.flow_vels[env_ids] = torch.rand_like(self.flow_vels[env_ids]) * self.max_flow_vel[env_ids]

        self.thrusters.reset(env_ids)

    def set_flow_velocities(self, env_ids: torch.Tensor, max_flow_velocity, flow_noise) -> None:
        env_ids = env_ids.to(self.device)
        self.max_flow_vel[env_ids] = torch.tensor(max_flow_velocity, dtype=torch.float32, device=self.device)
        self.flow_noise_scale[env_ids] = torch.tensor(flow_noise, dtype=torch.float32, device=self.device)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: (num_envs, 1, num_rotors) or (num_envs, num_rotors)
        """
        if self.base_link is None or self.rotors_view is None:
            raise RuntimeError("Call spawn() + initialize_views() before apply_action().")

        actions = actions.to(self.device)
        if actions.ndim == 3:
            actions = actions[:, 0, :]
        if actions.shape[-1] != self.num_rotors:
            raise ValueError(f"Expected {self.num_rotors} rotors, got {actions.shape[-1]}")

        self._last_throttle[:] = self.throttle

        thrusts, self.throttle = self.thrusters.step(actions, self.throttle)  # (N, R)

        # local thrust along +X
        self._thrusts_local.zero_()
        self._thrusts_local[..., 0] = thrusts

        # hydro wrench
        self._update_kinematics()

        flow = self.flow_vels + torch.randn_like(self.flow_vels) * self.flow_noise_scale
        hydro_f_local, hydro_tau_local = self.hydro.compute(
            rot_wxyz=self.rot,
            vel_w=self.vel_w,
            vel_b=self.vel_b,
            masses=self.masses,
            inertias_diag=self.inertias_diag,
            flow_vels_w=flow,
        )
        self._forces_local[:] = hydro_f_local
        self._torques_local[:] = hydro_tau_local

        # Apply forces:
        # rotors count = num_envs * num_rotors
        self.rotors_view.apply_forces_and_torques_at_pos(
            forces=self._thrusts_local.reshape(-1, 3),
            torques=None,
            positions=None,
            is_global=False,
        )

        # base_link count = num_envs
        self.base_link.apply_forces_and_torques_at_pos(
            forces=self._forces_local.reshape(-1, 3),
            torques=self._torques_local.reshape(-1, 3),
            positions=None,
            is_global=False,
        )

        self.throttle_difference[:] = torch.norm(self.throttle - self._last_throttle, dim=-1)

        effort = torch.sum(torch.abs(self.throttle), dim=-1)  # (N,)
        return effort

    def get_state(self, env_frame: bool = True) -> torch.Tensor:
        """
        Returns (num_envs, 1, 19+num_rotors)
        """
        self._update_kinematics()

        pos = self.pos
        if env_frame:
            pos = pos - self.envs_positions

        throttle_scaled = self.throttle * 2.0 - 1.0

        state = torch.cat(
            [
                pos,                       # 3
                self.rot,                  # 4
                self.vel_w,                # 6
                self.heading,              # 3
                self.up,                   # 3
                throttle_scaled,           # R
            ],
            dim=-1,
        )
        return state.unsqueeze(1)

    # ------------------------------------------------------------

    def _update_kinematics(self) -> None:
        pos, rot = self.base_link.get_world_poses(clone=False)
        vel = self.base_link.get_velocities(clone=False)

        self.pos[:] = torch.as_tensor(pos, dtype=torch.float32, device=self.device).reshape(self.num_envs, 3)
        self.rot[:] = torch.as_tensor(rot, dtype=torch.float32, device=self.device).reshape(self.num_envs, 4)
        self.vel_w[:] = torch.as_tensor(vel, dtype=torch.float32, device=self.device).reshape(self.num_envs, 6)

        # Body velocities
        self.vel_b[:] = quat_rotate_inverse_wxyz(self.rot, self.vel_w)

        # Heading/up
        self.heading[:] = quat_axis_wxyz(self.rot, axis=0)
        self.up[:] = quat_axis_wxyz(self.rot, axis=2)


# ------------------ minimal quaternion helpers ------------------

def quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )

def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    if inverse:
        q = quat_conjugate_wxyz(q)
    zeros = torch.zeros_like(v[..., :1])
    vq = torch.cat([zeros, v], dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(q, vq), quat_conjugate_wxyz(q))[..., 1:]

def quat_rotate_inverse_wxyz(q: torch.Tensor, v6: torch.Tensor) -> torch.Tensor:
    v_lin = v6[..., :3]
    v_ang = v6[..., 3:]
    v_lin_b = quat_rotate_wxyz(q, v_lin, inverse=True)
    v_ang_b = quat_rotate_wxyz(q, v_ang, inverse=True)
    return torch.cat([v_lin_b, v_ang_b], dim=-1)

def quat_axis_wxyz(q: torch.Tensor, axis: int) -> torch.Tensor:
    basis = torch.zeros(q.shape[0], 3, device=q.device, dtype=q.dtype)
    basis[:, axis] = 1.0
    return quat_rotate_wxyz(q, basis, inverse=False)
