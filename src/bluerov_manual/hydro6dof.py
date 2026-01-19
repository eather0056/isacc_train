# src/bluerov_manual/hydro6dof.py
#
# Minimal 6-DOF hydrodynamics + buoyancy model for an underwater vehicle.
#
# This is designed to plug into underwater_vehicle.py in this project.
# It follows the same structure as the MarineGym UnderwaterVehicle hydrodynamics:
#   - Convert flow velocities to body frame
#   - Compute body-relative velocity v (6D) and body acceleration a (6D)
#   - Damping: (Dl + Dq*|v|) v with a "maintained" cross-coupling layout (as in MarineGym)
#   - Added mass: Ma * a
#   - Coriolis-like term from added mass
#   - Buoyancy (force + restoring torque) from roll/pitch and center-of-buoyancy offset
#
# Output is LOCAL/BODY frame:
#   forces_local: (E,1,3)
#   torques_local: (E,1,3)
#
# Notes:
# - This is a pragmatic model to match your MarineGym code, not a full CFD fluid model.
# - Water density defaults to 997 kg/m^3 (freshwater). Change if needed.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch


@dataclass
class HydroParams:
    volume: float
    coBM: float  # center of buoyancy minus center of mass distance (scalar used in MarineGym)
    added_mass: torch.Tensor        # (6,)
    linear_damping: torch.Tensor    # (6,)
    quadratic_damping: torch.Tensor # (6,)
    water_density: float = 997.0
    gravity: float = 9.81


class Hydro6DOF:
    def __init__(self, params: HydroParams, dt: float, device: torch.device):
        self.p = params
        self.dt = float(dt)
        self.device = device

        # Matrices are diagonal in the parameterization used by your MarineGym code
        self.added_mass_matrix = torch.diag(self.p.added_mass.to(device).float())      # (6,6)
        self.linear_damping_matrix = torch.diag(self.p.linear_damping.to(device).float())
        self.quadratic_damping_matrix = torch.diag(self.p.quadratic_damping.to(device).float())

        # State for filtered acceleration (per env)
        self._prev_body_vels: Optional[torch.Tensor] = None  # (E,1,6)
        self._prev_body_acc: Optional[torch.Tensor] = None   # (E,1,6)

    def reset(self, env_ids: torch.Tensor, E: int):
        # lazily init storage
        if self._prev_body_vels is None:
            self._prev_body_vels = torch.zeros(E, 1, 6, device=self.device)
            self._prev_body_acc = torch.zeros(E, 1, 6, device=self.device)
            return
        env_ids = env_ids.to(self.device)
        self._prev_body_vels[env_ids] = 0.0
        self._prev_body_acc[env_ids] = 0.0

    def compute(
        self,
        rot_wxyz: torch.Tensor,     # (E,1,4)
        vel_w: torch.Tensor,        # (E,1,6) world lin+ang
        vel_b: torch.Tensor,        # (E,1,6) body lin+ang (already computed by vehicle)
        masses: torch.Tensor,       # (E,1,1)
        inertias_diag: torch.Tensor,# (E,1,3)  (not directly used here, included for future extensions)
        flow_vels_w: torch.Tensor,  # (E,1,6) world flow lin+ang
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (forces_local, torques_local) each shaped (E,1,3).
        """
        E = vel_b.shape[0]
        if self._prev_body_vels is None:
            self._prev_body_vels = torch.zeros(E, 1, 6, device=self.device)
            self._prev_body_acc = torch.zeros(E, 1, 6, device=self.device)

        # Convert flow velocities to body frame (like MarineGym)
        flow_vels_b = torch.cat(
            [
                quat_rotate_inverse(rot_wxyz, flow_vels_w[..., :3]),
                quat_rotate_inverse(rot_wxyz, flow_vels_w[..., 3:]),
            ],
            dim=-1,
        )

        # Relative body velocity (vehicle velocity relative to water)
        body_vels = vel_b.clone() - flow_vels_b

        # MarineGym sign convention adjustments:
        # body_vels[..., [1,2,4,5]] *= -1
        body_vels = body_vels.clone()
        body_vels[..., [1, 2, 4, 5]] *= -1.0

        # Roll/pitch sign adjustment happens in MarineGym after converting quaternion->euler.
        # We'll compute rpy with same sign convention.
        body_rpy = quaternion_to_euler(rot_wxyz)  # (E,1,3)
        body_rpy = body_rpy.clone()
        body_rpy[..., [1, 2]] *= -1.0  # flip pitch,yaw per MarineGym (they flip [1,2])

        # Acceleration estimate (filtered)
        body_acc = self._calculate_acc(body_vels)

        # Compute terms (squeezed to (E,6) to match MarineGym matrix ops)
        v = body_vels.squeeze(1)
        a = body_acc.squeeze(1)

        damping = self._calculate_damping(v)       # (E,6)
        added_mass = self._calculate_added_mass(a) # (E,6)
        coriolis = self._calculate_corilis(v)      # (E,6)
        buoyancy = self._calculate_buoyancy(body_rpy.squeeze(1), masses.squeeze(1))  # (E,6)

        hydro = -(added_mass + coriolis + damping)

        # Undo sign convention adjustments for output (like MarineGym)
        hydro[:, [1, 2, 4, 5]] *= -1.0
        buoyancy[:, [1, 2, 4, 5]] *= -1.0

        hydro = hydro.unsqueeze(1)
        buoyancy = buoyancy.unsqueeze(1)

        forces_local = hydro[..., 0:3] + buoyancy[..., 0:3]
        torques_local = hydro[..., 3:6] + buoyancy[..., 3:6]
        return forces_local, torques_local

    # ----------------- Internals (mirror MarineGym structure) -----------------

    def _calculate_acc(self, body_vels: torch.Tensor) -> torch.Tensor:
        """
        body_vels: (E,1,6)
        """
        alpha = 0.3
        acc = (body_vels - self._prev_body_vels) / self.dt
        filtered = (1.0 - alpha) * self._prev_body_acc + alpha * acc
        self._prev_body_vels = body_vels.clone()
        self._prev_body_acc = filtered.clone()
        return filtered

    def _calculate_damping(self, body_vels: torch.Tensor) -> torch.Tensor:
        """
        body_vels: (E,6)
        """
        E = body_vels.shape[0]
        # maintained_body_vels = diag_embed(v), with cross-coupling indices like MarineGym
        maintained = torch.diag_embed(body_vels)  # (E,6,6)

        # Cross coupling: same indices as your MarineGym code
        maintained[:, 1, 5] = body_vels[:, 5]
        maintained[:, 2, 4] = body_vels[:, 4]
        maintained[:, 4, 2] = body_vels[:, 2]
        maintained[:, 5, 1] = body_vels[:, 1]

        Dmat = (
            self.linear_damping_matrix.unsqueeze(0)
            + self.quadratic_damping_matrix.unsqueeze(0) * torch.abs(maintained)
        )  # (E,6,6)

        damping = (Dmat @ body_vels.unsqueeze(-1)).squeeze(-1)  # (E,6)
        return damping

    def _calculate_added_mass(self, body_acc: torch.Tensor) -> torch.Tensor:
        """
        body_acc: (E,6)
        """
        return (self.added_mass_matrix.unsqueeze(0) @ body_acc.unsqueeze(-1)).squeeze(-1)

    def _calculate_corilis(self, body_vels: torch.Tensor) -> torch.Tensor:
        """
        body_vels: (E,6)
        """
        ab = (self.added_mass_matrix.unsqueeze(0) @ body_vels.unsqueeze(-1)).squeeze(-1)  # (E,6)

        coriolis = torch.zeros(body_vels.shape[0], 6, device=self.device)
        coriolis[:, 0:3] = -torch.cross(ab[:, 0:3], body_vels[:, 3:6], dim=1)
        coriolis[:, 3:6] = -(
            torch.cross(ab[:, 0:3], body_vels[:, 0:3], dim=1)
            + torch.cross(ab[:, 3:6], body_vels[:, 3:6], dim=1)
        )
        return coriolis

    def _calculate_buoyancy(self, rpy: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """
        rpy:  (E,3) roll,pitch,yaw
        mass: (E,1) or (E,) mass
        Returns (E,6)
        """
        mass = mass.squeeze(-1)
        buoy = torch.zeros(rpy.shape[0], 6, device=self.device)

        # Buoyant force magnitude
        buoyancy_force = self.p.water_density * self.p.gravity * float(self.p.volume)
        dis = float(self.p.coBM)

        roll = rpy[:, 0]
        pitch = rpy[:, 1]

        # Same equations as your MarineGym calculate_buoyancy()
        buoy[:, 0] = buoyancy_force * torch.sin(pitch)
        buoy[:, 1] = -buoyancy_force * torch.sin(roll) * torch.cos(pitch)
        buoy[:, 2] = -buoyancy_force * torch.cos(roll) * torch.cos(pitch)
        buoy[:, 3] = -dis * buoyancy_force * torch.cos(pitch) * torch.sin(roll)
        buoy[:, 4] = -dis * buoyancy_force * torch.sin(pitch)
        # buoy[:, 5] = 0

        return buoy


# ---------------- Quaternion / Euler helpers ----------------

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)

def quat_rotate(q: torch.Tensor, v: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    if inverse:
        q = quat_conjugate(q)
    zeros = torch.zeros_like(v[..., :1])
    vq = torch.cat([zeros, v], dim=-1)
    return quat_mul(quat_mul(q, vq), quat_conjugate(q))[..., 1:]

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(q, v, inverse=True)

def quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (wxyz) to roll/pitch/yaw.
    Input: q shape (...,4)
    Output: (...,3) in radians
    """
    w, x, y, z = q.unbind(-1)

    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack([roll, pitch, yaw], dim=-1)
