# src/bluerov_manual/t200.py
#
# Thruster dynamics model for your project, based on the MarineGym T200 implementation you pasted.
# This version is written to work in a batched environment:
#   - actions:   (num_envs, 1, num_rotors) in [-1, 1]
#   - throttle:  (num_envs, 1, num_rotors) internal state
#   - rpm:       (num_envs, 1, num_rotors) internal state
#
# It returns:
#   - thrusts:   (num_envs, 1, num_rotors)   (Newtons, along local rotor axis)
#   - moments:   (num_envs, 1, num_rotors)   (Nm, reaction torque scalar)
#   - throttle:  updated throttle state

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch


class T200:
    """
    Batched T200 thruster model derived from the MarineGym file you provided.

    Differences vs MarineGym:
    - No nn.Module / Parameters required (manual control, no training of thruster params).
    - Fully batched: maintains per-env throttle and rpm states.
    """

    def __init__(self, rotor_config: dict, dt: float, device: torch.device):
        self.device = device
        self.dt = float(dt)

        # Rotor configuration constants (shape: [num_rotors])
        self.force_constants = torch.as_tensor(rotor_config["force_constants"], dtype=torch.float32, device=device)
        self.moment_constants = torch.as_tensor(rotor_config["moment_constants"], dtype=torch.float32, device=device)
        self.max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"], dtype=torch.float32, device=device)
        self.directions = torch.as_tensor(rotor_config["directions"], dtype=torch.float32, device=device)
        self.time_constants = torch.as_tensor(rotor_config["time_constants"], dtype=torch.float32, device=device)

        self.num_rotors = int(self.force_constants.numel())

        # Model hyperparameters copied from MarineGym T200
        self.noise_scale = 0.002
        self.tau_up = 0.43
        self.tau_down = 0.43

        # RPM clamp uses the config max rotation velocity (assume symmetric)
        # If config provides per-rotor max, we clamp per-rotor.
        self._rpm_min = -self.max_rot_vels
        self._rpm_max = self.max_rot_vels

    def reset(self, env_ids: torch.Tensor, throttle: Optional[torch.Tensor] = None, rpm: Optional[torch.Tensor] = None):
        """
        Reset internal states for specific env ids. If you keep throttle/rpm externally,
        pass them and we will zero them for env_ids.
        """
        if throttle is not None:
            throttle[env_ids] = 0.0
        if rpm is not None:
            rpm[env_ids] = 0.0

    def step(
        self,
        cmds: torch.Tensor,
        throttle: torch.Tensor,
        rpm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        cmds:     (E,1,R) in [-1,1]   (manual thruster command)
        throttle: (E,1,R) internal
        rpm:      (E,1,R) internal (if None, a zeros tensor is created each call; prefer passing persistent rpm)

        Returns:
          thrusts: (E,1,R)
          moments: (E,1,R)
          throttle: updated throttle
          rpm: updated rpm
        """
        cmds = cmds.to(self.device)
        throttle = throttle.to(self.device)

        if rpm is None:
            rpm = torch.zeros_like(throttle)

        # Clamp input
        target_throttle = torch.clamp(cmds, -1.0, 1.0)

        # First-order throttle update (tau_up/down depends on direction of change)
        tau = torch.where(target_throttle > throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0.0, 1.0)
        throttle = throttle + tau * (target_throttle - throttle)

        # Compute target RPM from throttle using the same piecewise fit you pasted
        # (these constants come from your MarineGym T200 version)
        target_rpm = torch.where(
            throttle > 0.075,
            3.6599e3 * throttle + 3.4521e2,
            torch.where(
                throttle < -0.075,
                3.4944e3 * throttle - 4.3350e2,
                torch.zeros_like(throttle),
            ),
        )

        # RPM first-order response with time constant per rotor
        # alpha = exp(-dt / time_constant)
        # time_constants shape [R] -> broadcast to (E,1,R)
        alpha = torch.exp(-self.dt / self.time_constants).view(1, 1, -1)

        # Noise is disabled in your pasted code (multiplied by 0.), keep same behavior.
        noise = torch.randn_like(rpm) * self.noise_scale * 0.0

        rpm = alpha * rpm + (1.0 - alpha) * target_rpm
        rpm = torch.clamp(rpm + noise, self._rpm_min.view(1, 1, -1), self._rpm_max.view(1, 1, -1))

        # Convert RPM -> thrust using the same polynomial you pasted
        # NOTE: This expression is copied literally, including the scaling by force_constants/4.4e-7 * 9.81
        # rpm2force fit is asymmetric for positive/negative rpm.
        rpm2 = rpm * rpm
        thrusts = (
            (self.force_constants / 4.4e-7) * 9.81
            * torch.where(
                rpm > 0,
                4.7368e-07 * rpm2 - 1.9275e-04 * rpm + 8.4452e-02,
                -3.8442e-07 * rpm2 - 1.6186e-04 * rpm - 3.9139e-02,
            )
        )

        # Reaction torque (moments) - your pasted code multiplies by 0, so it's always zero.
        # Keep that behavior for consistency.
        moments = thrusts * (-self.directions.view(1, 1, -1)) * 0.0

        return thrusts, moments, throttle, rpm
