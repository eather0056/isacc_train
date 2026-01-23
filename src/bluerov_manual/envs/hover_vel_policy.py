import torch

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict

from marinegym.envs.isaac_env import AgentSpec
from marinegym.envs.single.hover import Hover
from marinegym.utils.torch import quat_rotate_inverse, quaternion_to_euler
from bluerov_manual.navigation import Sonar, SonarConfig


class HoverVel(Hover):
    """Hover task with velocity + yaw-rate actions (4D)."""

    def __init__(self, cfg, headless):
        self.ctrl_cfg = cfg.task.get("controller", {}) or {}
        self.sonar_cfg = cfg.task.get("sonar", {}) or {}
        super().__init__(cfg, headless)
        self.sonar = Sonar(
            SonarConfig(
                num_rays=int(self.sonar_cfg.get("num_rays", 32)),
                fov_deg=float(self.sonar_cfg.get("fov_deg", 180.0)),
                max_range=float(self.sonar_cfg.get("max_range", 10.0)),
                include_angle_encoding=bool(self.sonar_cfg.get("include_angle_encoding", False)),
                min_range_filter=float(self.sonar_cfg.get("min_range_filter", 0.2)),
            ),
            device=self.device,
        )

        self.u_max = float(self.ctrl_cfg.get("u_max", 0.6))
        self.v_max = float(self.ctrl_cfg.get("v_max", 0.4))
        self.w_max = float(self.ctrl_cfg.get("w_max", 0.3))
        self.r_max = float(self.ctrl_cfg.get("r_max", 0.6))
        self.vel_kp = float(self.ctrl_cfg.get("vel_kp", 5.0))
        self.yaw_rate_kp = float(self.ctrl_cfg.get("yaw_rate_kp", 1.5))
        self.stabilize_rp = bool(self.ctrl_cfg.get("stabilize_rp", True))
        self.rp_kp = float(self.ctrl_cfg.get("rp_kp", 2.0))
        self.rp_kd = float(self.ctrl_cfg.get("rp_kd", 0.2))
        self.max_accel = self.ctrl_cfg.get("max_accel", None)
        self.max_force = self.ctrl_cfg.get("max_force", None)
        self.max_torque = self.ctrl_cfg.get("max_torque", None)

        self.thrust_axis = int(self.ctrl_cfg.get("thrust_axis", 0))
        self.thrust_sign = float(self.ctrl_cfg.get("thrust_sign", 1.0))
        self.alloc_update_every = int(self.ctrl_cfg.get("allocation_update_every", 1))
        self.action_smooth_alpha = float(self.ctrl_cfg.get("action_smooth_alpha", 0.8))
        self._alloc_step = 0
        self._A_pinv = None
        self._prev_action = None

    def _build_allocation(self):
        from marinegym.utils.torch import quat_axis

        rotor_pos_w, rotor_rot_w = self.drone.rotors_view.get_world_poses()
        r_w = rotor_pos_w[:, 0]
        q_w = rotor_rot_w[:, 0]

        base_pos_w, base_rot_w = self.drone.base_link.get_world_poses()
        base_pos_w = base_pos_w[:, 0]
        base_rot_w = base_rot_w[:, 0]

        d_w = quat_axis(q_w, axis=self.thrust_axis) * self.thrust_sign
        r_rel_w = r_w - base_pos_w.unsqueeze(1)

        base_q = base_rot_w.unsqueeze(1).expand_as(q_w)
        r_b = quat_rotate_inverse(base_q, r_rel_w)
        d_b = quat_rotate_inverse(base_q, d_w)
        tau_b = torch.cross(r_b, d_b, dim=-1)

        A = torch.zeros(self.num_envs, 6, r_b.shape[1], device=self.device)
        A[:, 0:3, :] = d_b.transpose(1, 2)
        A[:, 3:6, :] = tau_b.transpose(1, 2)
        self._A_pinv = torch.linalg.pinv(A)

    def _set_specs(self):
        num_rays = int(self.sonar_cfg.get("num_rays", 32))
        obs_dim = 3 + 2 + 3 + 1 + 1 + 2 + num_rays
        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec((1, obs_dim), device=self.device),
                        "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "action": BoundedTensorSpec(-1.0, 1.0, (1, 4), device=self.device),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec(
            {"agents": CompositeSpec({"reward": UnboundedContinuousTensorSpec((1, 1))})}
        ).expand(self.num_envs).to(self.device)

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

    def _pre_sim_step(self, tensordict):
        self._update_obstacles()
        self.drone.get_state()
        actions = tensordict[("agents", "action")]
        if self._prev_action is None:
            self._prev_action = torch.zeros_like(actions)
        actions = self.action_smooth_alpha * self._prev_action + (1.0 - self.action_smooth_alpha) * actions
        self._prev_action = actions

        a = actions.squeeze(1)
        u_cmd = torch.clamp(a[:, 0], -1.0, 1.0) * self.u_max
        v_cmd = torch.clamp(a[:, 1], -1.0, 1.0) * self.v_max
        w_cmd = torch.clamp(a[:, 2], -1.0, 1.0) * self.w_max
        r_cmd = torch.clamp(a[:, 3], -1.0, 1.0) * self.r_max

        v_cmd_b = torch.stack([u_cmd, v_cmd, w_cmd], dim=-1)
        vel_b = self.drone.vel_b[:, 0, :3]
        ang_b = self.drone.vel_b[:, 0, 3:6]
        e_v = v_cmd_b - vel_b
        F_b = self.vel_kp * e_v
        T_b = torch.zeros(self.num_envs, 3, device=self.device)
        T_b[:, 2] = self.yaw_rate_kp * (r_cmd - ang_b[:, 2])
        if self.stabilize_rp:
            rpy = quaternion_to_euler(self.drone.rot[:, 0, :])
            T_b[:, 0] = -self.rp_kp * rpy[:, 0] - self.rp_kd * ang_b[:, 0]
            T_b[:, 1] = -self.rp_kp * rpy[:, 1] - self.rp_kd * ang_b[:, 1]

        if self.max_accel is not None:
            masses = self.drone.masses[:, 0].squeeze(-1)
            max_force = masses * float(self.max_accel)
        else:
            max_force = self.max_force
        if max_force is not None:
            max_force_val = torch.as_tensor(max_force, device=self.device)
            force_norm = torch.norm(F_b, dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(max_force_val / force_norm.squeeze(-1), max=1.0).unsqueeze(-1)
            F_b = F_b * scale
        if self.max_torque is not None:
            max_torque_val = float(self.max_torque)
            torque_norm = torch.norm(T_b, dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(max_torque_val / torque_norm.squeeze(-1), max=1.0).unsqueeze(-1)
            T_b = T_b * scale

        if self._A_pinv is None or (self.alloc_update_every > 0 and self._alloc_step % self.alloc_update_every == 0):
            self._build_allocation()
        self._alloc_step += 1

        wrench = torch.cat([F_b, T_b], dim=-1).unsqueeze(1)
        forces = wrench @ self._A_pinv.transpose(-1, -2)
        f_max = float(self.drone.FORCE_CONSTANTS_0[0].item()) * (3900.0 ** 2)
        thr_action = torch.clamp(forces / f_max, -1.0, 1.0)
        self.effort = torch.abs(self.drone.apply_action(thr_action))

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        pos_w = self.drone.pos[:, 0, :]
        rot_w = self.drone.rot[:, 0, :]
        vel_b = self.drone.vel_b[:, 0, :3]
        ang_b = self.drone.vel_b[:, 0, 3:6]

        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        goal_w = self.target_pos.expand(self.num_envs, 1, 3)[:, 0, :]
        rel_w = goal_w - pos_w
        rel_b = quat_rotate_inverse(rot_w, rel_w)

        yaw_goal = torch.atan2(rel_w[:, 1], rel_w[:, 0])
        yaw = quaternion_to_euler(rot_w)[:, 2]
        yaw_err = (yaw_goal - yaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_sin = torch.sin(yaw_err).unsqueeze(-1)
        yaw_cos = torch.cos(yaw_err).unsqueeze(-1)

        rpy = quaternion_to_euler(rot_w)
        roll_pitch = rpy[:, :2]
        depth = pos_w[:, 2:3]

        sonar_obs = self.sonar.scan(pos_w, rot_w)["ranges"]

        obs = torch.cat(
            [
                rel_b,
                yaw_sin,
                yaw_cos,
                vel_b,
                ang_b[:, 2:3],
                depth,
                roll_pitch,
                sonar_obs,
            ],
            dim=-1,
        ).unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )
