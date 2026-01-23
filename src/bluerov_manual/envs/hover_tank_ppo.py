import os
import torch

import omni.isaac.core.utils.prims as prim_utils
import marinegym.utils.kit as kit_utils

from tensordict.tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec

from marinegym.envs.isaac_env import AgentSpec
from marinegym.utils.torch import quat_rotate_inverse, quaternion_to_euler, quat_axis, euler_to_quaternion
from bluerov_manual.navigation import BehaviorFusion, BehaviorFusionConfig, GlobalPlanner, GlobalPlannerConfig, Sonar, SonarConfig
from bluerov_manual.navigation.obstacles import Obstacle

from .hover_tank import HoverTank


class HoverTankPPO(HoverTank):
    """HoverTank with PPO behavior actions (scalar -> discrete behavior)."""

    def __init__(self, cfg, headless):
        self.ctrl_cfg = cfg.task.get("controller", {}) or {}
        planner_stack = cfg.task.get("planner_stack", {}) or {}
        if planner_stack.get("enable", False):
            self.sonar_cfg = planner_stack.get("sonar", {}) or {}
            self.planner_cfg = planner_stack.get("rrt", {}) or {}
            self.fusion_cfg = planner_stack.get("fusion", {}) or {}
        else:
            self.sonar_cfg = cfg.task.get("sonar", {}) or {}
            self.planner_cfg = cfg.task.get("planner", {}) or {}
            self.fusion_cfg = self.planner_cfg
        self.n_behaviors = int(cfg.task.get("n_behaviors", self.planner_cfg.get("n_behaviors", 6)))
        self.success_distance = float(cfg.task.get("success_distance", 0.5))
        self.success_bonus = float(cfg.task.get("success_bonus", 5.0))
        self.reward_goal_progress = float(cfg.task.get("reward_goal_progress", 1.0))
        self.obstacle_penalty = float(cfg.task.get("obstacle_penalty", 0.5))
        self.obstacle_threshold = float(cfg.task.get("obstacle_threshold", 1.0))
        self.debug = bool(cfg.task.get("debug", False))
        self.debug_interval = int(cfg.task.get("debug_interval", 50))
        super().__init__(cfg, headless)

        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, 1, device=self.device)
        self._last_sonar = None
        self._last_behavior = None
        self._last_target_vel_w = None
        self._waypoint_tol = float(self.planner_cfg.get("waypoint_tol", 0.5))
        goal_spawn = cfg.task.get("goal_spawn_ranges", cfg.task.get("spawn_ranges", {})) or {}
        if goal_spawn:
            def _axis_range(key, default):
                raw = goal_spawn.get(key, default)
                lo, hi = float(raw[0]), float(raw[1])
                return min(lo, hi), max(lo, hi)

            gx_min, gx_max = _axis_range("x_range", (-2.5, 2.5))
            gy_min, gy_max = _axis_range("y_range", (-2.5, 2.5))
            gz_min, gz_max = _axis_range("z_range", (1.5, 2.5))
            self.goal_spawn_low = torch.tensor([gx_min, gy_min, gz_min], device=self.device)
            self.goal_spawn_high = torch.tensor([gx_max, gy_max, gz_max], device=self.device)
        else:
            self.goal_spawn_low = None
            self.goal_spawn_high = None

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
        self.planner = GlobalPlanner(
            GlobalPlannerConfig(
                bounds_min=tuple(self.planner_cfg.get("bounds_min", [-10.0, -10.0, -5.0])),
                bounds_max=tuple(self.planner_cfg.get("bounds_max", [10.0, 10.0, 5.0])),
                step_size=float(self.planner_cfg.get("step_size", 1.0)),
                max_iters=int(self.planner_cfg.get("max_iters", 400)),
                goal_sample_rate=float(self.planner_cfg.get("goal_sample_rate", 0.1)),
                neighbor_radius=float(self.planner_cfg.get("neighbor_radius", 2.0)),
                goal_tolerance=float(self.planner_cfg.get("goal_tolerance", 1.0)),
            )
        )
        self.behavior_fusion = BehaviorFusion(
            BehaviorFusionConfig(
                slow_scale=float(self.fusion_cfg.get("slow_scale", 0.3)),
                avoidance_offset=float(self.fusion_cfg.get("avoidance_offset", 1.0)),
                max_speed=float(self.fusion_cfg.get("max_speed", 0.3)),
                max_vert_speed=float(self.fusion_cfg.get("max_vert_speed", 0.1)),
            ),
            device=self.device,
        )

        self._paths = [[] for _ in range(self.num_envs)]
        self._path_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._A_pinv = None
        self._prev_action = None
        self._alloc_step = 0

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
                        "action": DiscreteTensorSpec(self.n_behaviors, shape=(1, 1), device=self.device),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec(
            {"agents": CompositeSpec({"reward": UnboundedContinuousTensorSpec((1, 1))})}
        ).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )
        stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "pos_error": UnboundedContinuousTensorSpec(1),
                "heading_alignment": UnboundedContinuousTensorSpec(1),
                "uprightness": UnboundedContinuousTensorSpec(1),
                "action_smoothness": UnboundedContinuousTensorSpec(1),
            }
        ).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self.drone.get_state()
        if self.goal_spawn_low is not None:
            rand = torch.rand(len(env_ids), 3, device=self.device)
            goal_w = self.goal_spawn_low + rand * (self.goal_spawn_high - self.goal_spawn_low)
            self.target_pos[env_ids, 0, :] = goal_w
            goal_world = goal_w + self.envs_positions[env_ids]
            target_rot = euler_to_quaternion(torch.zeros(len(env_ids), 1, 3, device=self.device))
            self.target_vis.set_world_poses(positions=goal_world, orientations=target_rot, env_indices=env_ids)
            rel = goal_w - self.drone.pos[env_ids, 0, :]
            rel_norm = torch.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
            self.target_heading[env_ids] = (rel / rel_norm).unsqueeze(1)
            self._prev_goal_dist[env_ids] = torch.norm(rel, dim=-1, keepdim=True)
        if self.debug:
            start_pos = self.drone.pos[env_ids, 0, :].detach().cpu().numpy()
            goals = self.target_pos[env_ids, 0, :].detach().cpu().numpy()
            print(f"[HoverTankPPO] reset env_ids={env_ids.tolist()} start_pos={start_pos} goals={goals}")
        self._plan_paths(env_ids)
        if self._prev_action is None:
            self._prev_action = torch.zeros(self.num_envs, 1, 1, device=self.device)

    def _plan_paths(self, env_ids: torch.Tensor):
        pos_w = self.drone.pos[env_ids, 0, :]
        for i, env_id in enumerate(env_ids.tolist()):
            goal_w = self.target_pos[env_id, 0, :].tolist()
            obstacles = []
            if self.enable_obstacles:
                radius = float(self.obst_cfg.get("radius", 0.15))
                for pos in self.obst_center[env_id].tolist():
                    obstacles.append(Obstacle(tuple(pos), radius))
            plan = self.planner.plan(
                start_w=pos_w[i].tolist(),
                goal_w=goal_w,
                obstacles=obstacles,
            )
            self._paths[env_id] = plan
            self._path_idx[env_id] = 0

    def _behavior_from_action(self, action: torch.Tensor) -> torch.Tensor:
        # Discrete action -> behavior index
        return action.long().clamp(0, self.n_behaviors - 1)

    def _build_allocation(self):
        rotor_pos_w, rotor_rot_w = self.drone.rotors_view.get_world_poses()
        r_w = rotor_pos_w[:, 0]
        q_w = rotor_rot_w[:, 0]

        base_pos_w, base_rot_w = self.drone.base_link.get_world_poses()
        base_pos_w = base_pos_w[:, 0]
        base_rot_w = base_rot_w[:, 0]

        thrust_axis = int(self.ctrl_cfg.get("thrust_axis", 0))
        thrust_sign = float(self.ctrl_cfg.get("thrust_sign", 1.0))
        d_w = quat_axis(q_w, axis=thrust_axis) * thrust_sign
        r_rel_w = r_w - base_pos_w.unsqueeze(1)
        base_q = base_rot_w.unsqueeze(1).expand_as(q_w)
        r_b = quat_rotate_inverse(base_q, r_rel_w)
        d_b = quat_rotate_inverse(base_q, d_w)
        tau_b = torch.cross(r_b, d_b, dim=-1)

        A = torch.zeros(self.num_envs, 6, r_b.shape[1], device=self.device)
        A[:, 0:3, :] = d_b.transpose(1, 2)
        A[:, 3:6, :] = tau_b.transpose(1, 2)
        self._A_pinv = torch.linalg.pinv(A)

    def _pre_sim_step(self, tensordict):
        self._update_obstacles()
        self.drone.get_state()

        actions = tensordict[("agents", "action")]
        if self._prev_action is None:
            self._prev_action = torch.zeros_like(actions)
        alpha = float(self.ctrl_cfg.get("action_smooth_alpha", 0.8))
        actions = alpha * self._prev_action + (1.0 - alpha) * actions
        self._prev_action = actions

        behavior = self._behavior_from_action(actions.squeeze(1).squeeze(-1))
        self._last_behavior = behavior
        if behavior.eq(self.n_behaviors - 1).any():
            env_ids = torch.where(behavior.eq(self.n_behaviors - 1))[0]
            self._plan_paths(env_ids)

        waypoints = []
        for env_idx in range(self.num_envs):
            idx = int(self._path_idx[env_idx].item())
            path = self._paths[env_idx]
            idx = min(idx, len(path) - 1) if path else 0
            waypoints.append(path[idx] if path else self.target_pos[0].tolist())
        waypoint_w = torch.tensor(waypoints, device=self.device)
        dist_to_wp = torch.norm(waypoint_w - self.drone.pos[:, 0, :], dim=-1)
        reached = dist_to_wp < self._waypoint_tol
        if reached.any():
            for env_idx in torch.where(reached)[0].tolist():
                new_idx = min(
                    int(self._path_idx[env_idx].item()) + 1,
                    max(len(self._paths[env_idx]) - 1, 0),
                )
                self._path_idx[env_idx] = new_idx

        local_cmd = self.behavior_fusion.to_local_target(behavior, waypoint_w, self.drone.pos[:, 0, :])
        target_vel_w = local_cmd["target_vel_w"]
        goal_w = self.target_pos[:, 0, :]
        dz = goal_w[:, 2] - self.drone.pos[:, 0, 2]
        max_vz = float(self.fusion_cfg.get("max_vert_speed", self.planner_cfg.get("max_vert_speed", 0.1)))
        target_vel_w[:, 2] = torch.clamp(dz, -max_vz, max_vz)
        self._last_target_vel_w = target_vel_w
        vel_b = self.drone.vel_b[:, 0, :3]
        ang_b = self.drone.vel_b[:, 0, 3:6]
        yaw_des = local_cmd["target_yaw"]
        yaw = quaternion_to_euler(self.drone.rot[:, 0, :])[:, 2]
        yaw_err = (yaw_des - yaw + torch.pi) % (2 * torch.pi) - torch.pi
        r_max = float(self.ctrl_cfg.get("r_max", 0.6))
        yaw_rate_kp = float(self.ctrl_cfg.get("yaw_rate_kp", 1.5))
        r_cmd = torch.clamp(yaw_rate_kp * yaw_err, -r_max, r_max)

        yaw_align_threshold = float(self.fusion_cfg.get("yaw_align_threshold", 0.6))
        yaw_align_scale = float(self.fusion_cfg.get("yaw_align_speed_scale", 0.0))
        need_align = torch.abs(yaw_err) > yaw_align_threshold
        if need_align.any():
            target_vel_w[need_align, 0:2] *= yaw_align_scale
        self._last_yaw_err = yaw_err
        self._last_r_cmd = r_cmd

        v_cmd_b = quat_rotate_inverse(self.drone.rot[:, 0, :], target_vel_w)
        u_max = float(self.ctrl_cfg.get("u_max", 0.6))
        v_max = float(self.ctrl_cfg.get("v_max", 0.4))
        w_max = float(self.ctrl_cfg.get("w_max", 0.3))
        v_cmd_b[:, 0] = torch.clamp(v_cmd_b[:, 0], -u_max, u_max)
        v_cmd_b[:, 1] = torch.clamp(v_cmd_b[:, 1], -v_max, v_max)
        v_cmd_b[:, 2] = torch.clamp(v_cmd_b[:, 2], -w_max, w_max)
        e_v = v_cmd_b - vel_b
        F_b = float(self.ctrl_cfg.get("vel_kp", 5.0)) * e_v
        gravity_scale = float(self.ctrl_cfg.get("gravity_comp_scale", 0.0))
        if gravity_scale != 0.0:
            masses = self.drone.masses[:, 0].squeeze(-1)
            F_b[:, 2] += masses * 9.81 * gravity_scale
        T_b = torch.zeros(self.num_envs, 3, device=self.device)
        T_b[:, 2] = float(self.ctrl_cfg.get("yaw_rate_kp", 1.5)) * (r_cmd - ang_b[:, 2])
        if bool(self.ctrl_cfg.get("stabilize_rp", True)):
            rpy = quaternion_to_euler(self.drone.rot[:, 0, :])
            T_b[:, 0] = -float(self.ctrl_cfg.get("rp_kp", 2.0)) * rpy[:, 0] - float(self.ctrl_cfg.get("rp_kd", 0.2)) * ang_b[:, 0]
            T_b[:, 1] = -float(self.ctrl_cfg.get("rp_kp", 2.0)) * rpy[:, 1] - float(self.ctrl_cfg.get("rp_kd", 0.2)) * ang_b[:, 1]

        if self._A_pinv is None or (self._alloc_step % int(self.ctrl_cfg.get("allocation_update_every", 1)) == 0):
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

        goal_w = self.target_pos[:, 0, :]
        rel_w = goal_w - pos_w
        rel_b = quat_rotate_inverse(rot_w, rel_w)
        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        yaw_goal = torch.atan2(rel_w[:, 1], rel_w[:, 0])
        yaw = quaternion_to_euler(rot_w)[:, 2]
        yaw_err = (yaw_goal - yaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_sin = torch.sin(yaw_err).unsqueeze(-1)
        yaw_cos = torch.cos(yaw_err).unsqueeze(-1)

        rpy = quaternion_to_euler(rot_w)
        roll_pitch = rpy[:, :2]
        depth = pos_w[:, 2:3]

        sonar_obs = self.sonar.scan(pos_w, rot_w)["ranges"]
        self._last_sonar = sonar_obs

        if self.debug and self.num_envs > 0:
            step = int(self.progress_buf[0].item())
            if step % max(self.debug_interval, 1) == 0:
                dist = torch.norm(rel_w[0]).item()
                behavior = -1 if self._last_behavior is None else int(self._last_behavior[0].item())
                tgt_vel = (
                    [0.0, 0.0, 0.0]
                    if self._last_target_vel_w is None
                    else self._last_target_vel_w[0].detach().cpu().numpy().tolist()
                )
                yaw_err_val = 0.0 if not hasattr(self, "_last_yaw_err") else float(self._last_yaw_err[0].item())
                r_cmd_val = 0.0 if not hasattr(self, "_last_r_cmd") else float(self._last_r_cmd[0].item())
                print(
                    "[HoverTankPPO] step",
                    step,
                    "pos=",
                    pos_w[0].detach().cpu().numpy(),
                    "goal=",
                    goal_w[0].detach().cpu().numpy(),
                    "dist=",
                    f"{dist:.3f}",
                    "behavior=",
                    behavior,
                    "yaw_err=",
                    f"{yaw_err_val:.3f}",
                    "r_cmd=",
                    f"{r_cmd_val:.3f}",
                    "target_vel_w=",
                    tgt_vel,
                )

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
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        td = super()._compute_reward_and_done()
        reward = td["agents", "reward"]
        dist = torch.norm(self.rpos, dim=-1)
        progress = (self._prev_goal_dist - dist).clamp_min(-5.0)
        self._prev_goal_dist = dist.detach()
        reward = reward + self.reward_goal_progress * progress.unsqueeze(-1)

        if self._last_sonar is not None:
            min_range = torch.min(self._last_sonar, dim=1)[0]
            proximity = (self.obstacle_threshold - min_range).clamp_min(0.0) / max(self.obstacle_threshold, 1e-6)
            reward = reward - self.obstacle_penalty * proximity.view(-1, 1, 1)

        success = dist < self.success_distance
        if success.any():
            reward[success] = reward[success] + self.success_bonus
        terminated = td["terminated"] | success
        truncated = td["truncated"]
        done = terminated | truncated
        td["agents", "reward"] = reward
        td["terminated"] = terminated
        td["done"] = done
        if self.debug:
            done_mask = td["done"].squeeze(-1).reshape(-1).bool()
            if done_mask.any():
                pos_error = torch.norm(self.rpos, dim=-1)
                distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)
                print(
                    "[HoverTankPPO] done",
                    "env_ids=",
                    torch.where(done_mask)[0].tolist(),
                    "pos_error=",
                    pos_error[done_mask].detach().cpu().numpy(),
                    "distance=",
                    distance[done_mask].detach().cpu().numpy(),
                    "progress=",
                    self.progress_buf.reshape(-1)[done_mask].detach().cpu().numpy(),
                    "success=",
                    success[done_mask].detach().cpu().numpy(),
                )
        return td
