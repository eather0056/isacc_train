import os
import torch

import omni.isaac.core.utils.prims as prim_utils
import marinegym.utils.kit as kit_utils

from tensordict.tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec

from marinegym.envs.isaac_env import AgentSpec
from marinegym.utils.torch import quat_rotate, quat_rotate_inverse, quaternion_to_euler, quat_axis, euler_to_quaternion
from bluerov_manual.navigation import BehaviorFusion, BehaviorFusionConfig, GlobalPlanner, GlobalPlannerConfig, Sonar, SonarConfig
from bluerov_manual.navigation.local_policy import BehaviorAction
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
        self.terminate_on_collision = bool(cfg.task.get("terminate_on_collision", True))
        self.collision_distance = float(cfg.task.get("collision_distance", 0.15))
        self.collision_penalty = float(cfg.task.get("collision_penalty", 1.0))
        self.behavior_reward_cfg = cfg.task.get("behavior_reward", {}) or {}
        self.debug = bool(cfg.task.get("debug", False))
        self.debug_interval = int(cfg.task.get("debug_interval", 50))
        super().__init__(cfg, headless)
        print(
            "[HoverTankPPO] architecture="
            "PPO_behavior->BehaviorFusion->VelocityController->Thrusters | "
            "GlobalPlanner=RRT | Sonar=PhysX+Fallback"
        )

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
        finite_mask = torch.isfinite(A).all(dim=(1, 2))
        if not finite_mask.all():
            if self._A_pinv is None:
                self._A_pinv = torch.zeros(
                    self.num_envs, r_b.shape[1], 6, device=self.device
                )
            if self.debug:
                bad_envs = torch.where(~finite_mask)[0].tolist()
                print(f"[HoverTankPPO] allocation skip non-finite envs={bad_envs}")
        if finite_mask.any():
            A_good = A[finite_mask]
            pinv_good = torch.linalg.pinv(A_good)
            if self._A_pinv is None:
                self._A_pinv = torch.zeros_like(A)
            self._A_pinv[finite_mask] = pinv_good

    def _pre_sim_step(self, tensordict):
        self._update_obstacles()
        self.drone.get_state()

        pos_w = self.drone.pos[:, 0, :]
        rot_w = self.drone.rot[:, 0, :]
        bad_state = ~torch.isfinite(pos_w).all(dim=-1) | ~torch.isfinite(rot_w).all(dim=-1)
        if bad_state.any():
            bad_envs = torch.where(bad_state)[0]
            if self.debug:
                print(f"[HoverTankPPO] reset non-finite (pre_step) envs={bad_envs.tolist()}")
            self._reset_idx(bad_envs)
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
        max_speed = float(self.fusion_cfg.get("max_speed", self.ctrl_cfg.get("max_speed", 0.5)))
        target_vel_w = torch.clamp(target_vel_w, min=-max_speed, max=max_speed)
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
        thr_action = torch.nan_to_num(thr_action, nan=0.0, posinf=0.0, neginf=0.0)
        bad_thr = ~torch.isfinite(thr_action).all(dim=-1)
        if bad_thr.any():
            thr_action[bad_thr] = 0.0
            if self.debug:
                bad_envs = torch.where(bad_thr)[0].tolist()
                print(f"[HoverTankPPO] zero thrusters for envs={bad_envs}")
        self.effort = torch.abs(self.drone.apply_action(thr_action))

    def _fallback_sonar_ranges(self, pos_w: torch.Tensor, rot_w: torch.Tensor) -> torch.Tensor | None:
        """Fallback sonar: ray distances to walls/obstacles using analytic geometry."""
        num_envs = pos_w.shape[0]
        num_rays = int(self.sonar.cfg.num_rays)
        if num_rays <= 0:
            return None

        max_range = float(self.sonar.cfg.max_range)
        angles = self.sonar.angles.to(pos_w.device)
        dir_body = torch.stack(
            [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
        )
        dir_body = dir_body.unsqueeze(0).expand(num_envs, -1, -1)
        rot_expanded = rot_w.unsqueeze(1).expand(-1, num_rays, -1)
        dir_world = quat_rotate(rot_expanded.reshape(-1, 4), dir_body.reshape(-1, 3)).reshape(
            num_envs, num_rays, 3
        )
        dir_world = dir_world / (dir_world.norm(dim=-1, keepdim=True).clamp_min(1e-6))

        ranges = torch.full((num_envs, num_rays), max_range, device=pos_w.device)

        # Wall intersection (2D bounds).
        bounds_min = torch.as_tensor(self.planner_cfg.get("bounds_min", [0.0, 0.0, -2.0]), device=pos_w.device)
        bounds_max = torch.as_tensor(self.planner_cfg.get("bounds_max", [20.0, 6.0, 2.0]), device=pos_w.device)
        wall_margin = float(self.planner_cfg.get("wall_margin", 0.0))
        x_min = float(bounds_min[0] + wall_margin)
        x_max = float(bounds_max[0] - wall_margin)
        y_min = float(bounds_min[1] + wall_margin)
        y_max = float(bounds_max[1] - wall_margin)

        x0 = pos_w[:, 0].unsqueeze(1)
        y0 = pos_w[:, 1].unsqueeze(1)
        dx = dir_world[:, :, 0]
        dy = dir_world[:, :, 1]

        # Avoid division by zero.
        eps = 1e-6
        dx_safe = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
        dy_safe = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)

        t_xmin = (x_min - x0) / dx_safe
        t_xmax = (x_max - x0) / dx_safe
        t_ymin = (y_min - y0) / dy_safe
        t_ymax = (y_max - y0) / dy_safe

        def _valid_wall_t(t, x_hit, y_hit):
            return (t > 0.0) & (x_hit >= x_min) & (x_hit <= x_max) & (y_hit >= y_min) & (y_hit <= y_max)

        y_hit_xmin = y0 + t_xmin * dy
        y_hit_xmax = y0 + t_xmax * dy
        x_hit_ymin = x0 + t_ymin * dx
        x_hit_ymax = x0 + t_ymax * dx

        wall_ts = []
        wall_ts.append(torch.where(_valid_wall_t(t_xmin, x_min, y_hit_xmin), t_xmin, torch.full_like(t_xmin, max_range)))
        wall_ts.append(torch.where(_valid_wall_t(t_xmax, x_max, y_hit_xmax), t_xmax, torch.full_like(t_xmax, max_range)))
        wall_ts.append(torch.where(_valid_wall_t(t_ymin, x_hit_ymin, y_min), t_ymin, torch.full_like(t_ymin, max_range)))
        wall_ts.append(torch.where(_valid_wall_t(t_ymax, x_hit_ymax, y_max), t_ymax, torch.full_like(t_ymax, max_range)))
        wall_t = torch.minimum(torch.minimum(wall_ts[0], wall_ts[1]), torch.minimum(wall_ts[2], wall_ts[3]))
        ranges = torch.minimum(ranges, wall_t.clamp(min=0.0, max=max_range))

        # Obstacle intersection (spheres/capsules).
        if self.enable_obstacles and self.obstacles is not None:
            obst_pos, _ = self.obstacles.get_world_poses()
            obst_pos = obst_pos.to(pos_w.device)
            radius = float(self.obst_cfg.get("radius", 0.15))
            height = float(self.obst_cfg.get("height", 0.30))
            z_tol = height * 0.5 + radius

            O = pos_w[:, None, None, :]  # [E,1,1,3]
            D = dir_world[:, :, None, :]  # [E,R,1,3]
            C = obst_pos[:, None, :, :]   # [E,1,O,3]

            L = C - O
            t_ca = (L * D).sum(dim=-1)
            d2 = (L * L).sum(dim=-1) - t_ca**2
            z_ok = (L[..., 2].abs() <= z_tol)
            r2 = radius ** 2
            hit = (t_ca > 0.0) & (d2 <= r2) & z_ok
            thc = torch.sqrt(torch.clamp(r2 - d2, min=0.0))
            t_hit = torch.where(hit, t_ca - thc, torch.full_like(t_ca, max_range))
            t_min = t_hit.min(dim=-1).values  # [E,R]
            ranges = torch.minimum(ranges, t_min.clamp(min=0.0, max=max_range))

        return ranges

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        pos_w = self.drone.pos[:, 0, :]
        rot_w = self.drone.rot[:, 0, :]
        bad_state = ~torch.isfinite(pos_w).all(dim=-1) | ~torch.isfinite(rot_w).all(dim=-1)
        if bad_state.any():
            bad_envs = torch.where(bad_state)[0]
            if self.debug:
                print(f"[HoverTankPPO] reset non-finite envs={bad_envs.tolist()}")
            self._reset_idx(bad_envs)
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

        raw_sonar = self.sonar.scan(pos_w, rot_w)["ranges"]
        sonar_obs = raw_sonar
        fallback = self._fallback_sonar_ranges(pos_w, rot_w)
        self._last_sonar_fallback = False
        if fallback is not None:
            sonar_obs = torch.minimum(sonar_obs, fallback)
            self._last_sonar_fallback = True
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
                min_range = None
                zone = "clear"
                hit_path = None
                if self._last_sonar is not None:
                    min_range = float(torch.min(self._last_sonar[0]).item())
                    hit_count = int((self._last_sonar[0] < (self.sonar_cfg.get("max_range", 10.0) - 1e-3)).sum().item())
                    stop_dist = float(self.behavior_reward_cfg.get("stop_dist", 0.5))
                    avoid_dist = float(self.behavior_reward_cfg.get("avoid_dist", 1.2))
                    replan_dist = float(self.behavior_reward_cfg.get("replan_dist", 0.8))
                    if min_range < stop_dist:
                        zone = "stop"
                    elif min_range < avoid_dist:
                        zone = "avoid"
                    elif min_range < replan_dist:
                        zone = "replan"
                if getattr(self.sonar, "last_hit_path", None):
                    hit_path = self.sonar.last_hit_path
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
                    "min_range=",
                    f"{min_range:.3f}" if min_range is not None else "na",
                    "hits=",
                    f"{hit_count}" if min_range is not None else "na",
                    "zone=",
                    zone,
                    "hit_path=",
                    hit_path if hit_path else "na",
                    "fallback=",
                    "yes" if self._last_sonar_fallback else "no",
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

        if self._last_behavior is not None and self._last_sonar is not None and bool(self.behavior_reward_cfg.get("enable", True)):
            stop_dist = float(self.behavior_reward_cfg.get("stop_dist", 0.5))
            avoid_dist = float(self.behavior_reward_cfg.get("avoid_dist", 1.2))
            replan_dist = float(self.behavior_reward_cfg.get("replan_dist", 0.8))
            reward_stop = float(self.behavior_reward_cfg.get("reward_stop", 0.5))
            reward_avoid = float(self.behavior_reward_cfg.get("reward_avoid", 0.2))
            reward_replan = float(self.behavior_reward_cfg.get("reward_replan", 0.1))
            penalty_ignore = float(self.behavior_reward_cfg.get("penalty_ignore", 0.3))

            min_range = torch.min(self._last_sonar, dim=1)[0]
            behavior = self._last_behavior
            is_stop = behavior == int(BehaviorAction.STOP)
            is_avoid = (behavior == int(BehaviorAction.AVOID_LEFT)) | (behavior == int(BehaviorAction.AVOID_RIGHT))
            is_replan = behavior == int(BehaviorAction.REQUEST_REPLAN)

            stop_zone = min_range < stop_dist
            avoid_zone = (min_range < avoid_dist) & ~stop_zone
            replan_zone = (min_range < replan_dist) & ~(stop_zone | avoid_zone)

            bonus = torch.zeros_like(min_range)
            bonus = bonus + torch.where(stop_zone, torch.where(is_stop, reward_stop, -penalty_ignore), 0.0)
            bonus = bonus + torch.where(avoid_zone, torch.where(is_avoid, reward_avoid, -penalty_ignore), 0.0)
            bonus = bonus + torch.where(replan_zone, torch.where(is_replan, reward_replan, -penalty_ignore), 0.0)
            reward = reward + bonus.view(-1, 1, 1)

        success = dist < self.success_distance
        success_mask = success.unsqueeze(-1)
        if success.any():
            reward = reward + success_mask * self.success_bonus
        collision = torch.zeros_like(success)
        if self.terminate_on_collision and self._last_sonar is not None:
            min_range = torch.min(self._last_sonar, dim=1)[0]
            collision = (min_range < self.collision_distance).unsqueeze(-1)
            if collision.any():
                reward = reward - collision.unsqueeze(-1) * self.collision_penalty
        terminated = td["terminated"] | success | collision
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
