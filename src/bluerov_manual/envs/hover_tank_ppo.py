import os
import torch

import omni.isaac.core.utils.prims as prim_utils
import marinegym.utils.kit as kit_utils

from tensordict.tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec

from marinegym.envs.isaac_env import AgentSpec
from marinegym.utils.torch import quat_rotate, quat_rotate_inverse, quaternion_to_euler, quat_axis, euler_to_quaternion
from marinegym.sensors.camera import Camera
from marinegym.sensors.config import PinholeCameraCfg
from omni.isaac.core.prims import XFormPrimView
from bluerov_manual.navigation import BehaviorFusion, BehaviorFusionConfig, GlobalPlanner, GlobalPlannerConfig, Sonar, SonarConfig
from bluerov_manual.navigation.local_policy import BehaviorAction
from bluerov_manual.navigation.obstacles import Obstacle

from .hover_tank import HoverTank


class HoverTankPPO(HoverTank):
    """HoverTank with PPO behavior actions (scalar -> discrete behavior)."""

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication q = q1 * q2."""
        w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
        w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def __init__(self, cfg, headless):
        self.ctrl_cfg = cfg.task.get("controller", {}) or {}
        planner_stack = cfg.task.get("planner_stack", {}) or {}
        if planner_stack.get("enable", False):
            self.sonar_cfg = planner_stack.get("sonar", {}) or {}
            self.planner_cfg = planner_stack.get("rrt", {}) or {}
            self.fusion_cfg = planner_stack.get("fusion", {}) or {}
            self.policy_cfg = planner_stack.get("policy", {}) or {}
        else:
            self.sonar_cfg = cfg.task.get("sonar", {}) or {}
            self.planner_cfg = cfg.task.get("planner", {}) or {}
            self.fusion_cfg = self.planner_cfg
            self.policy_cfg = {}
        self.n_behaviors = int(cfg.task.get("n_behaviors", self.planner_cfg.get("n_behaviors", 6)))
        self.success_distance = float(cfg.task.get("success_distance", 0.5))
        self.success_bonus = float(cfg.task.get("success_bonus", 5.0))
        self.reward_goal_progress = float(cfg.task.get("reward_goal_progress", 1.0))
        self.obstacle_penalty = float(cfg.task.get("obstacle_penalty", 0.5))
        self.obstacle_threshold = float(cfg.task.get("obstacle_threshold", 1.0))
        self.terminate_on_collision = bool(cfg.task.get("terminate_on_collision", True))
        self.collision_distance = float(cfg.task.get("collision_distance", 0.15))
        self.collision_penalty = float(cfg.task.get("collision_penalty", 1.0))
        self.reset_cooldown_steps = int(cfg.task.get("reset_cooldown_steps", 5))
        self.behavior_reward_cfg = cfg.task.get("behavior_reward", {}) or {}
        self.safety_cfg = cfg.task.get("safety", {}) or {}
        self.debug = bool(cfg.task.get("debug", False))
        self.debug_interval = int(cfg.task.get("debug_interval", 50))
        self.depth_cfg = cfg.task.get("depth_camera", {}) or {}
        self.depth_enabled = bool(self.depth_cfg.get("enable", False))
        self.depth_camera = None
        self.depth_camera_view = None
        self._last_depth_min = None
        self._depth_viz_every = int(self.depth_cfg.get("viz_every", 50))
        self._depth_viz_count = 0
        self._depth_rot_offset_q = None
        super().__init__(cfg, headless)
        self._reset_cooldown = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        print(
            "[HoverTankPPO] architecture="
            "PPO_behavior->BehaviorFusion->VelocityController->Thrusters | "
            "GlobalPlanner=RRT | Sonar=PhysX+Fallback"
        )
        if self.depth_enabled:
            data_type = str(self.depth_cfg.get("data_type", "distance_to_camera"))
            if data_type == "depth":
                data_type = "distance_to_camera"
            resolution = tuple(self.depth_cfg.get("resolution", [128, 72]))
            clip_range = self.depth_cfg.get("clip_range", None)
            extra_types = self.depth_cfg.get("extra_data_types", ["depth"])
            extra_types = [t for t in extra_types if t != "depth"]
            data_types = list({data_type, *extra_types})
            cam_cfg = PinholeCameraCfg(
                sensor_tick=0,
                resolution=resolution,
                data_types=data_types,
            )
            if clip_range is not None:
                cam_cfg.usd_params.clipping_range = (float(clip_range[0]), float(clip_range[1]))
            self.depth_camera = Camera(cfg=cam_cfg)
            cam_paths = prim_utils.find_matching_prim_paths("/World/envs/.*/Camera_depth")
            for cam_path in cam_paths:
                self.depth_camera._define_usd_camera_attributes(cam_path)
            self.depth_camera.initialize(prim_paths_expr="/World/envs/.*/Camera_depth")
            self.depth_camera_view = XFormPrimView(
                prim_paths_expr="/World/envs/env_*/Camera_depth",
                reset_xform_properties=False,
            )
            self.depth_camera_view.initialize()
            rot_offset_deg = self.depth_cfg.get("rot_offset_deg", [0.0, -90.0, 0.0])
            rot_offset = torch.tensor(rot_offset_deg, device=self.device) * (torch.pi / 180.0)
            self._depth_rot_offset_q = euler_to_quaternion(rot_offset)

        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, 1, device=self.device)
        self._last_sonar = None
        self._last_behavior = None
        self._last_target_vel_w = None
        self._last_wall_dist = None
        self._last_maze_dist = None
        self._last_obst_dist = None
        self._waypoint_tol = float(self.planner_cfg.get("waypoint_tol", 0.5))
        self._stuck_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self._stuck_replan_steps = int(self.policy_cfg.get("stuck_replan_steps", 60))
        self._stuck_speed_thresh = float(self.policy_cfg.get("stuck_speed_thresh", 0.02))
        self._plan_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._plot_rrt_every = int(self.planner_cfg.get("plot_every", 0))
        self._plot_rrt_gif = bool(self.planner_cfg.get("plot_gif", False))
        self._plot_rrt_gif_every = int(self.planner_cfg.get("plot_gif_every", 10))
        self._plot_rrt_history = int(self.planner_cfg.get("plot_history", 50))
        self._plot_rrt_3d = bool(self.planner_cfg.get("plot_3d", False))
        self._rrt_plot_paths = {}
        bounds_min = self.planner_cfg.get("bounds_min", [0.0, 0.0, -2.0])
        bounds_max = self.planner_cfg.get("bounds_max", [20.0, 6.0, 2.0])
        self._rrt_bounds_min = (float(bounds_min[0]), float(bounds_min[1]))
        self._rrt_bounds_max = (float(bounds_max[0]), float(bounds_max[1]))
        self._rrt_bounds_z = (float(bounds_min[2]), float(bounds_max[2]))
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
        if self._reset_cooldown is not None:
            self._reset_cooldown[env_ids] = self.reset_cooldown_steps
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
            self._plan_counts[env_id] += 1
            if self._plot_rrt_every > 0 and (self._plan_counts[env_id] % self._plot_rrt_every == 0):
                try:
                    import matplotlib.pyplot as plt
                    out_dir = os.path.join(os.getcwd(), "outputs", "rrt_paths")
                    os.makedirs(out_dir, exist_ok=True)
                    xs = [p[0] for p in plan] if plan else []
                    ys = [p[1] for p in plan] if plan else []
                    zs = [p[2] for p in plan] if plan else []
                    x_min, y_min = self._rrt_bounds_min
                    x_max, y_max = self._rrt_bounds_max
                    z_min, z_max = self._rrt_bounds_z
                    width = max(x_max - x_min, 1.0)
                    height = max(y_max - y_min, 1.0)
                    fig_w = 6.0
                    fig_h = max(3.0, fig_w * (height / width))
                    if self._plot_rrt_3d:
                        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                        fig = plt.figure(figsize=(fig_w, fig_h))
                        ax = fig.add_subplot(111, projection="3d")
                        ax.plot(xs, ys, zs, "-o", markersize=2, linewidth=1.0, label="RRT")
                        ax.scatter(
                            [pos_w[i, 0].item()],
                            [pos_w[i, 1].item()],
                            [pos_w[i, 2].item()],
                            c="blue",
                            s=20,
                            label="start",
                        )
                        ax.scatter([goal_w[0]], [goal_w[1]], [goal_w[2]], c="red", s=20, label="goal")
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        ax.set_zlim(z_min, z_max)
                    else:
                        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                        ax.plot(xs, ys, "-o", markersize=2, linewidth=1.0, label="RRT")
                        ax.scatter([pos_w[i, 0].item()], [pos_w[i, 1].item()], c="blue", s=20, label="start")
                        ax.scatter([goal_w[0]], [goal_w[1]], c="red", s=20, label="goal")
                        ax.set_aspect("equal", adjustable="box")
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                    ax.set_title(f"env_{env_id} plan_{int(self._plan_counts[env_id].item())}")
                    ax.legend(loc="best")
                    frame_path = os.path.join(
                        out_dir, f"rrt_env_{env_id}_plan_{int(self._plan_counts[env_id].item()):06d}.png"
                    )
                    fig.savefig(frame_path, dpi=150)
                    plt.close(fig)
                    history = self._rrt_plot_paths.get(env_id, [])
                    history.append(frame_path)
                    if len(history) > self._plot_rrt_history:
                        history = history[-self._plot_rrt_history:]
                    self._rrt_plot_paths[env_id] = history
                    if self._plot_rrt_gif and (self._plan_counts[env_id] % self._plot_rrt_gif_every == 0):
                        try:
                            import imageio.v2 as imageio
                            gif_path = os.path.join(out_dir, f"rrt_env_{env_id}.gif")
                            frames = [imageio.imread(p) for p in history if os.path.exists(p)]
                            if frames:
                                imageio.mimsave(gif_path, frames, duration=0.2)
                        except Exception as gif_exc:
                            if self.debug:
                                print(f"[HoverTankPPO] rrt gif failed: {gif_exc}")
                except Exception as exc:
                    if self.debug:
                        print(f"[HoverTankPPO] rrt plot failed: {exc}")

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
            self._A_pinv[~finite_mask] = 0.0
            if self.debug:
                bad_envs = torch.where(~finite_mask)[0].tolist()
                print(f"[HoverTankPPO] allocation skip non-finite envs={bad_envs}")
        if finite_mask.any():
            A_good = A[finite_mask]
            pinv_good = torch.linalg.pinv(A_good)
            if self._A_pinv is None:
                self._A_pinv = torch.zeros_like(A)
            self._A_pinv[finite_mask] = pinv_good

    def _design_scene(self):
        global_paths = super()._design_scene()
        if self.depth_enabled:
            for env_idx in range(self.num_envs):
                cam_path = f"/World/envs/env_{env_idx}/Camera_depth"
                if not prim_utils.is_prim_path_valid(cam_path):
                    prim_utils.create_prim(
                        prim_path=cam_path,
                        prim_type="Camera",
                        translation=(0.0, 0.0, 0.0),
                    )
        return global_paths

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
            if self._reset_cooldown is not None:
                self._reset_cooldown[bad_envs] = self.reset_cooldown_steps

        if self.depth_enabled and self.depth_camera_view is not None:
            cam_offset = torch.tensor(
                self.depth_cfg.get("offset", [0.2, 0.0, 0.0]),
                device=self.device,
                dtype=pos_w.dtype,
            )
            cam_offset = cam_offset.unsqueeze(0).expand(pos_w.shape[0], 3)
            cam_pos = pos_w + quat_rotate(rot_w, cam_offset)
            cam_rot = rot_w
            if self._depth_rot_offset_q is not None:
                cam_rot = self._quat_mul(rot_w, self._depth_rot_offset_q.expand_as(rot_w))
            self.depth_camera_view.set_world_poses(
                positions=cam_pos,
                orientations=cam_rot,
            )

        actions = tensordict[("agents", "action")]
        if self._prev_action is None:
            self._prev_action = torch.zeros_like(actions)
        alpha = float(self.ctrl_cfg.get("action_smooth_alpha", 0.8))
        actions = alpha * self._prev_action + (1.0 - alpha) * actions
        self._prev_action = actions

        behavior = self._behavior_from_action(actions.squeeze(1).squeeze(-1))
        # Safety override using analytic clearance (walls/maze/obstacles)
        min_range = self._wall_clearance(pos_w)
        maze_clear = self._maze_clearance(pos_w)
        if maze_clear is not None:
            min_range = torch.minimum(min_range, maze_clear)
        obst_clear = self._obstacle_clearance(pos_w)
        if obst_clear is not None:
            min_range = torch.minimum(min_range, obst_clear)

        stop_dist = float(self.policy_cfg.get("min_range_stop", self.behavior_reward_cfg.get("stop_dist", 0.5)))
        avoid_dist = float(self.policy_cfg.get("min_range_avoid", self.behavior_reward_cfg.get("avoid_dist", 1.2)))
        replan_dist = float(self.policy_cfg.get("min_range_replan", self.behavior_reward_cfg.get("replan_dist", 0.8)))

        safety_stop = min_range < stop_dist
        safety_avoid = (min_range < avoid_dist) & ~safety_stop
        safety_replan = (min_range < replan_dist) & ~(safety_stop | safety_avoid)

        if safety_stop.any() or safety_avoid.any() or safety_replan.any():
            behavior = behavior.clone()
            if safety_replan.any():
                behavior = torch.where(
                    safety_replan,
                    torch.full_like(behavior, int(BehaviorAction.REQUEST_REPLAN)),
                    behavior,
                )
            if safety_stop.any():
                behavior = torch.where(
                    safety_stop,
                    torch.full_like(behavior, int(BehaviorAction.STOP)),
                    behavior,
                )
            if safety_avoid.any():
                avoid_behavior = torch.full_like(behavior, int(BehaviorAction.STOP))
                if self.enable_obstacles and self.num_obstacles > 0:
                    centers = self.obst_center + self.envs_positions.unsqueeze(1)
                    rel = centers - pos_w[:, None, :]
                    dist = torch.norm(rel, dim=-1)
                    idx = dist.argmin(dim=1)
                    nearest = rel[torch.arange(self.num_envs, device=self.device), idx]
                    rel_b = quat_rotate_inverse(rot_w, nearest)
                    avoid_left = rel_b[:, 1] < 0.0
                    avoid_behavior = torch.where(
                        avoid_left,
                        torch.full_like(behavior, int(BehaviorAction.AVOID_LEFT)),
                        torch.full_like(behavior, int(BehaviorAction.AVOID_RIGHT)),
                    )
                behavior = torch.where(safety_avoid, avoid_behavior, behavior)

        # Stuck replan trigger: low speed + not at goal for N steps.
        dist_goal = torch.norm(self.target_pos[:, 0, :] - pos_w, dim=-1)
        speed = torch.norm(self.drone.vel_b[:, 0, :3], dim=-1)
        stuck = (speed < self._stuck_speed_thresh) & (dist_goal > self.success_distance)
        self._stuck_steps = torch.where(
            stuck, self._stuck_steps + 1, torch.zeros_like(self._stuck_steps)
        )
        replan_stuck = self._stuck_steps >= self._stuck_replan_steps
        if replan_stuck.any():
            behavior = torch.where(
                replan_stuck,
                torch.full_like(behavior, int(BehaviorAction.REQUEST_REPLAN)),
                behavior,
            )
            self._stuck_steps = torch.where(
                replan_stuck, torch.zeros_like(self._stuck_steps), self._stuck_steps
            )

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
        if self._reset_cooldown is not None:
            cooldown_mask = self._reset_cooldown > 0
            if cooldown_mask.any():
                thr_action[cooldown_mask] = 0.0
                self._reset_cooldown[cooldown_mask] -= 1
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

        # Maze walls (axis-aligned rectangles per yaw 0/90 deg).
        if getattr(self, "enable_maze", False) and getattr(self, "num_maze_walls", 0) > 0:
            pos_local = pos_w - self.envs_positions
            centers = self.maze_centers
            yaw = self.maze_yaw
            is_vertical = torch.abs(torch.sin(yaw)) > torch.abs(torch.cos(yaw))
            half_x = torch.where(
                is_vertical,
                torch.full_like(centers[..., 0], self.maze_thickness * 0.5),
                torch.full_like(centers[..., 0], self.maze_length * 0.5),
            )
            half_y = torch.where(
                is_vertical,
                torch.full_like(centers[..., 1], self.maze_length * 0.5),
                torch.full_like(centers[..., 1], self.maze_thickness * 0.5),
            )
            x_min_w = (centers[..., 0] - half_x).unsqueeze(1)
            x_max_w = (centers[..., 0] + half_x).unsqueeze(1)
            y_min_w = (centers[..., 1] - half_y).unsqueeze(1)
            y_max_w = (centers[..., 1] + half_y).unsqueeze(1)
            z_min = (centers[..., 2] - self.maze_height * 0.5).unsqueeze(1)
            z_max = (centers[..., 2] + self.maze_height * 0.5).unsqueeze(1)

            ox = pos_local[:, 0].unsqueeze(1).unsqueeze(-1)
            oy = pos_local[:, 1].unsqueeze(1).unsqueeze(-1)
            oz = pos_local[:, 2].unsqueeze(1).unsqueeze(-1)
            dx = dir_world[:, :, 0].unsqueeze(-1)
            dy = dir_world[:, :, 1].unsqueeze(-1)
            eps = 1e-6
            dx_safe = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
            dy_safe = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)

            tx1 = (x_min_w - ox) / dx_safe
            tx2 = (x_max_w - ox) / dx_safe
            tmin_x = torch.minimum(tx1, tx2)
            tmax_x = torch.maximum(tx1, tx2)

            ty1 = (y_min_w - oy) / dy_safe
            ty2 = (y_max_w - oy) / dy_safe
            tmin_y = torch.minimum(ty1, ty2)
            tmax_y = torch.maximum(ty1, ty2)

            tmin = torch.maximum(tmin_x, tmin_y)
            tmax = torch.minimum(tmax_x, tmax_y)
            z_ok = (oz >= z_min) & (oz <= z_max)
            hit = (tmax >= tmin) & (tmax > 0.0) & z_ok
            t_hit = torch.where(hit, tmin, torch.full_like(tmin, max_range))
            t_min = t_hit.min(dim=-1).values  # [E,R]
            ranges = torch.minimum(ranges, t_min.clamp(min=0.0, max=max_range))

        return ranges

    def _wall_clearance(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Compute min distance to world bounds (x/y walls)."""
        bounds_min = torch.as_tensor(self.planner_cfg.get("bounds_min", [0.0, 0.0, -2.0]), device=pos_w.device)
        bounds_max = torch.as_tensor(self.planner_cfg.get("bounds_max", [20.0, 6.0, 2.0]), device=pos_w.device)
        wall_margin = float(self.planner_cfg.get("wall_margin", 0.0))
        x_min = bounds_min[0] + wall_margin
        x_max = bounds_max[0] - wall_margin
        y_min = bounds_min[1] + wall_margin
        y_max = bounds_max[1] - wall_margin
        dx = torch.minimum(pos_w[:, 0] - x_min, x_max - pos_w[:, 0])
        dy = torch.minimum(pos_w[:, 1] - y_min, y_max - pos_w[:, 1])
        return torch.minimum(dx, dy).clamp(min=0.0)

    def _maze_clearance(self, pos_w: torch.Tensor) -> torch.Tensor | None:
        """Compute min planar distance to maze walls (approx axis-aligned)."""
        if not getattr(self, "enable_maze", False) or getattr(self, "num_maze_walls", 0) == 0:
            return None
        pos_local = pos_w - self.envs_positions
        centers = self.maze_centers
        yaw = self.maze_yaw
        is_vertical = torch.abs(torch.sin(yaw)) > torch.abs(torch.cos(yaw))
        half_x = torch.where(
            is_vertical,
            torch.full_like(centers[..., 0], self.maze_thickness * 0.5),
            torch.full_like(centers[..., 0], self.maze_length * 0.5),
        )
        half_y = torch.where(
            is_vertical,
            torch.full_like(centers[..., 1], self.maze_length * 0.5),
            torch.full_like(centers[..., 1], self.maze_thickness * 0.5),
        )
        dx = torch.abs(pos_local[:, None, 0] - centers[..., 0]) - half_x
        dy = torch.abs(pos_local[:, None, 1] - centers[..., 1]) - half_y
        dx = dx.clamp_min(0.0)
        dy = dy.clamp_min(0.0)
        dist = torch.sqrt(dx * dx + dy * dy)
        return dist.min(dim=1).values

    def _obstacle_clearance(self, pos_w: torch.Tensor) -> torch.Tensor | None:
        """Compute min distance to dynamic obstacles (center distance minus radius)."""
        if not getattr(self, "enable_obstacles", False) or getattr(self, "num_obstacles", 0) == 0:
            return None
        radius = float(self.obst_cfg.get("radius", 0.15))
        centers = self.obst_center + self.envs_positions.unsqueeze(1)
        d = torch.norm(pos_w[:, None, :] - centers, dim=-1) - radius
        return d.clamp_min(0.0).min(dim=1).values

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
        expected_rays = int(self.sonar_cfg.get("num_rays", 32))
        if sonar_obs.dim() != 2 or sonar_obs.shape[1] != expected_rays:
            sonar_obs = torch.full(
                (pos_w.shape[0], expected_rays),
                float(self.sonar.cfg.max_range),
                device=self.device,
            )
        self._last_sonar = sonar_obs
        self._last_depth_min = None
        self._last_wall_dist = self._wall_clearance(pos_w)
        self._last_maze_dist = self._maze_clearance(pos_w)
        self._last_obst_dist = self._obstacle_clearance(pos_w)
        if self.depth_enabled and self.depth_camera is not None:
            try:
                imgs = self.depth_camera.get_images()
            except Exception as exc:
                if self.debug:
                    print(f"[HoverTankPPO] depth camera read failed: {exc}")
                imgs = None
            data_type = str(self.depth_cfg.get("data_type", "distance_to_camera"))
            depth_img = None
            if imgs is not None and data_type in imgs.keys(True, True):
                depth_img = imgs[data_type]
                depth_img = depth_img.squeeze(1)
                if depth_img.dim() == 2:
                    depth_img = depth_img.unsqueeze(0)
                elif depth_img.dim() != 3:
                    # Invalid frame, skip.
                    depth_img = None
                if depth_img is not None and depth_img.shape[0] != pos_w.shape[0]:
                    if depth_img.shape[0] == 1:
                        depth_img = depth_img.expand(pos_w.shape[0], -1, -1)
                    else:
                        depth_img = depth_img[: pos_w.shape[0]]
                if depth_img is not None:
                    max_range = float(self.depth_cfg.get("max_range", 10.0))
                    min_filter = float(self.depth_cfg.get("min_range_filter", 0.2))
                    depth_img = depth_img.clamp(min=0.0, max=max_range)
                    if float(depth_img.max() - depth_img.min()) < 1e-3:
                        alt_type = "depth" if data_type != "depth" else "distance_to_camera"
                        if alt_type in imgs.keys(True, True):
                            alt_img = imgs[alt_type].squeeze(1)
                            if alt_img.dim() == 2:
                                alt_img = alt_img.unsqueeze(0)
                            if alt_img.dim() == 3:
                                depth_img = alt_img.clamp(min=0.0, max=max_range)
                    valid = depth_img > min_filter
                    depth_min = torch.where(
                        valid.any(dim=(1, 2)),
                        torch.where(valid, depth_img, torch.full_like(depth_img, max_range)).amin(dim=(1, 2)),
                        torch.full((depth_img.shape[0],), max_range, device=depth_img.device),
                    )
                    self._last_depth_min = depth_min
                    if self._depth_viz_every > 0:
                        self._depth_viz_count += 1
                        if self._depth_viz_count % self._depth_viz_every == 0:
                            from marinegym.utils.image import save_depth
                            out_dir = os.path.join(os.getcwd(), "outputs")
                            os.makedirs(out_dir, exist_ok=True)
                            # save_depth expects (1, H, W)
                            depth_viz = depth_img[0].detach().cpu().unsqueeze(0)
                            step_dir = os.path.join(out_dir, f"depth_step_{self._depth_viz_count:06d}")
                            os.makedirs(step_dir, exist_ok=True)
                            save_depth(depth_viz, save_path=step_dir)

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
                depth_min_dbg = None
                wall_min_dbg = None
                maze_min_dbg = None
                obst_min_dbg = None
                zone = "clear"
                if self._last_sonar is not None:
                    min_range = float(torch.min(self._last_sonar[0]).item())
                    if self._last_depth_min is not None:
                        depth_min_dbg = float(self._last_depth_min[0].item())
                        min_range = min(min_range, depth_min_dbg)
                    if self._last_wall_dist is not None:
                        wall_min_dbg = float(self._last_wall_dist[0].item())
                        min_range = min(min_range, wall_min_dbg)
                    if self._last_maze_dist is not None:
                        maze_min_dbg = float(self._last_maze_dist[0].item())
                        min_range = min(min_range, maze_min_dbg)
                    if self._last_obst_dist is not None:
                        obst_min_dbg = float(self._last_obst_dist[0].item())
                        min_range = min(min_range, obst_min_dbg)
                    stop_dist = float(self.behavior_reward_cfg.get("stop_dist", 0.5))
                    avoid_dist = float(self.behavior_reward_cfg.get("avoid_dist", 1.2))
                    replan_dist = float(self.behavior_reward_cfg.get("replan_dist", 0.8))
                    if min_range < stop_dist:
                        zone = "stop"
                    elif min_range < avoid_dist:
                        zone = "avoid"
                    elif min_range < replan_dist:
                        zone = "replan"
                print(
                    "[HoverTankPPO]",
                    f"step={step}",
                    f"pos={pos_w[0].detach().cpu().numpy()}",
                    f"goal={goal_w[0].detach().cpu().numpy()}",
                    f"dist={dist:.3f}",
                    f"behavior={behavior}",
                    f"min_range={min_range:.3f}" if min_range is not None else "min_range=na",
                    f"depth_min={depth_min_dbg:.3f}" if depth_min_dbg is not None else "depth_min=na",
                    f"wall_min={wall_min_dbg:.3f}" if wall_min_dbg is not None else "wall_min=na",
                    f"maze_min={maze_min_dbg:.3f}" if maze_min_dbg is not None else "maze_min=na",
                    f"obst_min={obst_min_dbg:.3f}" if obst_min_dbg is not None else "obst_min=na",
                    f"zone={zone}",
                    f"yaw_err={yaw_err_val:.3f}",
                    f"r_cmd={r_cmd_val:.3f}",
                    f"target_vel_w={tgt_vel}",
                )

        parts = [
            rel_b,
            yaw_sin,
            yaw_cos,
            vel_b,
            ang_b[:, 2:3],
            depth,
            roll_pitch,
            sonar_obs,
        ]
        fixed = []
        for part in parts:
            if part.dim() == 1:
                part = part.unsqueeze(-1)
            if part.dim() > 2:
                part = part.reshape(part.shape[0], -1)
            fixed.append(part)
        obs = torch.cat(fixed, dim=-1).unsqueeze(1)

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
            if self._last_depth_min is not None:
                min_range = torch.minimum(min_range, self._last_depth_min)
            if self._last_wall_dist is not None:
                min_range = torch.minimum(min_range, self._last_wall_dist)
            if self._last_maze_dist is not None:
                min_range = torch.minimum(min_range, self._last_maze_dist)
            if self._last_obst_dist is not None:
                min_range = torch.minimum(min_range, self._last_obst_dist)
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
            if self._last_depth_min is not None:
                min_range = torch.minimum(min_range, self._last_depth_min)
            if self._last_wall_dist is not None:
                min_range = torch.minimum(min_range, self._last_wall_dist)
            if self._last_maze_dist is not None:
                min_range = torch.minimum(min_range, self._last_maze_dist)
            if self._last_obst_dist is not None:
                min_range = torch.minimum(min_range, self._last_obst_dist)
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
            if self._last_depth_min is not None:
                min_range = torch.minimum(min_range, self._last_depth_min)
            if self._last_wall_dist is not None:
                min_range = torch.minimum(min_range, self._last_wall_dist)
            if self._last_maze_dist is not None:
                min_range = torch.minimum(min_range, self._last_maze_dist)
            if self._last_obst_dist is not None:
                min_range = torch.minimum(min_range, self._last_obst_dist)
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
