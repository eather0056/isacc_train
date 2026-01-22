import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import hydra
import torch
from omegaconf import OmegaConf

from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

from bluerov_manual.navigation import (
    BehaviorAction,
    BehaviorFusion,
    BehaviorFusionConfig,
    GlobalPlanner,
    GlobalPlannerConfig,
    LocalPolicy,
    LocalPolicyConfig,
    ObstacleField,
    ObstacleFieldConfig,
    Sonar,
    SonarConfig,
)

def build_allocation_matrix_from_usd(drone, device, thrust_axis=0, thrust_sign=1.0):
    """
    Builds A (6 x N) such that: wrench_body = A @ f
    Assumes thruster force direction is rotor local +X axis (axis=0).
    """
    from marinegym.utils.torch import quat_rotate_inverse, quat_axis

    rotor_pos_w, rotor_rot_w = drone.rotors_view.get_world_poses()
    r_w = rotor_pos_w[0, 0]  # (N,3)
    q_w = rotor_rot_w[0, 0]  # (N,4)

    base_pos_w, base_rot_w = drone.base_link.get_world_poses()
    base_pos_w = base_pos_w[0, 0]  # (3,)
    base_rot_w = base_rot_w[0, 0]  # (4,)

    # local +X expressed in world
    d_w = quat_axis(q_w, axis=thrust_axis) * float(thrust_sign)  # (N,3)
    r_rel_w = r_w - base_pos_w.unsqueeze(0)  # (N,3)

    N = r_rel_w.shape[0]
    base_q = base_rot_w.expand(N, 4)  # (N,4)

    # world -> body
    r_b = quat_rotate_inverse(base_q, r_rel_w)  # (N,3)
    d_b = quat_rotate_inverse(base_q, d_w)      # (N,3)

    tau_b = torch.cross(r_b, d_b, dim=-1)       # (N,3)

    A = torch.zeros(6, N, device=device)
    A[0:3, :] = d_b.T
    A[3:6, :] = tau_b.T
    return A


def sample_random_goals(num_envs, device, *, x_range, y_range, z_range):
    """
    Random goals in WORLD frame. Returns (E,3).
    """
    lo = torch.tensor([x_range[0], y_range[0], z_range[0]], device=device)
    hi = torch.tensor([x_range[1], y_range[1], z_range[1]], device=device)
    u = torch.rand(num_envs, 3, device=device)
    return lo + (hi - lo) * u


def wrench_pd_world_to_body(
    drone,
    goals_w,
    kp=25.0,
    kd=12.0,
    target_vel_w=None,
    target_yaw=None,
    yaw_kp=2.0,
    yaw_kd=0.2,
    gravity_comp=False,
    gravity_scale=1.0,
    max_accel=None,
    max_force=None,
    max_torque=None,
    stabilize_rp=False,
    rp_kp=2.0,
    rp_kd=0.2,
):
    """
    Vectorized PD in world frame, output desired BODY wrench.
    returns wrench_b: (E,6) = [Fx,Fy,Fz,Tx,Ty,Tz] in BODY frame
    """
    from marinegym.utils.torch import quat_rotate_inverse

    pos_w = drone.pos[:, 0, :]                         # (E,3)
    rot_w = drone.rot[:, 0, :]                         # (E,4)
    vel_w6 = drone.get_velocities(True)[:, 0, :]       # (E,6)
    linvel_w = vel_w6[:, :3]                           # (E,3)

    e_p = goals_w - pos_w
    if target_vel_w is None:
        e_v = -linvel_w
    else:
        e_v = target_vel_w - linvel_w

    F_des_w = kp * e_p + kd * e_v                      # (E,3)
    if gravity_comp:
        masses = drone.masses[:, 0].squeeze(-1)
        F_des_w[:, 2] += masses * 9.81 * gravity_scale
    F_des_b = quat_rotate_inverse(rot_w, F_des_w)      # (E,3)

    T_des_b = torch.zeros(pos_w.shape[0], 3, device=pos_w.device)
    if target_yaw is not None:
        from marinegym.utils.torch import quaternion_to_euler

        rpy = quaternion_to_euler(rot_w)
        yaw = rpy[..., 2]
        yaw_err = target_yaw - yaw
        yaw_err = (yaw_err + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_rate = vel_w6[:, 5]
        T_des_b[:, 2] = yaw_kp * yaw_err - yaw_kd * yaw_rate
        if stabilize_rp:
            roll = rpy[..., 0]
            pitch = rpy[..., 1]
            T_des_b[:, 0] = -rp_kp * roll - rp_kd * vel_w6[:, 3]
            T_des_b[:, 1] = -rp_kp * pitch - rp_kd * vel_w6[:, 4]

    if max_accel is not None:
        masses = drone.masses[:, 0].squeeze(-1)
        max_force = masses * float(max_accel)
    if max_force is not None:
        max_force_val = torch.as_tensor(max_force, device=pos_w.device)
        force_norm = torch.norm(F_des_w, dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(max_force_val / force_norm.squeeze(-1), max=1.0).unsqueeze(-1)
        F_des_w = F_des_w * scale
    if max_torque is not None:
        max_torque_val = float(max_torque)
        torque_norm = torch.norm(T_des_b, dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(max_torque_val / torque_norm.squeeze(-1), max=1.0).unsqueeze(-1)
        T_des_b = T_des_b * scale
    return torch.cat([F_des_b, T_des_b], dim=-1)       # (E,6)


def forces_to_action(f, f_max):
    return torch.clamp(f / f_max, -1.0, 1.0)


def read_reward_done(td):
    """
    Handle both ("agents","reward") and ("next","agents","reward") conventions.
    """
    if ("agents", "reward") in td.keys(True, True):
        r = td[("agents", "reward")]
        done = td["done"]
    else:
        r = td[("next", "agents", "reward")]
        done = td[("next", "done")]
    return r, done


def _ensure_goal_markers(stage, env_paths, *, radius=0.5, height=0.02, color=(1.0, 0.0, 0.0)):
    from pxr import UsdGeom
    import omni.isaac.core.utils.prims as prim_utils

    marker_paths = []
    for env_path in env_paths:
        root_path = f"{env_path}/goal_marker"
        marker_path = f"{root_path}/circle"

        if not prim_utils.is_prim_path_valid(root_path):
            UsdGeom.Xform.Define(stage, root_path)

        if not prim_utils.is_prim_path_valid(marker_path):
            cyl = UsdGeom.Cylinder.Define(stage, marker_path)
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.CreateAxisAttr("Z")
            cyl.CreateDisplayColorAttr().Set([color])

        marker_paths.append(root_path)
    return marker_paths


def _update_goal_markers(stage, marker_paths, goals):
    from pxr import UsdGeom

    for marker_path, goal in zip(marker_paths, goals):
        xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(marker_path))
        xform.SetTranslate((float(goal[0]), float(goal[1]), float(goal[2])))

def _get_range(cfg, key, default):
    raw = cfg.task.get("spawn_ranges", {}).get(key, default)
    lo, hi = float(raw[0]), float(raw[1])
    return (min(lo, hi), max(lo, hi))

def _draw_cross(draw, center, size=0.2, color=(1.0, 1.0, 1.0, 1.0)):
    if draw is None:
        return
    cx, cy, cz = center.tolist()
    pts = [
        (cx - size, cy, cz),
        (cx + size, cy, cz),
        (cx, cy - size, cz),
        (cx, cy + size, cz),
        (cx, cy, cz - size),
        (cx, cy, cz + size),
    ]
    draw.plot(torch.tensor(pts, device="cpu"), size=2.0, color=color)


def _safe_write_video(path, frames, fps):
    if not frames:
        return
    try:
        import imageio.v3 as iio  # type: ignore

        iio.imwrite(path, frames, fps=fps)
        return
    except Exception:
        pass
    try:
        import imageio  # type: ignore

        with imageio.get_writer(path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return
    except Exception:
        pass


@hydra.main(version_base=None, config_path="../cfg", config_name="manual")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Start Isaac FIRST
    from bluerov_manual.sim_app import init_app
    sim_app = init_app(cfg)

    # Import AFTER app start
    from marinegym.envs.isaac_env import IsaacEnv
    from bluerov_manual.envs.hover_tank import HoverTank  # noqa: F401

    env_class = IsaacEnv.REGISTRY["HoverTank"]
    base_env = env_class(cfg, headless=cfg.headless)

    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(getattr(cfg, "seed", 0))

    viz_cfg = cfg.task.get("visualization", {}) or {}
    viz_enabled = bool(viz_cfg.get("enable", True))
    viz_every = int(viz_cfg.get("draw_every", 5))
    viz_trail_len = int(viz_cfg.get("trail_length", 200))
    viz_draw_path = bool(viz_cfg.get("draw_global_path", True))
    viz_draw_local = bool(viz_cfg.get("draw_local_target", True))
    viz_draw_trail = bool(viz_cfg.get("draw_trail", True))
    local_trail = []

    video_cfg = cfg.task.get("record_video", {}) or {}
    video_enabled = bool(video_cfg.get("enable", False))
    video_interval = int(video_cfg.get("interval", 2))
    video_fps = int(video_cfg.get("fps", 30))
    video_path = str(video_cfg.get("output", "manual_run.mp4"))
    video_max_frames = int(video_cfg.get("max_frames", 2000))
    video_frames = []

    debug_cfg = cfg.task.get("debug", {}) or {}
    debug_enabled = bool(debug_cfg.get("enable", False))
    debug_every = int(debug_cfg.get("print_every", 10))

    stack_cfg = cfg.task.get("planner_stack", {}) or {}
    planner_stack_enabled = bool(stack_cfg.get("enable", False))
    if planner_stack_enabled:
        rrt_cfg = stack_cfg.get("rrt", {}) or {}
        bounds_min = tuple(rrt_cfg.get("bounds_min", [-10.0, -10.0, -5.0]))
        bounds_max = tuple(rrt_cfg.get("bounds_max", [10.0, 10.0, 5.0]))
        planner = GlobalPlanner(
            GlobalPlannerConfig(
                fixed_depth=rrt_cfg.get("fixed_depth", None),
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                step_size=float(rrt_cfg.get("step_size", 1.0)),
                max_iters=int(rrt_cfg.get("max_iters", 400)),
                goal_sample_rate=float(rrt_cfg.get("goal_sample_rate", 0.1)),
                neighbor_radius=float(rrt_cfg.get("neighbor_radius", 2.0)),
                goal_tolerance=float(rrt_cfg.get("goal_tolerance", 1.0)),
            )
        )
        sonar_cfg = stack_cfg.get("sonar", {}) or {}
        sonar = Sonar(
            SonarConfig(
                num_rays=int(sonar_cfg.get("num_rays", 16)),
                fov_deg=float(sonar_cfg.get("fov_deg", 180.0)),
                max_range=float(sonar_cfg.get("max_range", 10.0)),
                include_angle_encoding=bool(sonar_cfg.get("include_angle_encoding", True)),
                min_range_filter=float(sonar_cfg.get("min_range_filter", 0.2)),
            ),
            device=env.device,
        )
        policy_cfg = stack_cfg.get("policy", {}) or {}
        local_policy = LocalPolicy(
            LocalPolicyConfig(
                min_range_stop=float(policy_cfg.get("min_range_stop", 0.5)),
                min_range_replan=float(policy_cfg.get("min_range_replan", 0.8)),
                min_range_avoid=float(policy_cfg.get("min_range_avoid", 1.2)),
            ),
            device=env.device,
        )
        fusion_cfg = stack_cfg.get("fusion", {}) or {}
        behavior_fusion = BehaviorFusion(
            BehaviorFusionConfig(
                slow_scale=float(fusion_cfg.get("slow_scale", 0.3)),
                avoidance_offset=float(fusion_cfg.get("avoidance_offset", 1.0)),
                max_speed=float(fusion_cfg.get("max_speed", 0.3)),
                max_vert_speed=float(fusion_cfg.get("max_vert_speed", 0.1)),
            ),
            device=env.device,
        )
        obstacles = ObstacleField(ObstacleFieldConfig(), device=env.device)
    else:
        planner = sonar = local_policy = behavior_fusion = obstacles = None

    td = env.reset()

    ctrl_cfg = cfg.task.get("controller", {}) or {}
    # Build allocation matrix (views valid after reset)
    base_env.drone.get_state()
    thrust_axis = int(ctrl_cfg.get("thrust_axis", 0))
    thrust_sign = float(ctrl_cfg.get("thrust_sign", 1.0))
    A = build_allocation_matrix_from_usd(
        base_env.drone, device=env.device, thrust_axis=thrust_axis, thrust_sign=thrust_sign
    )  # (6,N)
    A_pinv = torch.linalg.pinv(A)                                            # (N,6)
    N = A.shape[1]

    # Estimate per-thruster max thrust
    fc = float(base_env.drone.FORCE_CONSTANTS_0[0].item())
    rpm_max = 3900.0
    f_max = torch.tensor(fc * (rpm_max ** 2), device=env.device)             # scalar

    x_range = _get_range(cfg, "x_range", (0.2, 20.0))
    y_range = _get_range(cfg, "y_range", (0.2, 6.0))
    z_range = _get_range(cfg, "z_range", (0.0, -1.5))

    # Random goals per env (WORLD frame)
    goals = sample_random_goals(
        env.num_envs,
        env.device,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    )

    from omni.isaac.core.utils import stage as stage_utils
    stage = stage_utils.get_current_stage()
    marker_paths = _ensure_goal_markers(stage, base_env.envs_prim_paths, radius=0.6, height=0.03)
    _update_goal_markers(stage, marker_paths, goals)

    base_env.drone.get_state()
    start_pos = base_env.drone.pos[:, 0, :].detach().cpu().tolist()
    print(f"start_pos[0:{min(5, env.num_envs)}]={start_pos[:min(5, env.num_envs)]}")
    print(f"goals[0:{min(5, env.num_envs)}]={goals[:min(5, env.num_envs)].detach().cpu().tolist()}")

    # Stop condition
    goal_tol = 0.15

    # Track which envs reached
    reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    step_i = 0
    success_count = 0
    session_cfg = cfg.task.get("session", {}) or {}
    continue_on_success = bool(session_cfg.get("continue_on_success", True))
    max_successes = int(session_cfg.get("max_successes", 0))

    print(f"num_envs={env.num_envs} thrusters={N} f_max={float(f_max.item()):.6f}")
    print(f"goals[0: min(5,E)]={goals[: min(5, env.num_envs)].detach().cpu().tolist()}")

    if planner_stack_enabled:
        base_env.drone.get_state()
        pos_w = base_env.drone.pos[:, 0, :]
        paths = []
        path_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        for env_idx in range(env.num_envs):
            if getattr(base_env, "enable_obstacles", False):
                obst_pos_w, _ = base_env.obstacles.get_world_poses()
                obstacles.update_from_positions(
                    obst_pos_w[env_idx],
                    radius=float(cfg.task.get("obstacles", {}).get("radius", 0.5)),
                )
            plan = planner.plan(
                start_w=pos_w[env_idx].tolist(),
                goal_w=goals[env_idx].tolist(),
                obstacles=obstacles.get_obstacles(),
            )
            paths.append(plan)

    prev_action = torch.zeros(env.num_envs, 1, N, device=env.device)
    ctrl_cfg = cfg.task.get("controller", {}) or {}
    alloc_update_every = int(ctrl_cfg.get("allocation_update_every", 1))
    action_smooth_alpha = float(ctrl_cfg.get("action_smooth_alpha", 0.8))

    while True:
        step_i += 1

        # Update state buffers
        base_env.drone.get_state()
        if alloc_update_every > 0 and (step_i % alloc_update_every == 0):
            A = build_allocation_matrix_from_usd(
                base_env.drone,
                device=env.device,
                thrust_axis=thrust_axis,
                thrust_sign=thrust_sign,
            )
            A_pinv = torch.linalg.pinv(A)
            N = A.shape[1]

        if planner_stack_enabled:
            pos_w = base_env.drone.pos[:, 0, :]
            rot_w = base_env.drone.rot[:, 0, :]
            vel_w = base_env.drone.get_velocities(True)[:, 0, :3]
            obs = sonar.scan(pos_w, rot_w)
            obs["vel_w"] = vel_w

            waypoints = []
            for env_idx in range(env.num_envs):
                idx = int(path_indices[env_idx].item())
                path = paths[env_idx]
                idx = min(idx, len(path) - 1)
                waypoints.append(path[idx])
            waypoint_w = torch.tensor(waypoints, device=env.device)
            goal_dir = waypoint_w - pos_w
            dist_to_wp = torch.norm(goal_dir, dim=-1, keepdim=True)
            obs["goal_dir"] = goal_dir / dist_to_wp.clamp_min(1e-6)
            obs["dist_to_wp"] = dist_to_wp

            behavior = local_policy.decide(obs)

            min_ranges, _ = torch.min(obs["ranges"], dim=1)
            safety_cfg = stack_cfg.get("safety", {}) if isinstance(stack_cfg, dict) else {}
            safety_min_range = float(safety_cfg.get("min_range", 0.6))
            safety_max_speed = float(safety_cfg.get("max_speed", 0.5))
            speed = torch.norm(vel_w, dim=-1)
            safety_stop = (min_ranges < safety_min_range) & (speed > safety_max_speed)
            behavior = torch.where(
                safety_stop,
                torch.tensor(int(BehaviorAction.STOP), device=env.device),
                behavior,
            )

            replan = behavior == int(BehaviorAction.REQUEST_REPLAN)
            if replan.any():
                for env_idx in torch.where(replan)[0].tolist():
                    if getattr(base_env, "enable_obstacles", False):
                        obst_pos_w, _ = base_env.obstacles.get_world_poses()
                        obstacles.update_from_positions(
                            obst_pos_w[env_idx],
                            radius=float(cfg.task.get("obstacles", {}).get("radius", 0.5)),
                        )
                    paths[env_idx] = planner.plan(
                        start_w=pos_w[env_idx].tolist(),
                        goal_w=goals[env_idx].tolist(),
                        obstacles=obstacles.get_obstacles(),
                    )
                    path_indices[env_idx] = 0

            local_cmd = behavior_fusion.to_local_target(behavior, waypoint_w, pos_w)

            waypoint_reached = dist_to_wp.squeeze(-1) < 0.5
            if waypoint_reached.any():
                for env_idx in torch.where(waypoint_reached)[0].tolist():
                    new_idx = min(
                        int(path_indices[env_idx].item()) + 1, len(paths[env_idx]) - 1
                    )
                    path_indices[env_idx] = new_idx

        # Controller: desired body wrench (E,6)
        target_pos = local_cmd["target_pos_w"] if planner_stack_enabled else goals
        target_vel = local_cmd["target_vel_w"] if planner_stack_enabled else None
        target_yaw = local_cmd["target_yaw"] if planner_stack_enabled else None
        ctrl_cfg = cfg.task.get("controller", {}) or {}
        wrench_b = wrench_pd_world_to_body(
            base_env.drone,
            target_pos,
            kp=float(ctrl_cfg.get("kp", 25.0)),
            kd=float(ctrl_cfg.get("kd", 12.0)),
            target_vel_w=target_vel,
            target_yaw=target_yaw,
            yaw_kp=float(stack_cfg.get("fusion", {}).get("yaw_kp", 2.0)),
            yaw_kd=float(stack_cfg.get("fusion", {}).get("yaw_kd", 0.2)),
            gravity_comp=bool(ctrl_cfg.get("gravity_comp", True)),
            gravity_scale=float(ctrl_cfg.get("gravity_scale", 1.0)),
            max_accel=ctrl_cfg.get("max_accel", None),
            max_force=ctrl_cfg.get("max_force", None),
            max_torque=ctrl_cfg.get("max_torque", None),
            stabilize_rp=bool(ctrl_cfg.get("stabilize_rp", True)),
            rp_kp=float(ctrl_cfg.get("rp_kp", 2.0)),
            rp_kd=float(ctrl_cfg.get("rp_kd", 0.2)),
        )

        # Wrench -> thruster forces: f = wrench_b @ A_pinv.T  => (E,N)
        f = wrench_b @ A_pinv.T

        # Forces -> action in expected shape (E,1,N)
        action = forces_to_action(f, f_max).view(env.num_envs, 1, N)
        action = action_smooth_alpha * prev_action + (1.0 - action_smooth_alpha) * action
        prev_action = action

        # Step
        td.set(("agents", "action"), action)
        td = env.step(td)

        if video_enabled and (step_i % video_interval == 0) and (len(video_frames) < video_max_frames):
            frame = base_env.render(mode="rgb_array")
            video_frames.append(frame)

        if viz_enabled and base_env.debug_draw is not None and (step_i % viz_every == 0):
            draw = base_env.debug_draw
            draw.clear()
            env_idx = int(base_env.central_env_idx)
            pos_w = base_env.drone.pos[env_idx, 0, :].detach().cpu()

            if planner_stack_enabled and viz_draw_path:
                path = paths[env_idx]
                if len(path) >= 2:
                    pts = torch.tensor(path, device="cpu")
                    draw.plot(pts, size=2.0, color=(0.2, 1.0, 0.2, 1.0))

            if planner_stack_enabled and viz_draw_local:
                tgt = local_cmd["target_pos_w"][env_idx].detach().cpu()
                _draw_cross(draw, tgt, size=0.3, color=(1.0, 0.2, 0.2, 1.0))
                draw.vector(pos_w.unsqueeze(0), (tgt - pos_w).unsqueeze(0), size=2.0, color=(1.0, 0.2, 0.2, 1.0))

            if viz_draw_trail:
                local_trail.append(pos_w.tolist())
                if len(local_trail) > viz_trail_len:
                    local_trail = local_trail[-viz_trail_len:]
                if len(local_trail) >= 2:
                    draw.plot(torch.tensor(local_trail, device="cpu"), size=1.0, color=(0.2, 0.6, 1.0, 1.0))

        # Reward/done
        r, done = read_reward_done(td)
        r_mean = float(r.mean().item())
        done_any = bool(done.any().item())

        # Distances to goals
        pos_w = base_env.drone.pos[:, 0, :]  # (E,3)
        dists = torch.norm(goals - pos_w, dim=-1)  # (E,)

        # Update reached mask
        newly = (dists < goal_tol) & (~reached)
        if newly.any():
            reached = reached | newly

        reached_count = int(reached.sum().item())

        # Print a compact per-step summary + a few env samples
        if step_i % 10 == 1 or newly.any():
            sample_k = min(3, env.num_envs)
            pos_s = pos_w[:sample_k].detach().cpu().tolist()
            dist_s = dists[:sample_k].detach().cpu().tolist()
            act_s = action[:sample_k, 0, :].detach().cpu().tolist()

            print(
                f"step={step_i} reward_mean={r_mean:.4f} "
                f"reached={reached_count}/{env.num_envs} done_any={done_any}\n"
                f"  pos[0:{sample_k}]={pos_s}\n"
                f"  dist[0:{sample_k}]={[round(x,3) for x in dist_s]}\n"
                f"  action[0:{sample_k}]={act_s}"
            )
        if debug_enabled and planner_stack_enabled and (step_i % debug_every == 0):
            env_idx = 0
            pos = base_env.drone.pos[env_idx, 0, :]
            tgt = local_cmd["target_pos_w"][env_idx]
            delta = tgt - pos
            norm = torch.norm(delta).clamp_min(1e-6)
            dir_unit = (delta / norm).detach().cpu().tolist()
            behavior_id = int(local_cmd["behavior"][env_idx].item())
            min_range = float(obs["ranges"][env_idx].min().item())
            behavior_names = {
                0: "FOLLOW",
                1: "AVOID_LEFT",
                2: "AVOID_RIGHT",
                3: "SLOW",
                4: "STOP",
                5: "REPLAN",
            }
            print(
                f"debug step={step_i} env={env_idx} behavior={behavior_id} "
                f"({behavior_names.get(behavior_id, 'UNK')}) "
                f"min_range={min_range:.3f} dir_unit={dir_unit} "
                f"target_pos={tgt.detach().cpu().tolist()}"
            )

        # Stop when all envs reached their goals
        if reached.all():
            success_count += 1
            print(f"All envs reached goals. steps={step_i} successes={success_count}")
            if not continue_on_success or (max_successes > 0 and success_count >= max_successes):
                break
            td = env.reset()
            reached.zero_()
            goals = sample_random_goals(
                env.num_envs,
                env.device,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
            )
            _update_goal_markers(stage, marker_paths, goals)
            if planner_stack_enabled:
                pos_w = base_env.drone.pos[:, 0, :]
                paths = []
                path_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
                for env_idx in range(env.num_envs):
                    if getattr(base_env, "enable_obstacles", False):
                        obst_pos_w, _ = base_env.obstacles.get_world_poses()
                        obstacles.update_from_positions(
                            obst_pos_w[env_idx],
                            radius=float(cfg.task.get("obstacles", {}).get("radius", 0.5)),
                        )
                    plan = planner.plan(
                        start_w=pos_w[env_idx].tolist(),
                        goal_w=goals[env_idx].tolist(),
                        obstacles=obstacles.get_obstacles(),
                    )
                    paths.append(plan)
            continue

        # If episode ends, reset and continue (keep the same goals by default)
        if done_any:
            td = env.reset()
            reached.zero_()  # optional: reset reached if reset happens
            # optional: resample goals on reset
            goals = sample_random_goals(
                env.num_envs,
                env.device,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
            )
            _update_goal_markers(stage, marker_paths, goals)
            base_env.drone.get_state()
            start_pos = base_env.drone.pos[:, 0, :].detach().cpu().tolist()
            print(f"start_pos[0:{min(5, env.num_envs)}]={start_pos[:min(5, env.num_envs)]}")
            print(f"goals[0:{min(5, env.num_envs)}]={goals[:min(5, env.num_envs)].detach().cpu().tolist()}")
            if planner_stack_enabled:
                pos_w = base_env.drone.pos[:, 0, :]
                paths = []
                path_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
                for env_idx in range(env.num_envs):
                    if getattr(base_env, "enable_obstacles", False):
                        obst_pos_w, _ = base_env.obstacles.get_world_poses()
                        obstacles.update_from_positions(
                            obst_pos_w[env_idx],
                            radius=float(cfg.task.get("obstacles", {}).get("radius", 0.5)),
                        )
                    plan = planner.plan(
                        start_w=pos_w[env_idx].tolist(),
                        goal_w=goals[env_idx].tolist(),
                        obstacles=obstacles.get_obstacles(),
                    )
                    paths.append(plan)
            print("Episode reset -> resampled goals.")

    if video_enabled:
        _safe_write_video(video_path, video_frames, video_fps)
    sim_app.close()


if __name__ == "__main__":
    main()
