import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import hydra
import torch
from omegaconf import OmegaConf

from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

from bluerov_manual.navigation import (
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

def build_allocation_matrix_from_usd(drone, device):
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
    d_w = quat_axis(q_w, axis=0)  # (N,3)
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


def wrench_pd_world_to_body(drone, goals_w, kp=25.0, kd=12.0):
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
    e_v = -linvel_w

    F_des_w = kp * e_p + kd * e_v                      # (E,3)
    F_des_b = quat_rotate_inverse(rot_w, F_des_w)      # (E,3)

    T_des_b = torch.zeros(pos_w.shape[0], 3, device=pos_w.device)
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

    stack_cfg = cfg.task.get("planner_stack", {}) or {}
    planner_stack_enabled = bool(stack_cfg.get("enable", False))
    if planner_stack_enabled:
        planner = GlobalPlanner(GlobalPlannerConfig())
        sonar = Sonar(SonarConfig(), device=env.device)
        local_policy = LocalPolicy(LocalPolicyConfig(), device=env.device)
        behavior_fusion = BehaviorFusion(BehaviorFusionConfig(), device=env.device)
        obstacles = ObstacleField(ObstacleFieldConfig(), device=env.device)
    else:
        planner = sonar = local_policy = behavior_fusion = obstacles = None

    td = env.reset()

    # Build allocation matrix (views valid after reset)
    base_env.drone.get_state()
    A = build_allocation_matrix_from_usd(base_env.drone, device=env.device)  # (6,N)
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

    print(f"num_envs={env.num_envs} thrusters={N} f_max={float(f_max.item()):.6f}")
    print(f"goals[0: min(5,E)]={goals[: min(5, env.num_envs)].detach().cpu().tolist()}")

    while True:
        step_i += 1

        # Update state buffers
        base_env.drone.get_state()

        if planner_stack_enabled:
            pos_w = base_env.drone.pos[:, 0, :]
            rot_w = base_env.drone.rot[:, 0, :]
            obs = sonar.scan(pos_w, rot_w)
            behavior = local_policy.decide(obs)
            waypoints = []
            for env_idx in range(env.num_envs):
                plan = planner.plan(
                    start_w=pos_w[env_idx].tolist(),
                    goal_w=goals[env_idx].tolist(),
                    obstacles=obstacles.get_obstacles(),
                )
                waypoints.append(plan[0])
            waypoint_w = torch.tensor(waypoints, device=env.device)
            _ = behavior_fusion.to_local_target(behavior, waypoint_w, pos_w)

        # Controller: desired body wrench (E,6)
        wrench_b = wrench_pd_world_to_body(base_env.drone, goals, kp=25.0, kd=12.0)

        # Wrench -> thruster forces: f = wrench_b @ A_pinv.T  => (E,N)
        f = wrench_b @ A_pinv.T

        # Forces -> action in expected shape (E,1,N)
        action = forces_to_action(f, f_max).view(env.num_envs, 1, N)

        # Step
        td.set(("agents", "action"), action)
        td = env.step(td)

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

        # Stop when all envs reached their goals
        if reached.all():
            print(f"All envs reached goals. steps={step_i}")
            break

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
            print("Episode reset -> resampled goals.")

    sim_app.close()


if __name__ == "__main__":
    main()
