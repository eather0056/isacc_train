import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import hydra
import torch
from omegaconf import OmegaConf

from tensordict import TensorDict
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


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



def goto_goal_wrench_pd(drone, goal_pos_w, kp=25.0, kd=12.0):
    """
    Simple position PD controller producing desired wrench in BODY frame.
    Returns (6,) tensor: [Fx,Fy,Fz,Tx,Ty,Tz] in BODY.
    """
    from marinegym.utils.torch import quat_rotate_inverse

    pos_w = drone.pos[0, 0]  # (3,)
    rot_w = drone.rot[0, 0]  # (4,)

    vel_w6 = drone.get_velocities(True)[0, 0]  # (6,)
    linvel_w = vel_w6[:3]

    e_p = goal_pos_w - pos_w
    e_v = -linvel_w

    F_des_w = kp * e_p + kd * e_v
    F_des_b = quat_rotate_inverse(rot_w.unsqueeze(0), F_des_w.unsqueeze(0)).squeeze(0)


    T_des_b = torch.zeros(3, device=F_des_b.device)

    return torch.cat([F_des_b, T_des_b], dim=0)


def thruster_forces_to_action(f, f_max):
    """
    Linear map force->action in [-1,1]. Tune f_max if needed.
    """
    return torch.clamp(f / f_max, -1.0, 1.0)


@hydra.main(version_base=None, config_path="../cfg", config_name="manual")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    

    # Start Isaac FIRST
    from bluerov_manual.sim_app import init_app
    sim_app = init_app(cfg)
    print(OmegaConf.to_yaml(cfg))
    print("cfg.task keys:", list(cfg.task.keys()))
    print("tank path:", cfg.task.get("tank_usd_path", None))


    # Import AFTER app start (good)
    from marinegym.envs.isaac_env import IsaacEnv

    # Import your env so it registers in IsaacEnv.REGISTRY
    from bluerov_manual.envs.hover_tank import HoverTank  # noqa: F401

    env_class = IsaacEnv.REGISTRY["HoverTank"]
    # env_class = HoverTank


    base_env = env_class(cfg, headless=cfg.headless)
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    print("tank valid:", stage.GetPrimAtPath("/World/envs/env_0/tank").IsValid())


    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(getattr(cfg, "seed", 0))

    td = env.reset()

    # Goal in WORLD coordinates (Hover env uses target_pos around z=2)
    goal = torch.tensor([20.0, 0.0, -20.0], device=env.device)

    # Build allocation matrix after first reset (views are valid)
    # Ensure drone state is updated at least once
    base_env.drone.get_state()
    A = build_allocation_matrix_from_usd(base_env.drone, device=env.device)
    A_pinv = torch.linalg.pinv(A)  # (N,6)

    # Estimate per-thruster max thrust from yaml constants (KF * rpm^2)
    # BlueROV.yaml: force_constants=4.4e-7, max_rotation_velocities=3900
    # If you changed these, update here or load from params.
    fc = float(base_env.drone.FORCE_CONSTANTS_0[0].item())
    rpm_max = 3900.0
    f_max = torch.tensor(fc * (rpm_max ** 2), device=env.device)

    print("A shape:", tuple(A.shape), "f_max:", float(f_max.item()))

    while True:
        # Update drone state buffers
        base_env.drone.get_state()

        # Desired wrench (BODY)
        wrench_b = goto_goal_wrench_pd(base_env.drone, goal, kp=25.0, kd=12.0)  # (6,)

        # Solve thruster forces f (N,)
        f = (A_pinv @ wrench_b).reshape(-1)

        # Convert to action [-1,1]
        action_vec = thruster_forces_to_action(f, f_max)        # (6,)
        action = action_vec.view(1, 1, -1).expand(env.num_envs, 1, -1)  # (num_envs,1,6)


        # Step with rolling tensordict
        td.set(("agents", "action"), action)
        td = env.step(td)

        # Read reward/done regardless of "next" convention
        if ("agents", "reward") in td.keys(True, True):
            r = td[("agents", "reward")].mean().item()
            done = bool(td["done"].any().item())
        else:
            r = td[("next", "agents", "reward")].mean().item()
            done = bool(td[("next", "done")].any().item())

        # Position and distance to goal
        pos = base_env.drone.pos[0, 0].detach()
        dist = torch.norm(goal - pos).item()

        print(f"reward={r:.4f} pos={pos.cpu().tolist()} dist={dist:.3f} action={action_vec.detach().cpu().tolist()}")

        # Stop when close enough
        if dist < 0.15:
            print("Reached goal.")
            break

        # Reset if episode ends
        if done:
            td = env.reset()

    sim_app.close()


if __name__ == "__main__":
    main()
