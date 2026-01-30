import logging
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf

from setproctitle import setproctitle
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@hydra.main(version_base=None, config_path="../cfg", config_name="train_behavior")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from bluerov_manual.sim_app import init_app
    sim_app = init_app(cfg)

    # Import MarineGym/Isaac modules only after SimulationApp is created.
    from marinegym.utils.torchrl import SyncDataCollector
    from marinegym.utils.torchrl import RenderCallback, EpisodeStats
    from marinegym.learning import ALGOS
    import bluerov_manual.envs  # noqa: F401
    from marinegym.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    print(f"[train_ppo_behavior] task={cfg.task.name} env_class={env_class.__name__}")
    base_env = env_class(cfg, headless=cfg.headless)

    wandb_run = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg and wandb_cfg.get("enable", False):
        try:
            import wandb
            run_name = wandb_cfg.get("name", None)
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "bluerov_manual"),
                name=run_name,
                tags=wandb_cfg.get("tags", []),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as exc:
            print(f"[train_ppo_behavior] wandb init failed: {exc}")

    transforms = [InitTracker()]
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device
    )

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    save_interval = int(cfg.get("save_interval", -1))
    checkpoint_dir = cfg.get("checkpoint_dir", None)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(seed: int = 0, exploration_type: ExplorationType = ExplorationType.MODE):
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)
        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            _ = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()
        return {}

    pbar = iter(collector)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(policy.train_op(data.to_tensordict()))

        # Extra diagnostics for W&B (guarded to avoid shape issues).
        try:
            reward_mean = data["next", "agents", "reward"].mean().item()
            info["train/reward_mean"] = reward_mean
        except Exception:
            pass
        try:
            actions = data["agents", "action"]
            sat = (actions.abs() - 1.0).clamp_min(0.0).mean().item()
            info["train/action_saturation"] = sat
        except Exception:
            pass
        try:
            done_rate = data["next", "done"].float().mean().item()
            info["train/done_rate"] = done_rate
        except Exception:
            pass
        # Environment-derived metrics (if available)
        if hasattr(base_env, "_last_yaw_err") and base_env._last_yaw_err is not None:
            try:
                info["train/yaw_err_mean_abs"] = base_env._last_yaw_err.abs().mean().item()
            except Exception:
                pass
        if hasattr(base_env, "_path_idx") and hasattr(base_env, "_paths"):
            try:
                info["train/path_idx_mean"] = base_env._path_idx.float().mean().item()
                path_lens = torch.tensor([len(p) if p is not None else 0 for p in base_env._paths], device=base_env.device)
                info["train/path_len_mean"] = path_lens.float().mean().item()
            except Exception:
                pass
        if hasattr(base_env, "_waypoint_tol"):
            try:
                info["train/wp_tol"] = float(base_env._waypoint_tol)
            except Exception:
                pass
        if hasattr(base_env, "_current_waypoints"):
            try:
                pos_w = base_env.drone.pos[:, 0, :]
                wp = base_env._current_waypoints()
                info["train/wp_dist_mean"] = torch.norm(wp - pos_w, dim=-1).mean().item()
            except Exception:
                pass
        if hasattr(base_env, "_last_occ_grid") and base_env._last_occ_grid is not None:
            try:
                import wandb
                occ = base_env._last_occ_grid[0].detach().cpu().numpy()
                info["viz/occ_grid"] = wandb.Image(occ, caption="occ_grid_env0")
            except Exception:
                pass
        if hasattr(base_env, "_last_action_clamped") and base_env._last_action_clamped is not None:
            try:
                rpy = base_env.drone.rot[:, 0, :]
                rpy = base_env.drone.rot[:, 0, :]
                # Use env's helper for roll/pitch magnitude if available.
                from marinegym.utils.torch import quaternion_to_euler
                eul = quaternion_to_euler(rpy)
                info["train/tilt_mean_deg"] = (eul[:, :2].abs().mean() * (180.0 / torch.pi)).item()
            except Exception:
                pass
        if hasattr(base_env, "_last_target_vel_w") and base_env._last_target_vel_w is not None:
            try:
                info["train/target_speed_mean"] = base_env._last_target_vel_w.norm(dim=-1).mean().item()
            except Exception:
                pass
        if hasattr(base_env, "_last_sonar") and base_env._last_sonar is not None:
            try:
                info["train/min_range_mean"] = base_env._last_sonar.min(dim=1).values.mean().item()
            except Exception:
                pass
        if wandb_run is not None:
            wandb_run.log(info, step=int(collector._frames))
        if i % 10 == 0:
            reward_mean = data["next", "agents", "reward"].mean().item()
            progress = 0.0
            eta = None
            if total_frames > 0:
                progress = 100.0 * (collector._frames / total_frames)
                if collector._fps > 1e-6:
                    eta = (total_frames - collector._frames) / collector._fps
            info["train/progress_pct"] = progress
            if eta is not None:
                info["train/eta_seconds"] = float(eta)
            if eta is not None:
                print(
                    f"iter={i} frames={collector._frames} reward_mean={reward_mean:.4f} "
                    f"progress={progress:.1f}% eta={eta:.1f}s"
                )
            else:
                print(f"iter={i} frames={collector._frames} reward_mean={reward_mean:.4f}")

        if save_interval > 0 and i % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ppo_behavior_iter_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[train_ppo_behavior] saved checkpoint: {ckpt_path}")

        if max_iters > 0 and i >= max_iters - 1:
            break

    final_ckpt = os.path.join(checkpoint_dir, "ppo_behavior_final.pt")
    torch.save(policy.state_dict(), final_ckpt)
    print(f"[train_ppo_behavior] saved final checkpoint: {final_ckpt}")

    if wandb_run is not None:
        wandb_run.finish()

    sim_app.close()


if __name__ == "__main__":
    main()
