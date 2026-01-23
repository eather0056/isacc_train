import os
import sys

import hydra
import torch
from omegaconf import OmegaConf

from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

from marinegym.learning import ALGOS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@hydra.main(version_base=None, config_path="../cfg", config_name="play_behavior")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    if cfg.algo.get("checkpoint_path", None) in (None, ""):
        raise RuntimeError("Missing algo.checkpoint_path. Provide a trained checkpoint to play.")

    from bluerov_manual.sim_app import init_app
    sim_app = init_app(cfg)

    import bluerov_manual.envs  # noqa: F401
    from marinegym.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    print(f"[play_ppo_behavior] task={cfg.task.name} env_class={env_class.__name__}")
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]
    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    env.set_seed(cfg.seed)
    base_env.enable_render(not cfg.headless)

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )
    policy.eval()

    num_episodes = int(cfg.get("num_episodes", 1))
    max_steps = int(cfg.get("max_steps", base_env.max_episode_length))

    with set_exploration_type(ExplorationType.MODE):
        if num_episodes <= 0:
            # Continuous play until the app is closed.
            while sim_app.is_running():
                td = env.rollout(
                    max_steps=max_steps,
                    policy=policy,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )
                reward_mean = td["next", "agents", "reward"].mean().item()
                print(f"[play_ppo_behavior] reward_mean={reward_mean:.4f}")
        else:
            for ep in range(num_episodes):
                td = env.rollout(
                    max_steps=max_steps,
                    policy=policy,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )
                reward_mean = td["next", "agents", "reward"].mean().item()
                print(f"[play_ppo_behavior] episode={ep} reward_mean={reward_mean:.4f}")

    sim_app.close()


if __name__ == "__main__":
    main()
