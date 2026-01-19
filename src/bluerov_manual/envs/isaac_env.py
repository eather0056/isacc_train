# src/bluerov_manual/isaac_env.py
#
# Minimal Isaac Sim environment base class for your "bluerov_manual" project.
# - Supports N cloned env instances under /World/envs/env_i
# - Provides reset()/step() loop suitable for manual control
# - Delays omni.isaac imports until AFTER SimulationApp is created
#
# Usage pattern (important):
#   1) Start SimulationApp first (in your script via sim_app.init_app(cfg))
#   2) Then import/construct this environment

from __future__ import annotations

import abc
from typing import Callable, List, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
import torch
from bluerov_manual.isaac_env import IsaacEnv



class IsaacEnvBase(abc.ABC):
    """
    Minimal Isaac Sim environment wrapper:
    - Creates SimulationContext
    - Builds template scene in /World/envs/env_0 via _design_scene()
    - Clones env_0 into env_0..env_{num_envs-1} via GridCloner
    - Steps physics with optional rendering
    - Provides reset()/step() that return a dict of tensors
    """

    env_ns = "/World/envs"
    template_env_ns = "/World/envs/env_0"

    def __init__(self, cfg, headless: bool):
        self.cfg = cfg
        self.headless = bool(headless)

        # Torch device used for tensors (NOT necessarily Isaac simulation device)
        self.device = torch.device(cfg.sim.device)

        # Common parameters
        self.num_envs = int(cfg.env.num_envs)
        self.max_episode_length = int(cfg.env.max_episode_length)
        self.substeps = int(cfg.sim.substeps)
        self.dt_cfg = float(cfg.sim.dt)

        # Render control (function of substep -> bool)
        self.enable_viewport = True
        self.enable_render((not self.headless) or bool(getattr(cfg, "enable_livestream", False)))

        # IMPORTANT: omni imports must happen after SimulationApp exists.
        self._lazy_import_isaac()

        # Stage must exist
        if self.stage_utils.get_current_stage() is None:
            raise RuntimeError(
                "USD stage not created. Ensure SimulationApp is instantiated before creating the env."
            )

        # Build sim params as plain dict (avoid OmegaConf struct issues)
        sim_params = self._to_plain_dict(cfg.sim)
        if "physx" in sim_params:
            physx_params = sim_params.pop("physx")
            if isinstance(physx_params, dict):
                sim_params.update(physx_params)

        # Configure extensions & flags
        self._configure_simulation_flags(sim_params)

        # Create SimulationContext
        self.sim = self.SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self.dt_cfg,
            rendering_dt=self.dt_cfg,
            backend="torch",
            sim_params=sim_params,
            physics_prim_path="/physicsScene",
            device="cpu",
        )
        self.dt = float(self.sim.get_physics_dt())

        # Viewport / render product (only if enabled)
        self._rgb_annotator = None
        self._render_product = None
        self._create_viewport_render_product()

        # Set up cloner and template env prim
        cloner = self.GridCloner(spacing=float(cfg.env.env_spacing))
        cloner.define_base_env(self.env_ns)

        if not self.prim_utils.is_prim_path_valid(self.template_env_ns):
            self.prim_utils.define_prim(self.template_env_ns)

        # Create template scene (subclass)
        global_prim_paths = self._design_scene() or []

        # Clone template into envs
        self.envs_prim_paths = cloner.generate_paths(self.env_ns + "/env", self.num_envs)
        self.envs_positions = cloner.clone(
            source_prim_path=self.template_env_ns,
            prim_paths=self.envs_prim_paths,
            replicate_physics=bool(getattr(cfg.sim, "replicate_physics", True)),
        )
        self.envs_positions = torch.tensor(
            self.envs_positions, dtype=torch.float32, device=self.device
        )

        # Camera aimed at env closest to origin
        if hasattr(cfg, "viewer") and self.enable_viewport:
            self._set_camera_to_central_env()

        # Collision filtering between env instances
        physics_scene_path = self.sim.get_physics_context().prim_path
        cloner.filter_collisions(
            physics_scene_path,
            "/World/collisions",
            prim_paths=self.envs_prim_paths,
            global_paths=global_prim_paths,
        )

        # Reset sim once after setup
        self.sim.reset()

        # Episode progress buffer
        self.progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

        # Subclass defines observation/action layout; optional
        self._set_specs()

        # Initial reset
        self.reset()

    # ---------- Required subclass hooks ----------

    def _design_scene(self):
        # 1) Tank
        tank_prim = self.prim_utils.create_prim(
            prim_path="/World/envs/env_0/tank",
            usd_path=self.cfg.task.tank_usd_path,
            translation=(0.0, 0.0, 0.0),
        )

        self.kit_utils.set_nested_collision_properties(
            tank_prim.GetPath(),
            collision_enabled=True,
        )
        self.kit_utils.set_nested_rigid_body_properties(
            tank_prim.GetPath(),
            disable_gravity=True,
        )

        # 2) BlueROV
        self.drone, self.controller = self.UnderwaterVehicle.make(
            self.cfg.task.drone_model.name,
            self.cfg.task.drone_model.controller,
        )

        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])

        return []


    @abc.abstractmethod
    def _set_specs(self) -> None:
        """Optionally define action/obs dimensions, etc. Kept for parity with MarineGym."""
        raise NotImplementedError

    @abc.abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset robot + task state for env indices."""
        raise NotImplementedError

    @abc.abstractmethod
    def _pre_sim_step(self, action: torch.Tensor) -> None:
        """
        Apply action to the simulation (e.g. call robot.apply_action(action)).
        action shape should be (num_envs, 1, action_dim) for single-agent.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_obs(self) -> Dict[str, torch.Tensor]:
        """Return observation dict (tensors on self.device) for all envs."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_reward_done(self) -> Dict[str, torch.Tensor]:
        """
        Return dict with keys:
          - reward: (num_envs, 1) or (num_envs, 1, 1)
          - terminated: (num_envs, 1) bool
          - truncated: (num_envs, 1) bool
          - done: (num_envs, 1) bool
        """
        raise NotImplementedError

    # ---------- Public API: reset/step ----------

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(self.device)

        self._reset_idx(env_ids)
        self.progress[env_ids] = 0

        obs = self._compute_obs()
        truncated = (self.progress >= self.max_episode_length).unsqueeze(-1)

        return {
            **obs,
            "truncated": truncated,
            "terminated": torch.zeros_like(truncated, dtype=torch.bool),
            "done": torch.zeros_like(truncated, dtype=torch.bool),
        }

    def step(self, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        # action expected on self.device
        action = action.to(self.device)

        for substep in range(self.substeps):
            self._pre_sim_step(action)
            self.sim.step(self._should_render(substep))

        self.progress += 1

        obs = self._compute_obs()
        rd = self._compute_reward_done()

        # enforce truncation by time limit
        truncated = (self.progress >= self.max_episode_length).unsqueeze(-1)
        if "truncated" in rd:
            rd["truncated"] = rd["truncated"] | truncated
        else:
            rd["truncated"] = truncated

        if "done" in rd:
            rd["done"] = rd["done"] | rd.get("truncated", truncated)
        else:
            rd["done"] = rd.get("terminated", torch.zeros_like(truncated)) | rd["truncated"]

        return {**obs, **rd}

    # ---------- Rendering ----------

    def enable_render(self, enable: Union[bool, Callable[[int], bool]] = True) -> None:
        if isinstance(enable, bool):
            self._should_render = lambda substep: enable
        elif callable(enable):
            self._should_render = enable
        else:
            raise TypeError("enable_render must be bool or callable(substep)->bool")

    def render_rgb(self) -> np.ndarray:
        """
        Return HxWx3 uint8 RGB from viewport (requires viewer config + enable_viewport True).
        """
        if not self.enable_viewport or self._rgb_annotator is None:
            raise RuntimeError("Viewport rendering not available. Ensure viewer config exists and headless is false.")
        rgb_data = self._rgb_annotator.get_data()
        rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        return rgb_data[:, :, :3]

    # ---------- Coordinate helpers ----------

    def get_env_poses(self, world_pos: torch.Tensor) -> torch.Tensor:
        """
        Convert world positions to env-local by subtracting envs_positions.
        Supports shapes:
          - (num_envs, 3)
          - (num_envs, N, 3)
        """
        if world_pos.ndim == 3:
            return world_pos - self.envs_positions.unsqueeze(1)
        return world_pos - self.envs_positions

    def get_world_poses(self, env_pos: torch.Tensor) -> torch.Tensor:
        """
        Convert env-local positions to world by adding envs_positions.
        """
        if env_pos.ndim == 3:
            return env_pos + self.envs_positions.unsqueeze(1)
        return env_pos + self.envs_positions

    # ---------- Cleanup ----------

    def close(self) -> None:
        try:
            self.sim.stop()
        except Exception:
            pass
        try:
            self.sim.clear_all_callbacks()
        except Exception:
            pass
        try:
            self.sim.clear()
        except Exception:
            pass

    # ---------- Internal helpers ----------

    def _lazy_import_isaac(self) -> None:
        # Imports that require SimulationApp to exist
        import carb
        import omni.usd
        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.core.utils import prims as prim_utils, stage as stage_utils
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.viewports import set_camera_view
        import marinegym.utils.kit as kit_utils
        self.kit_utils = kit_utils
        from bluerov_manual.underwater_vehicle import UnderwaterVehicle
        self.UnderwaterVehicle = UnderwaterVehicle



        self.carb = carb
        self.omni_usd = omni.usd
        self.GridCloner = GridCloner
        self.SimulationContext = SimulationContext
        self.prim_utils = prim_utils
        self.stage_utils = stage_utils
        self.enable_extension = enable_extension
        self.set_camera_view = set_camera_view

    def _to_plain_dict(self, node) -> dict:
        # Convert OmegaConf nodes or objects to plain dicts safely
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(node):
                return OmegaConf.to_container(node, resolve=True)
        except Exception:
            pass
        # fall back
        if isinstance(node, dict):
            return dict(node)
        return dict(node.__dict__) if hasattr(node, "__dict__") else {}

    def _configure_simulation_flags(self, sim_params: dict) -> None:
        carb_settings_iface = self.carb.settings.get_settings()

        # Hydra scene-graph instancing
        carb_settings_iface.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)

        # PhysX dispatcher
        carb_settings_iface.set_bool("/physics/physxDispatcher", True)

        # Enable viewport-related extensions (order matters)
        if self.enable_viewport:
            sim_params["enable_scene_query_support"] = True

            self.enable_extension("omni.kit.window.toolbar")
            self.enable_extension("omni.kit.viewport.rtx")
            self.enable_extension("omni.kit.viewport.pxr")
            self.enable_extension("omni.kit.viewport.bundle")
            self.enable_extension("omni.kit.window.status_bar")

        # Replicator (needed for rgb render product)
        self.enable_extension("omni.replicator.isaac")

    def _set_camera_to_central_env(self) -> None:
        # Find env closest to origin
        norms = self.envs_positions.norm(dim=-1)
        idx = int(norms.argmin().item())
        center = self.envs_positions[idx].detach().cpu().numpy()

        eye = np.asarray(self.cfg.viewer.eye, dtype=np.float32)
        lookat = np.asarray(self.cfg.viewer.lookat, dtype=np.float32)

        self.set_camera_view(
            eye=center + eye,
            target=center + lookat,
        )

    def _create_viewport_render_product(self) -> None:
        # viewer config required for viewport
        if not self.enable_viewport:
            return
        if not hasattr(self.cfg, "viewer"):
            return

        # Set camera for the default viewport camera
        self.set_camera_view(
            eye=self.cfg.viewer.eye,
            target=self.cfg.viewer.lookat,
        )

        # Create render product + annotator
        import omni.replicator.core as rep

        self._render_product = rep.create.render_product(
            "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
        )
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._rgb_annotator.attach([self._render_product])
