# # src/bluerov_manual/envs/hover_tank.py

# import torch
# import torch.distributions as D
# import omni.isaac.core.utils.prims as prim_utils

# from marinegym.envs.isaac_env import AgentSpec, IsaacEnv
# from marinegym.views import ArticulationView, RigidPrimView
# from marinegym.utils.torch import euler_to_quaternion, quat_axis
# from tensordict.tensordict import TensorDict, TensorDictBase
# from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

# from marinegym.robots.drone import UnderwaterVehicle
# from marinegym.envs.utils import attach_payload
# import os

# import marinegym.utils.kit as kit_utils

# from marinegym.envs.single.hover import Hover  # reuse Hover logic

# class HoverTank(IsaacEnv):
#     """
#     Same as MarineGym Hover, but loads a tank USD into env_0 so it gets cloned.
#     """

#     def __init__(self, cfg, headless):
#         self.reward_effort_weight = cfg.task.reward_effort_weight
#         self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
#         self.reward_distance_scale = cfg.task.reward_distance_scale
#         self.time_encoding = cfg.task.time_encoding
#         self.mode = cfg.mode

#         self.disturbances = cfg.task.get("disturbances", {})
#         self.enable_payload = self.disturbances[self.mode]["payload"]["enable_payload"]
#         self.enable_flow = self.disturbances[self.mode]["flow"]["enable_flow"]
#         self.max_flow_velocity = self.disturbances[self.mode]["flow"]["max_flow_velocity"]
#         self.flow_velocity_gaussian_noise = self.disturbances[self.mode]["flow"]["flow_velocity_gaussian_noise"]

#         # tank path
#         self.tank_usd_path = cfg.task.get("tank_usd_path", None)

#         super().__init__(cfg, headless)

#         self.drone.initialize()

#         if self.enable_payload:
#             payload_cfg = self.disturbances[self.mode]["payload"]
#             self.payload_z_dist = D.Uniform(
#                 torch.tensor([payload_cfg["z"][0]], device=self.device),
#                 torch.tensor([payload_cfg["z"][1]], device=self.device),
#             )
#             self.payload_mass_dist = D.Uniform(
#                 torch.tensor([payload_cfg["mass"][0]], device=self.device),
#                 torch.tensor([payload_cfg["mass"][1]], device=self.device),
#             )
#             self.payload = RigidPrimView(
#                 f"/World/envs/env_*/{self.drone.name}_*/payload",
#                 reset_xform_properties=False,
#                 shape=(-1, self.drone.n),
#             )
#             self.payload.initialize()

#         self.target_vis = ArticulationView("/World/envs/env_*/target", reset_xform_properties=False)
#         self.target_vis.initialize()

#         self.init_poses = self.drone.get_world_poses(clone=True)
#         self.init_vels = torch.zeros_like(self.drone.get_velocities())

#         self.init_pos_dist = D.Uniform(
#             torch.tensor([-2.5, -2.5, 1.5], device=self.device),
#             torch.tensor([2.5, 2.5, 2.5], device=self.device),
#         )
#         self.init_rpy_dist = D.Uniform(
#             torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
#             torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi,
#         )
#         self.target_rpy_dist = D.Uniform(
#             torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
#             torch.tensor([0., 0., 2.], device=self.device) * torch.pi,
#         )

#         self.target_pos = torch.tensor([[0.0, 0.0, 2.0]], device=self.device)
#         self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
#         self.alpha = 0.8

#     def _design_scene(self):
#         # First build the original Hover scene (target + BlueROV + ground plane)
#         global_paths = super()._design_scene()

#         # ---- Load tank into env_0 ----
#         tank_path_cfg = None

#         # Prefer cfg.robot.tank_usd_path (your current config structure)
#         if hasattr(self.cfg, "robot") and "tank_usd_path" in self.cfg.robot:
#             tank_path_cfg = self.cfg.robot.tank_usd_path

#         # Fallback if you later move it to cfg.task.tank_usd_path
#         if tank_path_cfg is None and hasattr(self.cfg, "task") and "tank_usd_path" in self.cfg.task:
#             tank_path_cfg = self.cfg.task.tank_usd_path

#         if tank_path_cfg is None:
#             raise RuntimeError(
#                 "tank_usd_path not found. Put it in cfg.robot.tank_usd_path or cfg.task.tank_usd_path."
#             )

#         tank_usd = os.path.abspath(os.path.expanduser(str(tank_path_cfg)))
#         if not os.path.exists(tank_usd):
#             raise FileNotFoundError(f"Tank USD not found: {tank_usd}")

#         tank_prim_path = "/World/envs/env_0/tank"

#         # Create only once
#         if not prim_utils.is_prim_path_valid(tank_prim_path):
#             tank_prim = prim_utils.create_prim(
#                 prim_path=tank_prim_path,
#                 usd_path=tank_usd,
#                 translation=(0.0, 0.0, 0.0),
#             )

#             # Make tank static + collidable
#             kit_utils.set_nested_collision_properties(tank_prim.GetPath(), collision_enabled=True)
#             kit_utils.set_nested_rigid_body_properties(tank_prim.GetPath(), disable_gravity=True)

#         return global_paths

#     def _set_specs(self):
#         drone_state_dim = self.drone.state_spec.shape[-1]
#         observation_dim = drone_state_dim + 3

#         if self.cfg.task.time_encoding:
#             self.time_encoding_dim = 4
#             observation_dim += self.time_encoding_dim

#         self.observation_spec = CompositeSpec({
#             "agents": CompositeSpec({
#                 "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
#                 "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device),
#             })
#         }).expand(self.num_envs).to(self.device)

#         self.action_spec = CompositeSpec({
#             "agents": CompositeSpec({
#                 "action": self.drone.action_spec.unsqueeze(0),
#             })
#         }).expand(self.num_envs).to(self.device)

#         self.reward_spec = CompositeSpec({
#             "agents": CompositeSpec({
#                 "reward": UnboundedContinuousTensorSpec((1, 1)),
#             })
#         }).expand(self.num_envs).to(self.device)

#         self.agent_spec["drone"] = AgentSpec(
#             "drone", 1,
#             observation_key=("agents", "observation"),
#             action_key=("agents", "action"),
#             reward_key=("agents", "reward"),
#             state_key=("agents", "intrinsics"),
#         )

#         stats_spec = CompositeSpec({
#             "return": UnboundedContinuousTensorSpec(1),
#             "episode_len": UnboundedContinuousTensorSpec(1),
#             "pos_error": UnboundedContinuousTensorSpec(1),
#             "heading_alignment": UnboundedContinuousTensorSpec(1),
#             "uprightness": UnboundedContinuousTensorSpec(1),
#             "action_smoothness": UnboundedContinuousTensorSpec(1),
#         }).expand(self.num_envs).to(self.device)

#         self.observation_spec["stats"] = stats_spec
#         self.stats = stats_spec.zero()

#     def _reset_idx(self, env_ids: torch.Tensor):
#         if self.enable_flow:
#             self.drone.set_flow_velocities(env_ids, self.max_flow_velocity, self.flow_velocity_gaussian_noise)

#         self.drone._reset_idx(env_ids, self.training)

#         pos = self.init_pos_dist.sample((*env_ids.shape, 1))
#         rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
#         rot = euler_to_quaternion(rpy)

#         self.drone.set_world_poses(pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids)
#         self.drone.set_velocities(self.init_vels[env_ids], env_ids)

#         target_rpy = self.target_rpy_dist.sample((*env_ids.shape, 1))
#         target_rot = euler_to_quaternion(target_rpy)
#         self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)
#         self.target_vis.set_world_poses(orientations=target_rot, env_indices=env_ids)

#         self.stats[env_ids] = 0.0

#     def _pre_sim_step(self, tensordict: TensorDictBase):
#         actions = tensordict[("agents", "action")]
#         self.effort = torch.abs(self.drone.apply_action(actions))

#     def _compute_state_and_obs(self):
#         self.drone_state = self.drone.get_state()

#         self.rpos = self.target_pos - self.drone_state[..., :3]
#         self.rheading = self.target_heading - self.drone_state[..., 13:16]

#         obs = [self.rpos, self.drone_state[..., 3:], self.rheading]
#         if self.time_encoding:
#             t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
#             obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
#         obs = torch.cat(obs, dim=-1)

#         return TensorDict(
#             {
#                 "agents": {
#                     "observation": obs,
#                     "intrinsics": self.drone.intrinsics,
#                 },
#                 "stats": self.stats.clone(),
#             },
#             self.batch_size,
#         )

#     def _compute_reward_and_done(self):
#         pos_error = torch.norm(self.rpos, dim=-1)
#         heading_alignment = torch.sum(self.drone.heading * self.target_heading, dim=-1)
#         distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)

#         reward_pose = 0.5 * 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
#         reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)
#         spinnage = torch.square(self.drone.vel[..., -1])
#         reward_spin = 1.0 / (1.0 + torch.square(spinnage))

#         reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
#         reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

#         reward = reward_pose + reward_pose * (reward_up + reward_spin) + reward_effort + reward_action_smoothness

#         misbehave = (self.drone.pos[..., 2] < 0.2) | (distance > 4)
#         hasnan = torch.isnan(self.drone_state).any(-1)

#         terminated = misbehave | hasnan
#         truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

#         self.stats["pos_error"].lerp_(pos_error, (1 - self.alpha))
#         self.stats["heading_alignment"].lerp_(heading_alignment, (1 - self.alpha))
#         self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1 - self.alpha))
#         self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1 - self.alpha))
#         self.stats["return"] += reward
#         self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

#         return TensorDict(
#             {
#                 "agents": {"reward": reward.unsqueeze(-1)},
#                 "done": terminated | truncated,
#                 "terminated": terminated,
#                 "truncated": truncated,
#             },
#             self.batch_size,
#         )


# src/bluerov_manual/envs/hover_tank.py

import os
import omni.isaac.core.utils.prims as prim_utils
import marinegym.utils.kit as kit_utils

from marinegym.envs.single.hover import Hover  # IMPORTANT


class HoverTank(Hover):
    """
    MarineGym Hover + tank USD loaded into /World/envs/env_0/tank
    """

    def _design_scene(self):
        # Build normal Hover scene (target + BlueROV + ground plane)
        global_paths = super()._design_scene()

        # Read tank path from cfg.robot.tank_usd_path (your config layout)
        if not hasattr(self.cfg, "task") or "tank_usd_path" not in self.cfg.task:
            raise RuntimeError("Missing cfg.task.tank_usd_path in your Hydra config.")
        tank_usd = os.path.abspath(os.path.expanduser(str(self.cfg.task.tank_usd_path)))

        if not os.path.exists(tank_usd):
            raise FileNotFoundError(f"Tank USD not found: {tank_usd}")

        tank_prim_path = "/World/envs/env_0/tank"

        if not prim_utils.is_prim_path_valid(tank_prim_path):
            tank_prim = prim_utils.create_prim(
                prim_path=tank_prim_path,
                usd_path=tank_usd,
                translation=(0.0, 0.0, 0.0),
            )

            # Static + collidable
            kit_utils.set_nested_collision_properties(tank_prim.GetPath(), collision_enabled=True)
            kit_utils.set_nested_rigid_body_properties(tank_prim.GetPath(), disable_gravity=True)

        return global_paths
