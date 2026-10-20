"""Holonomic MPPI (Williams et al. 2017) tracking a path from the ``holonomic_6D`` IGHA* planner.

State    x, y, yaw (world), vx, vy, wz (body frame)
Control  body-frame velocity command vx, vy, wz, i.e. a geometry_msgs/Twist
Model    each velocity follows its command through a first-order lag (``tau``) with an acceleration limit;
         the pose is the integral of the body velocity.
Cost     tracking error to the planned path, plus a lethal-cell penalty from the costmap at the footprint corners.

Config keys (SSoT ``nav:``): MPPI_config {ROLLOUTS, TIMESTEPS}, Dynamics_config {dt, tau, v_max, a_max},
Sampling_config {noise, temperature}, Cost_config {pos_w, heading_w, vel_w, end_w, lethal_w, car_bb_length,
car_bb_width, hold_yaw (optional)}, Map_config {map_size, map_res}.

``hold_yaw`` (rad, world frame) replaces the path's heading as the heading reference. Use it when the planner does not
rotate: it replans from the robot's current heading, so otherwise nothing pulls a drifting heading back.
"""
from __future__ import annotations

import numpy as np
import torch


class HolonomicMPPI:
    def __init__(self, mppi_cfg, dyn_cfg, sampling_cfg, cost_cfg, map_cfg, device="cuda") -> None:
        f = dict(device=device, dtype=torch.float32)
        self.K, self.T = int(mppi_cfg["ROLLOUTS"]), int(mppi_cfg["TIMESTEPS"])
        self.dt, self.tau = float(dyn_cfg["dt"]), float(dyn_cfg["tau"])
        self.v_max = torch.tensor(dyn_cfg["v_max"], **f)
        self.dv_max = torch.tensor(dyn_cfg["a_max"], **f) * self.dt          # velocity change allowed per step
        self.sigma = torch.tensor(sampling_cfg["noise"], **f)
        self.temperature = float(sampling_cfg["temperature"])
        self.w = {k: float(cost_cfg[k + "_w"]) for k in ("pos", "heading", "vel", "end", "lethal")}
        self.hold_yaw = cost_cfg.get("hold_yaw")
        half_l, half_w = cost_cfg["car_bb_length"] / 2, cost_cfg["car_bb_width"] / 2
        self.corners = torch.tensor([[0, 0], [half_l, half_w], [half_l, -half_w], [-half_l, half_w],
                                     [-half_l, -half_w]], **f)
        self.res = float(map_cfg["map_res"])
        self.n_px = round(float(map_cfg["map_size"]) / self.res)
        self.costmap = torch.full((self.n_px, self.n_px), 255.0, **f)       # 255 free .. 0 lethal
        self.center = torch.zeros(2, **f)
        self.U = torch.zeros((self.T, 3), **f)                              # nominal control sequence

    def reset(self) -> None:
        self.U.zero_()

    def set_costmap(self, costmap, center_xy) -> None:
        self.costmap = torch.as_tensor(costmap, device=self.U.device, dtype=torch.float32)
        self.center = torch.as_tensor(center_xy, device=self.U.device, dtype=torch.float32)

    def rollout(self, s0, U):
        """s0: 6, U: K x T x 3  ->  K x T x 6 states."""
        pose, v = s0[:3].repeat(len(U), 1), s0[3:].repeat(len(U), 1)
        out = []
        for u in U.unbind(1):
            v = v + ((u - v) * min(1.0, self.dt / self.tau)).clamp(-self.dv_max, self.dv_max)
            c, s = pose[:, 2].cos(), pose[:, 2].sin()
            pose = pose + self.dt * torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
            out.append(torch.cat([pose, v], 1))
        return torch.stack(out, 1)

    def lethal(self, pose):
        """Worst (255 - cost) / 255 over the footprint corners, for K x T x 3 poses."""
        c, s = pose[..., 2:3].cos(), pose[..., 2:3].sin()
        x = pose[..., 0:1] + c * self.corners[:, 0] - s * self.corners[:, 1] - self.center[0]
        y = pose[..., 1:2] + s * self.corners[:, 0] + c * self.corners[:, 1] - self.center[1]
        ix = (x / self.res + self.n_px / 2).long().clamp(0, self.n_px - 1)
        iy = (y / self.res + self.n_px / 2).long().clamp(0, self.n_px - 1)
        return ((255.0 - self.costmap[iy, ix]) / 255.0).max(-1).values

    def cost(self, states, ref):
        err = states - ref
        err[..., 2] = torch.atan2(err[..., 2].sin(), err[..., 2].cos())
        step = (self.w["pos"] * err[..., :2].square().sum(-1) + self.w["heading"] * err[..., 2].square()
                + self.w["vel"] * err[..., 3:].square().sum(-1) + self.w["lethal"] * self.lethal(states[..., :3]))
        return step.sum(1) + self.w["end"] * step[:, -1]

    @torch.no_grad()
    def step(self, state, ref) -> np.ndarray:
        """state: 6 numbers, ref: T x 6 desired states one dt apart  ->  command vx, vy, wz."""
        state, ref = (torch.as_tensor(a, device=self.U.device, dtype=torch.float32) for a in (state, ref))
        if self.hold_yaw is not None:
            ref[:, 2] = self.hold_yaw
        self.U = self.U.roll(-1, 0)                                         # warm start from the last solution
        self.U[-1] = self.U[-2]
        U = (self.U + torch.randn((self.K, self.T, 3), device=self.U.device) * self.sigma).clamp(-self.v_max, self.v_max)
        cost = torch.nan_to_num(self.cost(self.rollout(state, U), ref), nan=1e9)
        weights = torch.softmax(-(cost - cost.min()) / self.temperature, 0)
        self.U = self.U + (weights[:, None, None] * (U - self.U)).sum(0)
        return self.U[0].cpu().numpy()
