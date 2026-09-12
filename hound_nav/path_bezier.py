"""Cubic Bezier densify + lookahead carrot (old hound_hl_control interpolate_path)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def interpolate_path(pts: np.ndarray, n_per_seg: int = 40) -> np.ndarray:
    """Join waypoints with cubic Beziers (Catmull–Rom control points), N samples/segment."""
    xy = np.asarray(pts, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xy = np.ascontiguousarray(xy[:, :2], dtype=np.float64)
    n = int(xy.shape[0])
    if n == 1:
        return xy.copy()
    n_per_seg = max(2, int(n_per_seg))
    chunks = []
    for i in range(n - 1):
        p0 = xy[i]
        p3 = xy[i + 1]
        p_im1 = xy[i - 1] if i > 0 else p0
        p_ip2 = xy[i + 2] if i + 2 < n else p3
        p1 = p0 + (p3 - p_im1) / 6.0
        p2 = p3 - (p_ip2 - p0) / 6.0
        last = i == n - 2
        ts = np.linspace(0.0, 1.0, n_per_seg, endpoint=last)
        omt = 1.0 - ts
        chunks.append(
            (omt**3)[:, None] * p0
            + (3.0 * omt**2 * ts)[:, None] * p1
            + (3.0 * omt * ts**2)[:, None] * p2
            + (ts**3)[:, None] * p3
        )
    return np.vstack(chunks)


def _seg_lengths(path: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def carrot_on_path(
    path: np.ndarray,
    pos_xy: np.ndarray,
    lookahead_m: float,
    done_radius_m: float,
) -> Optional[Tuple[np.ndarray, float, bool]]:
    """Closest sample, then ``lookahead_m`` along remaining polyline.

    Returns (goal_xy, yaw_along_tangent, done).
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 1:
        return None
    xy = path[:, :2]
    pos = np.asarray(pos_xy, dtype=np.float64).reshape(-1)[:2]
    i0 = int(np.argmin(np.linalg.norm(xy - pos, axis=1)))
    s = _seg_lengths(xy)
    target_s = float(s[i0]) + max(0.0, float(lookahead_m))
    total = float(s[-1])
    if target_s >= total - 1e-9:
        goal = xy[-1].copy()
        if xy.shape[0] >= 2:
            t = xy[-1] - xy[-2]
            yaw = math.atan2(float(t[1]), float(t[0])) if float(np.linalg.norm(t)) > 1e-9 else 0.0
        else:
            yaw = 0.0
        remaining = max(0.0, total - float(s[i0]))
        done = remaining <= float(done_radius_m) or float(np.linalg.norm(xy[-1] - pos)) <= float(
            done_radius_m
        )
        return goal, float(yaw), bool(done)

    j = int(np.searchsorted(s, target_s, side="left"))
    j = min(max(j, 1), int(xy.shape[0]) - 1)
    s0 = float(s[j - 1])
    s1 = float(s[j])
    span = max(s1 - s0, 1e-12)
    alpha = (target_s - s0) / span
    goal = xy[j - 1] + alpha * (xy[j] - xy[j - 1])
    t = xy[j] - xy[j - 1]
    yaw = math.atan2(float(t[1]), float(t[0])) if float(np.linalg.norm(t)) > 1e-9 else 0.0
    remaining = total - target_s
    done = remaining <= float(done_radius_m)
    return goal.astype(np.float64), float(yaw), bool(done)
