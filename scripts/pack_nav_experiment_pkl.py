#!/usr/bin/env python3
"""Pack nav rosbag trials into a pickle for in_the_loop_plot-style analysis.

Each trial has the sim keys (timestamps, state_data, path_data, expansions_data,
goal_data, progress, time_taken, planner_availability, frame_rate) plus bag
series that already exist: wheelspeed, vx, planner ok/dt, valid-plan times,
local_map rate.

  PYTHONPATH=/root/colcon_ws/src/hound_nav python3 \\
    /root/colcon_ws/src/hound_nav/scripts/pack_nav_experiment_pkl.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_nav_mission_time import (  # noqa: E402
    ARMED_TOPIC,
    AUTO_PWM,
    RC_MODE_CH,
    RC_TOPIC,
    armed_auto_mask,
    cluster_goals,
    finish_in_odom,
    gated_duration,
)
from nav_bag_export import (  # noqa: E402
    bag_root,
    bag_start_pacific,
    bag_start_utc,
    export_stem,
    hud_label,
    list_variant_bags,
    unique_stem,
)

STATE_TOPIC = "/hound_fcu_control/control_state"
GOAL_TOPIC = "/goal_pose"
WP_TOPIC = "/hound_nav/mission/waypoints"
PLAN_TOPIC = "/hound_nav/local_plan"
DIAG_TOPIC = "/hound_nav/planner_diagnostics"
MAP_TOPIC = "/hound_mapping/local_map"


def _yaw_from_quat(q) -> float:
    siny = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return float(np.arctan2(siny, cosy))


def _parse_plan(msg) -> Optional[np.ndarray]:
    poses = list(msg.poses)
    if not poses:
        return None
    rows = []
    for ps in poses:
        p = ps.pose.position
        yaw = _yaw_from_quat(ps.pose.orientation)
        signed_t = float(p.z)
        rows.append([float(p.x), float(p.y), yaw, 1.0, signed_t])
    return np.asarray(rows, dtype=np.float64)


def _parse_diag(msg) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for st in getattr(msg, "status", None) or []:
        for kv in getattr(st, "values", None) or []:
            key = str(kv.key)
            raw = str(kv.value)
            if key in {"ok", "expansions", "expansion_limit", "path_n"}:
                try:
                    out[key] = int(float(raw))
                except ValueError:
                    out[key] = raw
            elif key in {"dt_s", "goal_dist_m", "query_t"}:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
            else:
                out[key] = raw
        if getattr(st, "message", None):
            out.setdefault("message", str(st.message))
    return out


def _hold_last(event_t: np.ndarray, values: list[Any], query_t: np.ndarray) -> list[Any]:
    """values[i] is current at event_t[i]; sample at query_t with hold-last."""
    out: list[Any] = [None] * int(query_t.size)
    if not event_t.size:
        return out
    j = -1
    n_e = int(event_t.size)
    for i, t in enumerate(query_t):
        while j + 1 < n_e and event_t[j + 1] <= t:
            j += 1
        if j >= 0:
            out[i] = values[j]
    return out


def _paths_differ(a: Any, b: Any, atol: float = 1e-1) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    try:
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape != bb.shape:
            return True
        return not np.allclose(aa, bb, atol=atol)
    except (TypeError, ValueError):
        return True


def planner_availability(
    timestamps: np.ndarray, path_data: list[Any], min_dt: float = 0.1
) -> tuple[float, float, np.ndarray]:
    """Hz of path changes, same filter as process_trial_data (dt >= 0.1).

    Paths in the bag are already in odom, so they are compared as-is.
    """
    change_t: list[float] = []
    for i in range(1, len(path_data)):
        if _paths_differ(path_data[i - 1], path_data[i]):
            change_t.append(float(timestamps[i]))
    ts = np.asarray(change_t, dtype=np.float64)
    if ts.size < 2:
        return 0.0, 0.0, ts
    dt = np.diff(ts)
    filt = dt[dt >= min_dt]
    outlier = 1.0 - (len(filt) / len(dt)) if len(dt) else 0.0
    mean_dt = float(np.mean(filt)) if filt.size else float(np.mean(dt))
    hz = (1.0 / mean_dt) if mean_dt > 0.0 else 0.0
    return hz, float(outlier), ts


def frame_rate(timestamps: np.ndarray, min_dt: float = 0.01) -> float:
    if timestamps.size < 2:
        return 0.0
    dt = np.diff(timestamps)
    filt = dt[dt >= min_dt]
    mean_dt = float(np.mean(filt)) if filt.size else float(np.mean(dt))
    return (1.0 / mean_dt) if mean_dt > 0.0 else 0.0


def load_trial_bag(bag: Path) -> dict[str, Any]:
    from rosbags.highlevel import AnyReader

    t_s: list[float] = []
    state: list[list[float]] = []
    g_xy: list[tuple[float, float]] = []
    g_t: list[float] = []
    wp_xy: Optional[np.ndarray] = None
    wp_frame = ""
    plan_t: list[float] = []
    plans: list[Optional[np.ndarray]] = []
    diag_t: list[float] = []
    diags: list[dict[str, Any]] = []
    map_t: list[float] = []
    armed_t: list[float] = []
    armed_v: list[bool] = []
    rc_t: list[float] = []
    rc_auto: list[bool] = []

    want = {
        STATE_TOPIC,
        GOAL_TOPIC,
        WP_TOPIC,
        PLAN_TOPIC,
        DIAG_TOPIC,
        MAP_TOPIC,
        ARMED_TOPIC,
        RC_TOPIC,
    }
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in want]
        for conn, ts, raw in reader.messages(connections=conns):
            t = ts * 1e-9
            if conn.topic == MAP_TOPIC:
                map_t.append(t)
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == STATE_TOPIC:
                d = [float(x) for x in msg.data]
                if len(d) < 9:
                    continue
                if len(d) < 17:
                    d = d + [float("nan")] * (17 - len(d))
                t_s.append(t)
                state.append(d[:17])
            elif conn.topic == GOAL_TOPIC:
                g_t.append(t)
                g_xy.append((float(msg.pose.position.x), float(msg.pose.position.y)))
            elif conn.topic == WP_TOPIC:
                poses = list(msg.poses)
                if not poses:
                    continue
                wp_frame = str(msg.header.frame_id)
                wp_xy = np.array(
                    [[float(p.pose.position.x), float(p.pose.position.y)] for p in poses],
                    dtype=np.float64,
                )
            elif conn.topic == PLAN_TOPIC:
                plan_t.append(t)
                plans.append(_parse_plan(msg))
            elif conn.topic == DIAG_TOPIC:
                diag_t.append(t)
                diags.append(_parse_diag(msg))
            elif conn.topic == ARMED_TOPIC:
                armed_t.append(t)
                raw_a = getattr(msg, "data", False)
                if isinstance(raw_a, (list, tuple, np.ndarray)):
                    raw_a = raw_a[0] if len(raw_a) else False
                armed_v.append(bool(raw_a))
            elif conn.topic == RC_TOPIC:
                raw_ch = getattr(msg, "data", None)
                if raw_ch is None:
                    continue
                ch = list(raw_ch)
                if len(ch) <= RC_MODE_CH:
                    continue
                rc_t.append(t)
                rc_auto.append(int(ch[RC_MODE_CH]) > AUTO_PWM)

    return {
        "t": np.asarray(t_s, dtype=np.float64),
        "state": np.asarray(state, dtype=np.float64) if state else np.zeros((0, 17)),
        "g_xy": np.asarray(g_xy, dtype=np.float64) if g_xy else np.zeros((0, 2)),
        "g_t": np.asarray(g_t, dtype=np.float64),
        "wp_xy": wp_xy,
        "wp_frame": wp_frame,
        "plan_t": np.asarray(plan_t, dtype=np.float64),
        "plans": plans,
        "diag_t": np.asarray(diag_t, dtype=np.float64),
        "diags": diags,
        "map_t": np.asarray(map_t, dtype=np.float64),
        "armed_t": np.asarray(armed_t, dtype=np.float64),
        "armed_v": np.asarray(armed_v, dtype=bool),
        "rc_t": np.asarray(rc_t, dtype=np.float64),
        "rc_auto": np.asarray(rc_auto, dtype=bool),
    }


def pack_one(
    bag: Path,
    variant: str,
    *,
    v_start: float,
    goal_radius: float,
    cluster_jump: float,
    timeout: float,
    video_stem: str,
) -> dict[str, Any]:
    raw = load_trial_bag(bag)
    t = raw["t"]
    state = raw["state"]
    if t.size < 2:
        raise ValueError("no control_state")
    xy = state[:, :2]
    vx = state[:, 6]
    roll = state[:, 3]
    wheelspeed = state[:, 16]
    start_xy = xy[0]
    moving = np.where(np.abs(vx) > v_start)[0]
    if moving.size == 0:
        raise ValueError(f"vx never exceeded {v_start} m/s")
    i0 = int(moving[0])
    t_start = float(t[i0])
    clusters = cluster_goals(raw["g_xy"], raw["g_t"], cluster_jump)
    finish, t_gate, finish_src, last_pub = finish_in_odom(
        clusters, raw["wp_xy"], start_xy
    )
    dist = np.linalg.norm(xy - finish.reshape(1, 2), axis=1)
    t_end: Optional[float] = None
    i_end: Optional[int] = None
    if last_pub and t_gate is not None:
        gate = np.where(t >= max(t_start, t_gate))[0]
        if gate.size:
            hit = np.where(dist[gate] <= goal_radius)[0]
            if hit.size:
                i_end = int(gate[int(hit[0])])
                t_end = float(t[i_end])
    success = t_end is not None
    i_last = i_end if i_end is not None else (int(t.size) - 1)
    wall_s = (t_end - t_start) if success else None
    on = armed_auto_mask(raw, t)
    if success:
        mission_s = gated_duration(t, on, i0, i_last) if on is not None else wall_s
    else:
        mission_s = None
    sl = slice(i0, i_last + 1)
    ts = t[sl] - t_start
    st = state[sl]
    path_held = _hold_last(raw["plan_t"], raw["plans"], t[sl])
    diag_held = _hold_last(raw["diag_t"], raw["diags"], t[sl])
    goal_held = _hold_last(
        raw["g_t"],
        [np.asarray(g, dtype=np.float64) for g in raw["g_xy"]],
        t[sl],
    )
    goal_data = []
    for g in goal_held:
        if g is None:
            goal_data.append(np.array([float("nan"), float("nan")], dtype=np.float64))
        else:
            goal_data.append(np.asarray(g, dtype=np.float64))
    expansions = np.array(
        [float((d or {}).get("expansions", 0) or 0) for d in diag_held],
        dtype=np.float64,
    )
    planner_ok = np.array(
        [int((d or {}).get("ok", 0) or 0) for d in diag_held],
        dtype=np.int8,
    )
    planner_dt = np.array(
        [float((d or {}).get("dt_s", float("nan"))) for d in diag_held],
        dtype=np.float64,
    )
    path_n = np.array(
        [
            0 if p is None else int(np.asarray(p).shape[0])
            for p in path_held
        ],
        dtype=np.int32,
    )
    start_goal = float(np.linalg.norm(xy[i0] - finish))
    if start_goal < 1e-6:
        start_goal = 1.0
    progress = 1.0 - dist[sl] / start_goal
    if last_pub and t_gate is not None:
        reached = (t[sl] >= max(t_start, t_gate)) & (dist[sl] <= goal_radius)
        progress[reached] = 1.0
    rolled = bool(np.any(np.abs(roll[sl]) > 1.2))
    if rolled:
        time_taken = float(timeout)
        success = False
        mission_s = None
    elif success and mission_s is not None:
        time_taken = float(mission_s)
        if time_taken > timeout:
            time_taken = float(timeout)
            success = False
    else:
        time_taken = float(timeout)

    avail, outlier, change_t = planner_availability(ts, path_held)
    valid_plan_t = []
    for pt, plan in zip(raw["plan_t"].tolist(), raw["plans"]):
        if plan is None or len(plan) == 0:
            continue
        if t_start <= pt <= t[i_last]:
            valid_plan_t.append(pt - t_start)
    valid_plan_t = np.asarray(valid_plan_t, dtype=np.float64)
    if valid_plan_t.size > 1:
        vdt = np.diff(valid_plan_t)
        vfilt = vdt[vdt >= 0.1]
        mean_v = float(np.mean(vfilt)) if vfilt.size else float(np.mean(vdt))
        valid_plan_hz = (1.0 / mean_v) if mean_v > 0.0 else 0.0
    else:
        valid_plan_hz = 0.0

    map_win = raw["map_t"]
    map_win = map_win[(map_win >= t_start) & (map_win <= t[i_last])]
    path_m = 0.0
    if i_last > i0:
        path_m = float(np.sum(np.linalg.norm(np.diff(xy[i0 : i_last + 1], axis=0), axis=1)))

    utc = bag_start_utc(bag)
    pacific = bag_start_pacific(bag)
    return {
        "bag": bag.name,
        "variant": variant,
        "label": hud_label(variant),
        "video_stem": video_stem,
        "start_utc": utc.isoformat(),
        "start_pacific": pacific.isoformat(),
        "timestamps": ts,
        "t_abs": t[sl],
        "t_start_abs": t_start,
        "state_data": st,
        "path_data": path_held,
        "expansions_data": expansions,
        "goal_data": goal_data,
        "progress": progress,
        "time_taken": time_taken,
        "success": bool(success and time_taken <= timeout),
        "mission_s": mission_s,
        "mission_wall_s": wall_s,
        "armed_auto_gated": on is not None,
        "armed_auto": on[sl] if on is not None else None,
        "planner_availability": avail,
        "planner_outlier_fraction": outlier,
        "path_change_times": change_t,
        "valid_plan_times": valid_plan_t,
        "valid_plan_hz": valid_plan_hz,
        "frame_rate": frame_rate(ts),
        "local_map_times": map_win - t_start if map_win.size else map_win,
        "local_map_hz": frame_rate(map_win, min_dt=0.05),
        "vx": st[:, 6],
        "wheelspeed": wheelspeed[sl],
        "planner_ok": planner_ok,
        "planner_dt_s": planner_dt,
        "path_n": path_n,
        "finish_xy": [float(finish[0]), float(finish[1])],
        "finish_src": finish_src,
        "last_wp_published": last_pub,
        "n_goal_clusters": len(clusters),
        "n_map_wps": 0 if raw["wp_xy"] is None else int(raw["wp_xy"].shape[0]),
        "path_m": path_m,
        "vx_max": float(np.max(np.abs(st[:, 6]))) if st.size else float("nan"),
        "wheelspeed_max": (
            float(np.nanmax(np.abs(wheelspeed[sl]))) if sl.stop > sl.start else float("nan")
        ),
        "rollover": rolled,
        "timeout_s": timeout,
        "wp_frame": raw["wp_frame"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bags", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--v-start", type=float, default=0.25)
    ap.add_argument("--goal-radius", type=float, default=2.0)
    ap.add_argument("--cluster-jump", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    root = bag_root(args.bags)
    jobs = list_variant_bags(root)
    if not jobs:
        raise SystemExit(f"no hound_* bags under {root}/{{unidirectional,bidirectional}}")

    used: set[str] = set()
    trials: list[dict[str, Any]] = []
    for variant, bag in jobs:
        stem = unique_stem(export_stem(variant, bag), used, bag)
        try:
            trial = pack_one(
                bag,
                variant,
                v_start=float(args.v_start),
                goal_radius=float(args.goal_radius),
                cluster_jump=float(args.cluster_jump),
                timeout=float(args.timeout),
                video_stem=stem,
            )
        except Exception as exc:
            print(f"{variant} {bag.name} ERR {exc}", file=sys.stderr)
            continue
        trials.append(trial)
        t = trial["time_taken"]
        print(
            f"{variant:<16} {bag.name:<10} {trial['label']:<8} "
            f"{'ok' if trial['success'] else 'no':>3}  T={t:6.2f}s  "
            f"plan={trial['planner_availability']:.2f}Hz  "
            f"valid={trial['valid_plan_hz']:.2f}Hz  "
            f"ws={trial['wheelspeed_max']:.2f}  {stem}"
        )

    payload = {
        "timeout_s": float(args.timeout),
        "v_start": float(args.v_start),
        "goal_radius": float(args.goal_radius),
        "cluster_jump": float(args.cluster_jump),
        "trials": trials,
        "unidirectional": [t for t in trials if t["variant"] == "unidirectional"],
        "bidirectional": [t for t in trials if t["variant"] == "bidirectional"],
    }
    out = args.output or (root / "experiment_data.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_u = len(payload["unidirectional"])
    n_b = len(payload["bidirectional"])
    print(f"wrote {out}  trials={len(trials)}  IGHA*={n_u}  Bi-IGHA*={n_b}")


if __name__ == "__main__":
    main()
