#!/usr/bin/env python3
"""Export live planner OpenCV viz (costmap, path, start, goal) from a nav bag to MP4.

Uses LocalMap as the clock, holds the latest plan / control_state / goal.
HUD is just IGHA* or Bi-IGHA*. Default names are Pacific local time:

  [Prefix]_[YYYYMMDD]_[HHMMSS].mp4

Bag metadata starting_time is UTC; the stamp is America/Los_Angeles.

  PYTHONPATH=/root/colcon_ws/src/hound_nav python3 \\
    /root/colcon_ws/src/hound_nav/scripts/bag_nav_video.py --all
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

_SCRIPTS = Path(__file__).resolve().parent
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hound_nav.bag_problem_io import (  # noqa: E402
    find_ssot_yaml,
    load_planner_config,
    relative_height,
)
from hound_nav.planner_viz import car_dims_m, visualize_map_with_path  # noqa: E402
from nav_bag_export import (  # noqa: E402
    bag_root,
    export_stem,
    hud_label,
    list_variant_bags,
    unique_stem,
    variant_of,
)

MAP_TOPIC = "/hound_mapping/local_map"
PLAN_TOPIC = "/hound_nav/local_plan"
STATE_TOPIC = "/hound_fcu_control/control_state"
GOAL_TOPIC = "/goal_pose"
DIAG_TOPIC = "/hound_nav/planner_diagnostics"


def _ssot_nav() -> dict:
    raw = yaml.safe_load(find_ssot_yaml().read_text(encoding="utf-8")) or {}
    return dict(raw.get("nav") or {})


def _img_f32(img) -> Optional[np.ndarray]:
    w, h = int(img.width), int(img.height)
    if w < 1 or h < 1:
        return None
    if str(img.encoding) != "32FC1":
        return None
    data = np.frombuffer(bytes(img.data), dtype=np.float32)
    if data.size < w * h:
        return None
    return data[: w * h].reshape(h, w).copy()


def _yaw_from_quat(q) -> float:
    siny = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny, cosy)


def _parse_map(msg) -> Optional[dict]:
    elev = _img_f32(msg.elevation)
    cost = _img_f32(msg.costmap)
    if elev is None or cost is None or elev.shape != cost.shape:
        return None
    res = float(msg.info.resolution)
    h, w = cost.shape
    ox = float(msg.info.origin.position.x)
    oy = float(msg.info.origin.position.y)
    return {
        "cost": cost,
        "elev": relative_height(elev, 0.0),
        "center": np.array([ox + 0.5 * w * res, oy + 0.5 * h * res], dtype=np.float64),
        "res": res,
    }


def _parse_state(msg) -> Optional[np.ndarray]:
    d = list(msg.data)
    if len(d) < 2:
        return None
    yaw = float(d[5]) if len(d) > 5 else 0.0
    return np.array([float(d[0]), float(d[1]), yaw], dtype=np.float64)


def _parse_goal(msg) -> np.ndarray:
    p = msg.pose.position
    return np.array([float(p.x), float(p.y)], dtype=np.float64)


def _parse_plan(msg, dt: float) -> Optional[np.ndarray]:
    poses = list(msg.poses)
    if not poses:
        return None
    rows = []
    for ps in poses:
        p = ps.pose.position
        yaw = _yaw_from_quat(ps.pose.orientation)
        signed_t = float(p.z)
        rows.append([float(p.x), float(p.y), yaw, 0.0, signed_t])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[0] >= 2 and dt > 1e-6:
        step = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
        speed = step / float(dt)
        arr[:-1, 3] = speed
        arr[-1, 3] = speed[-1]
    return arr


def _parse_diag(msg) -> dict[str, str]:
    out: dict[str, str] = {}
    for st in getattr(msg, "status", None) or []:
        for kv in getattr(st, "values", None) or []:
            out[str(kv.key)] = str(kv.value)
        if getattr(st, "message", None):
            out.setdefault("message", str(st.message))
    return out


def _topic_counts(bag: Path) -> dict[str, int]:
    meta = bag / "metadata.yaml"
    if not meta.is_file():
        return {}
    info = yaml.safe_load(meta.read_text())["rosbag2_bagfile_information"]
    out: dict[str, int] = {}
    for item in info.get("topics_with_message_count") or []:
        out[item["topic_metadata"]["name"]] = int(item.get("message_count") or 0)
    return out


def list_bags(root: Path) -> None:
    print(
        f"{'bag':<40} {'map':>6} {'plan':>6} {'diag':>6} {'state':>6}  usable"
    )
    bags = sorted(p for p in root.rglob("metadata.yaml") if p.parent.is_dir())
    for meta in bags:
        bag = meta.parent
        c = _topic_counts(bag)
        n_map = c.get(MAP_TOPIC, 0)
        n_plan = c.get(PLAN_TOPIC, 0)
        ok = n_map > 0
        rel = str(bag.relative_to(root)) if bag.is_relative_to(root) else str(bag)
        print(
            f"{rel:<40} {n_map:>6} {n_plan:>6} {c.get(DIAG_TOPIC, 0):>6} "
            f"{c.get(STATE_TOPIC, 0):>6}  {'yes' if ok else 'no'}"
        )


def _hud(img: np.ndarray, label: str) -> None:
    if not label:
        return
    cv2.putText(img, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        img, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA
    )


def _render(
    m: dict,
    path: Optional[np.ndarray],
    goal: Optional[np.ndarray],
    state: Optional[np.ndarray],
    label: str,
    *,
    map_size: int,
    wp_radius: float,
    car_length: float,
    car_width: float,
    speed_scale: float,
) -> np.ndarray:
    extent = float(m["cost"].shape[0]) * float(m["res"])
    res_inv = float(map_size) / extent if extent > 1e-6 else 1.0
    g = goal if goal is not None else m["center"]
    st = state if state is not None else np.array([m["center"][0], m["center"][1], 0.0])
    img = visualize_map_with_path(
        m["cost"],
        m["elev"],
        path,
        g,
        st,
        wp_radius,
        m["center"],
        map_size,
        res_inv,
        car_length,
        car_width,
        speed_scale=speed_scale,
    )
    _hud(img, label)
    return img


def render_bag(
    bag: Path,
    out: Path,
    *,
    map_size: int,
    fps: float,
    wp_radius: float,
    car_length: float,
    car_width: float,
    label: str = "",
    path_dt: float = 0.05,
    speed_scale: float = 5.0,
) -> None:
    from rosbags.highlevel import AnyReader

    want = {MAP_TOPIC, PLAN_TOPIC, STATE_TOPIC, GOAL_TOPIC, DIAG_TOPIC}
    events: list[tuple[int, str, Any]] = []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in want]
        if not any(c.topic == MAP_TOPIC for c in conns):
            raise SystemExit(f"no {MAP_TOPIC} in {bag}")
        for conn, ts, raw in reader.messages(connections=conns):
            events.append((int(ts), conn.topic, reader.deserialize(raw, conn.msgtype)))
    events.sort(key=lambda e: e[0])

    latest_map: Optional[dict] = None
    latest_plan: Optional[np.ndarray] = None
    latest_state: Optional[np.ndarray] = None
    latest_goal: Optional[np.ndarray] = None
    latest_diag: dict[str, str] = {}
    frames: list[tuple[int, np.ndarray]] = []
    n_map = n_plan = 0
    for ts, topic, msg in events:
        if topic == MAP_TOPIC:
            parsed = _parse_map(msg)
            if parsed is None:
                continue
            latest_map = parsed
            n_map += 1
        elif topic == PLAN_TOPIC:
            latest_plan = _parse_plan(msg, path_dt)
            n_plan += 1
        elif topic == STATE_TOPIC:
            latest_state = _parse_state(msg)
        elif topic == GOAL_TOPIC:
            latest_goal = _parse_goal(msg)
        elif topic == DIAG_TOPIC:
            latest_diag = _parse_diag(msg)
        if latest_map is None:
            continue
        if topic != MAP_TOPIC:
            continue
        frames.append(
            (
                ts,
                _render(
                    latest_map,
                    latest_plan,
                    latest_goal,
                    latest_state,
                    label,
                    map_size=map_size,
                    wp_radius=wp_radius,
                    car_length=car_length,
                    car_width=car_width,
                    speed_scale=speed_scale,
                ),
            )
        )

    if not frames:
        raise SystemExit(f"no renderable LocalMap frames in {bag}")

    t0, t1 = frames[0][0], frames[-1][0]
    span_s = max((t1 - t0) * 1e-9, 1.0 / max(fps, 1.0))
    n_out = max(1, int(round(span_s * fps)))
    h, w = frames[0][1].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise SystemExit(f"could not open VideoWriter for {out}")
    idx = 0
    for i in range(n_out):
        t = t0 + int(i * 1e9 / fps)
        while idx + 1 < len(frames) and frames[idx + 1][0] <= t:
            idx += 1
        writer.write(frames[idx][1])
    writer.release()
    print(
        f"wrote {out}  {n_out} frames @ {fps:.1f} Hz  ({span_s:.1f}s)  "
        f"maps={n_map} plans={n_plan}"
    )


def _default_video_dir(root: Path) -> Path:
    return root / "videos"


def _render_one(
    bag: Path,
    out: Path,
    args: argparse.Namespace,
    label: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    render_bag(
        bag,
        out,
        map_size=int(args.map_size),
        fps=float(args.fps),
        wp_radius=float(args.wp_radius),
        car_length=float(args.car_length),
        car_width=float(args.car_width),
        label=label,
        path_dt=float(args.path_dt),
        speed_scale=float(args.speed_scale),
    )


def main() -> None:
    nav = _ssot_nav()
    planner_cfg = load_planner_config()
    car_l, car_w = car_dims_m(planner_cfg)
    wp = float(nav.get("wp_radius", 1.0))
    node = (
        (planner_cfg or {}).get("experiment_info_default") or {}
    ).get("node_info") or {}
    path_dt = float(node.get("dt", 0.05))
    speed_scale = float((nav.get("Sampling_config") or {}).get("max_spd", 5.0))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", nargs="?", help="rosbag2 directory")
    ap.add_argument(
        "--bags",
        type=Path,
        default=None,
        help="root for --list / --all (default bags_nav)",
    )
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--all",
        action="store_true",
        help="render every unidirectional/bidirectional bag",
    )
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="directory for --all (default <bags>/videos)",
    )
    ap.add_argument("--label", default="", help="HUD text (default from bag path)")
    ap.add_argument("--map-size", type=int, default=480)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--wp-radius", type=float, default=wp)
    ap.add_argument("--car-length", type=float, default=car_l)
    ap.add_argument("--car-width", type=float, default=car_w)
    ap.add_argument(
        "--path-dt",
        type=float,
        default=path_dt,
        help="planner sample interval (s); speed = spacing / dt",
    )
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=speed_scale,
        help="m/s that maps to full-bright green/red (SSoT wheelspeed_max)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing mp4 files",
    )
    args = ap.parse_args()
    root = bag_root(args.bags)

    if args.list:
        list_bags(root)
        if args.bag is None and not args.all:
            return

    if args.all:
        jobs = list_variant_bags(root)
        if not jobs:
            raise SystemExit(f"no hound_* bags under {root}/{{unidirectional,bidirectional}}")
        dest = args.video_dir or _default_video_dir(root)
        dest.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        for variant, bag in jobs:
            stem = unique_stem(export_stem(variant, bag), used, bag)
            out = dest / f"{stem}.mp4"
            if out.is_file() and not args.overwrite:
                print(f"skip existing {out}")
                continue
            print(f"render {variant} {bag.name} -> {out.name}")
            _render_one(bag, out, args, hud_label(variant))
        return

    if args.bag is None:
        list_bags(root)
        return

    bag = Path(args.bag).expanduser().resolve()
    if not bag.is_dir():
        raise SystemExit(f"bag not found: {args.bag}")
    label = args.label
    if not label:
        try:
            label = hud_label(variant_of(bag))
        except ValueError:
            label = ""
    if args.output is not None:
        out = args.output
    else:
        try:
            variant = variant_of(bag)
            out = _default_video_dir(root) / f"{export_stem(variant, bag)}.mp4"
        except (ValueError, FileNotFoundError):
            out = bag.with_name(bag.name + "_cv.mp4")
    _render_one(bag, out, args, label)


if __name__ == "__main__":
    main()
