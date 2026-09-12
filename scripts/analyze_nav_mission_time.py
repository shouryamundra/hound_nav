#!/usr/bin/env python3
"""Mission time from nav rosbags (uni vs bi).

Bag t0 is not the run start — the car is always at rest first. Start is the
first control_state sample with |vx| > --v-start (default 0.25 m/s).

When /hound_fcu_control/state/armed and /hound_fcu_control/rc/in exist,
mission time is only the armed+auto portion of [t_start, t_end] (RC CH5
data[4] > 1800). Missing those topics keeps wall-clock t_end - t_start.

The finish is the last mission waypoint in odom. /goal_pose is already in
odom, but some runs never publish the last WP (they fail earlier), so we
cannot use the last /goal_pose message. WP0 and the loop finish are also
~1 m apart, so a 2 m radius around the finish matches the first carrot.

  1. Cluster successive /goal_pose xy (jump > --cluster-jump) → odom WPs
     in visit order.
  2. Read the full list from /hound_nav/mission/waypoints (map frame).
  3. If every map WP was published, the last cluster is the finish.
     Arrival only counts after that last WP is published.
  4. Else the last WP was never the active goal → not a finish. Estimate
     it via a 2D rigid map→odom for min-distance diagnostics only.

  python3 analyze_nav_mission_time.py
  python3 analyze_nav_mission_time.py --bags /root/colcon_ws/bags_nav
  python3 analyze_nav_mission_time.py --timeout 60
  python3 analyze_nav_mission_time.py --best 20
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

STATE_TOPIC = "/hound_fcu_control/control_state"
GOAL_TOPIC = "/goal_pose"
WP_TOPIC = "/hound_nav/mission/waypoints"
ARMED_TOPIC = "/hound_fcu_control/state/armed"
RC_TOPIC = "/hound_fcu_control/rc/in"
AUTO_PWM = 1800
RC_MODE_CH = 4

DEFAULT_BAG_ROOTS = (
    Path("/root/colcon_ws/bags_nav"),
    Path("/home/hound/colcon_ws/bags_nav"),
)


def _bag_root() -> Path:
    for p in DEFAULT_BAG_ROOTS:
        if p.is_dir():
            return p
    return DEFAULT_BAG_ROOTS[0]


def list_variant_bags(root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for variant in ("unidirectional", "bidirectional"):
        d = root / variant
        if not d.is_dir():
            continue
        for bag in sorted(d.glob("hound_*")):
            if (bag / "metadata.yaml").is_file():
                out.append((variant, bag))
    return out


def rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch 2D: dst ≈ R @ src + t. src/dst are Nx2."""
    if src.shape[0] == 0:
        return np.eye(2), np.zeros(2)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    if src.shape[0] == 1:
        return np.eye(2), mu_d - mu_s
    h = (src - mu_s).T @ (dst - mu_d)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt = vt.copy()
        vt[-1] *= -1.0
        r = vt.T @ u.T
    t = mu_d - r @ mu_s
    return r, t


def apply_rigid(xy: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (r @ xy.reshape(2, 1)).ravel() + t


def cluster_goals(
    xy: np.ndarray, t: np.ndarray, jump_m: float
) -> list[tuple[np.ndarray, float]]:
    """Visit-ordered (centroid, first_timestamp) clusters."""
    if xy.shape[0] == 0:
        return []
    groups: list[list[int]] = [[0]]
    for i in range(1, xy.shape[0]):
        last = groups[-1][-1]
        if float(np.linalg.norm(xy[i] - xy[last])) > jump_m:
            groups.append([i])
        else:
            groups[-1].append(i)
    out: list[tuple[np.ndarray, float]] = []
    for g in groups:
        idx = np.asarray(g)
        out.append((np.mean(xy[idx], axis=0), float(t[g[0]])))
    return out


def load_bag(bag: Path) -> dict[str, Any]:
    from rosbags.highlevel import AnyReader

    t_s: list[float] = []
    xy: list[tuple[float, float]] = []
    vx: list[float] = []
    wheelspeed: list[float] = []
    g_xy: list[tuple[float, float]] = []
    g_t: list[float] = []
    wp_xy: Optional[np.ndarray] = None
    wp_frame = ""
    armed_t: list[float] = []
    armed_v: list[bool] = []
    rc_t: list[float] = []
    rc_auto: list[bool] = []
    with AnyReader([bag]) as reader:
        want = {STATE_TOPIC, GOAL_TOPIC, WP_TOPIC, ARMED_TOPIC, RC_TOPIC}
        conns = [c for c in reader.connections if c.topic in want]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == STATE_TOPIC:
                d = list(msg.data)
                if len(d) < 9:
                    continue
                t_s.append(ts * 1e-9)
                xy.append((float(d[0]), float(d[1])))
                vx.append(float(d[6]))
                wheelspeed.append(float(d[16]) if len(d) > 16 else float("nan"))
            elif conn.topic == GOAL_TOPIC:
                g_t.append(ts * 1e-9)
                g_xy.append(
                    (float(msg.pose.position.x), float(msg.pose.position.y))
                )
            elif conn.topic == WP_TOPIC:
                poses = list(msg.poses)
                if not poses:
                    continue
                wp_frame = str(msg.header.frame_id)
                wp_xy = np.array(
                    [
                        [float(p.pose.position.x), float(p.pose.position.y)]
                        for p in poses
                    ],
                    dtype=np.float64,
                )
            elif conn.topic == ARMED_TOPIC:
                armed_t.append(ts * 1e-9)
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
                rc_t.append(ts * 1e-9)
                rc_auto.append(int(ch[RC_MODE_CH]) > AUTO_PWM)
    return {
        "t": np.asarray(t_s, dtype=np.float64),
        "xy": np.asarray(xy, dtype=np.float64),
        "vx": np.asarray(vx, dtype=np.float64),
        "wheelspeed": np.asarray(wheelspeed, dtype=np.float64),
        "g_xy": np.asarray(g_xy, dtype=np.float64)
        if g_xy
        else np.zeros((0, 2)),
        "g_t": np.asarray(g_t, dtype=np.float64),
        "wp_xy": wp_xy,
        "wp_frame": wp_frame,
        "armed_t": np.asarray(armed_t, dtype=np.float64),
        "armed_v": np.asarray(armed_v, dtype=bool),
        "rc_t": np.asarray(rc_t, dtype=np.float64),
        "rc_auto": np.asarray(rc_auto, dtype=bool),
    }


def _hold_last_bool(
    event_t: np.ndarray, event_v: np.ndarray, query_t: np.ndarray
) -> Optional[np.ndarray]:
    if event_t.size == 0 or query_t.size == 0:
        return None
    out = np.zeros(query_t.size, dtype=bool)
    j = -1
    n_e = int(event_t.size)
    for i, tq in enumerate(query_t):
        while j + 1 < n_e and event_t[j + 1] <= tq:
            j += 1
        if j >= 0:
            out[i] = bool(event_v[j])
    if j < 0:
        return None
    return out


def armed_auto_mask(data: dict[str, Any], t: np.ndarray) -> Optional[np.ndarray]:
    armed = _hold_last_bool(data["armed_t"], data["armed_v"], t)
    auto = _hold_last_bool(data["rc_t"], data["rc_auto"], t)
    if armed is None or auto is None:
        return None
    return armed & auto


def gated_duration(t: np.ndarray, on: np.ndarray, i0: int, i_last: int) -> float:
    """Sum dt in [i0, i_last] while hold-last `on` is true at the interval start."""
    if i_last <= i0:
        return 0.0
    dt = np.diff(t[i0 : i_last + 1])
    return float(np.sum(dt[on[i0:i_last]]))


def finish_in_odom(
    clusters: list[tuple[np.ndarray, float]],
    map_wps: Optional[np.ndarray],
    start_xy: np.ndarray,
) -> tuple[np.ndarray, Optional[float], str, bool]:
    """Return (finish_xy, t_last_wp_published, src, last_wp_published).

    WP0 and the loop finish are ~1 m apart, so "within 2 m of finish" also
    matches driving to the first carrot. Only treat arrival as a finish after
    the last mission WP has been published as /goal_pose.
    """
    if not clusters:
        raise ValueError("no /goal_pose clusters")
    cents = [c[0] for c in clusters]
    n_map = 0 if map_wps is None else int(map_wps.shape[0])
    if n_map >= 2 and len(clusters) >= n_map:
        return cents[n_map - 1], clusters[n_map - 1][1], "last_goal_cluster", True
    if n_map >= 2:
        n_pair = min(len(clusters), n_map)
        r, t = rigid_2d(map_wps[:n_pair], np.stack(cents[:n_pair], axis=0))
        return apply_rigid(map_wps[-1], r, t), None, "map_wp_rigid", False
    rest = clusters[1:] if len(clusters) > 1 else clusters
    d = [float(np.linalg.norm(c[0] - start_xy)) for c in rest]
    i = int(np.argmin(d))
    chosen = rest[i]
    published = len(clusters) > 1
    return chosen[0], chosen[1] if published else None, "closest_to_start_skip_first", published


def analyze_one(
    bag: Path,
    *,
    v_start: float,
    goal_radius: float,
    cluster_jump: float,
) -> dict[str, Any]:
    data = load_bag(bag)
    t = data["t"]
    xy = data["xy"]
    vx = data["vx"]
    wheelspeed = data["wheelspeed"]
    if t.size < 2:
        raise ValueError("no control_state")
    start_xy = xy[0]
    moving = np.where(np.abs(vx) > v_start)[0]
    if moving.size == 0:
        raise ValueError(f"vx never exceeded {v_start} m/s")
    i0 = int(moving[0])
    t_start = float(t[i0])
    clusters = cluster_goals(data["g_xy"], data["g_t"], cluster_jump)
    finish, t_gate, finish_src, last_pub = finish_in_odom(
        clusters, data["wp_xy"], start_xy
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
    i_last = i_end if i_end is not None else (len(t) - 1)
    wall_s = (t_end - t_start) if success else None
    on = armed_auto_mask(data, t)
    used_gate = on is not None
    if success:
        if used_gate:
            mission_s = gated_duration(t, on, i0, i_last)
        else:
            mission_s = wall_s
    else:
        mission_s = None
    path_m = 0.0
    if i_last > i0:
        path_m = float(
            np.sum(np.linalg.norm(np.diff(xy[i0 : i_last + 1], axis=0), axis=1))
        )
    dist_after_gate = dist[t >= (t_gate if t_gate is not None else t_start)]
    min_after = (
        float(dist_after_gate.min()) if dist_after_gate.size else float("nan")
    )
    window = wheelspeed[i0 : i_last + 1]
    vx_win = vx[i0 : i_last + 1]
    wheelspeed_max = (
        float(np.nanmax(np.abs(window))) if window.size else float("nan")
    )
    vx_max = float(np.max(np.abs(vx_win))) if vx_win.size else float("nan")
    return {
        "bag": bag.name,
        "n_state": int(t.size),
        "n_goal_clusters": len(clusters),
        "n_map_wps": 0 if data["wp_xy"] is None else int(data["wp_xy"].shape[0]),
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "finish_xy": [float(finish[0]), float(finish[1])],
        "finish_src": finish_src,
        "last_wp_published": last_pub,
        "dist_start_to_finish_m": float(np.linalg.norm(start_xy - finish)),
        "t_start_from_bag_s": t_start - float(t[0]),
        "mission_s": mission_s,
        "mission_wall_s": wall_s,
        "armed_auto_gated": used_gate,
        "success": success,
        "min_dist_to_finish_m": min_after,
        "path_m": path_m,
        "bag_span_s": float(t[-1] - t[0]),
        "vx_max": vx_max,
        "wheelspeed_max": wheelspeed_max,
    }


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


_Z95 = 1.959963984540054


def bernoulli_ci(binary: np.ndarray) -> tuple[float, float, float]:
    """Wilson score interval (95%). Returns (p_hat, lo, hi)."""
    n = int(binary.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    p = float(np.mean(binary))
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    err = _Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = max(0.0, center - err)
    hi = min(1.0, center + err)
    return p, lo, hi


VARIANT_STYLE = {
    "unidirectional": {
        "label": "IGHA*",
        "color": "#c2a5cf",
        "alpha": 1.0,
        "linewidth": 2,
        "marker_size": 20,
        "linestyle": "-",
        "marker": "o",
    },
    "bidirectional": {
        "label": "Bi-IGHA*",
        "color": "#a6dba0",
        "alpha": 1.0,
        "linewidth": 2,
        "marker_size": 20,
        "linestyle": "-",
        "marker": "o",
    },
}


def rank_time(row: dict[str, Any], timeout: float) -> float:
    ok = (
        bool(row.get("success"))
        and row.get("mission_s") is not None
        and float(row["mission_s"]) <= timeout
    )
    return float(row["mission_s"]) if ok else float(timeout)


def select_best(
    results: list[dict[str, Any]], n: int, timeout: float
) -> list[dict[str, Any]]:
    """Keep the n fastest bags per variant (timeouts rank last)."""
    if n <= 0:
        return results
    kept: list[dict[str, Any]] = []
    for variant in ("unidirectional", "bidirectional"):
        sub = [r for r in results if r["variant"] == variant]
        sub.sort(key=lambda r: (rank_time(r, timeout), r["bag"]))
        kept.extend(sub[:n])
    return kept


def timed_outcomes(
    rows: list[dict[str, Any]], timeout: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fails and late finishes count as timeout (not success)."""
    times: list[float] = []
    success: list[float] = []
    for r in rows:
        ok = (
            bool(r["success"])
            and r["mission_s"] is not None
            and float(r["mission_s"]) <= timeout
        )
        times.append(float(r["mission_s"]) if ok else float(timeout))
        success.append(1.0 if ok else 0.0)
    return np.asarray(times, dtype=np.float64), np.asarray(success, dtype=np.float64)


def success_vs_time_curve(
    rows: list[dict[str, Any]], timeout: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative success rate (%) vs time, sampled at trial finish times.

    Matches in_the_loop_plot_new: at each trial time t, rate is the fraction of
    *all* trials that succeeded with time_taken <= t. Timeouts sit at `timeout`
    with success=0 so they never lift the curve.
    """
    times, success = timed_outcomes(rows, timeout)
    if times.size == 0:
        z = np.array([0.0, timeout])
        return z, np.zeros(2), np.zeros(2), np.zeros(2)
    order = np.argsort(times)
    times = times[order]
    success = success[order]
    p0, lo0, hi0 = bernoulli_ci(np.zeros_like(times))
    ts = [0.0]
    cdf = [p0 * 100.0]
    lo = [lo0 * 100.0]
    hi = [hi0 * 100.0]
    for t in times:
        completion = np.zeros_like(times)
        mask = times <= t
        completion[mask] = success[mask]
        p, a, b = bernoulli_ci(completion)
        ts.append(float(t))
        cdf.append(p * 100.0)
        lo.append(a * 100.0)
        hi.append(b * 100.0)
    if ts[-1] < timeout:
        p, a, b = bernoulli_ci(success)
        ts.append(float(timeout))
        cdf.append(p * 100.0)
        lo.append(a * 100.0)
        hi.append(b * 100.0)
    return np.asarray(ts), np.asarray(cdf), np.asarray(lo), np.asarray(hi)


def plot_success_vs_time(
    results: list[dict[str, Any]],
    timeout: float,
    output_prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 3))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 12

    for variant, style in VARIANT_STYLE.items():
        sub = [r for r in results if r["variant"] == variant]
        if not sub:
            continue
        t, cdf, lo, hi = success_vs_time_curve(sub, timeout)
        plt.plot(
            t,
            cdf,
            label=style["label"],
            color=style["color"],
            alpha=style["alpha"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            drawstyle="steps-post",
        )
        plt.scatter(
            t[t > 0],
            cdf[t > 0],
            s=style["marker_size"],
            color=style["color"],
            alpha=style["alpha"],
            marker=style["marker"],
            zorder=3,
        )
        plt.fill_between(
            t,
            lo,
            hi,
            step="post",
            alpha=0.2 * style["alpha"],
            color=style["color"],
        )

    plt.xlabel("Mission Time (seconds)", fontsize=16)
    plt.ylabel("Success Rate (%)", fontsize=16)
    ax = plt.gca()
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
    plt.xlim(0, timeout)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, timeout + 1e-9, 10))
    plt.yticks(np.arange(0.0, 125, 25))

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    legend = plt.legend()
    handles = getattr(legend, "legend_handles", None) or legend.legendHandles
    for handle in handles:
        handle.set_linewidth(5)
    plt.tight_layout()
    with_leg = output_prefix.with_name(output_prefix.name + "_with_legend.pdf")
    plt.savefig(with_leg, bbox_inches="tight", pad_inches=0)
    legend.remove()
    plt.tight_layout()
    pdf = output_prefix.with_suffix(".pdf")
    plt.savefig(pdf, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {with_leg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bags", type=Path, default=_bag_root())
    ap.add_argument("--v-start", type=float, default=0.25)
    ap.add_argument("--goal-radius", type=float, default=2.0)
    ap.add_argument("--cluster-jump", type=float, default=2.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PDF prefix (default <bags>/success_vs_time)",
    )
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--best",
        type=int,
        default=20,
        help="Keep the N fastest bags per variant (0 = all). Default 20.",
    )
    args = ap.parse_args()
    root = args.bags.expanduser()
    rows = list_variant_bags(root)
    if not rows:
        raise SystemExit(f"no hound_* bags under {root}/{{unidirectional,bidirectional}}")

    results: list[dict[str, Any]] = []
    print(
        f"{'variant':<16} {'bag':<10} {'ok':>3} {'T_s':>8} "
        f"{'path_m':>7} {'v_ws':>6} {'vx':>6} {'dmin':>6} {'WPs':>5}"
    )
    for variant, bag in rows:
        try:
            r = analyze_one(
                bag,
                v_start=float(args.v_start),
                goal_radius=float(args.goal_radius),
                cluster_jump=float(args.cluster_jump),
            )
        except Exception as exc:
            print(f"{variant:<16} {bag.name:<10} ERR {exc}", file=sys.stderr)
            continue
        r["variant"] = variant
        results.append(r)
        t = "—" if r["mission_s"] is None else f"{r['mission_s']:.2f}"
        wall = r.get("mission_wall_s")
        extra = ""
        if r.get("armed_auto_gated") and wall is not None and r["mission_s"] is not None:
            if abs(float(wall) - float(r["mission_s"])) > 0.05:
                extra = f"  wall={float(wall):.2f}"
        print(
            f"{variant:<16} {r['bag']:<10} {'yes' if r['success'] else 'no':>3} "
            f"{t:>8} {r['path_m']:7.1f} {r['wheelspeed_max']:6.2f} "
            f"{r['vx_max']:6.2f} {r['min_dist_to_finish_m']:6.2f} "
            f"{r['n_goal_clusters']}/{r['n_map_wps']}{extra}"
        )

    n_best = int(args.best)
    selected = select_best(results, n_best, float(args.timeout))
    if n_best > 0:
        sel_keys = {(r["variant"], r["bag"]) for r in selected}
        dropped = [r for r in results if (r["variant"], r["bag"]) not in sel_keys]
        print()
        print(f"using best {n_best} bags per variant (fastest mission time; timeout={args.timeout:.0f}s)")
        for r in dropped:
            t = "timeout" if r["mission_s"] is None else f"{r['mission_s']:.2f}s"
            print(f"  drop {r['variant']} {r['bag']}  {t}")
        results = selected

    print()
    for variant in ("unidirectional", "bidirectional"):
        sub = [r for r in results if r["variant"] == variant]
        ok = [
            r["mission_s"]
            for r in sub
            if r["success"]
            and r["mission_s"] is not None
            and float(r["mission_s"]) <= float(args.timeout)
        ]
        mean, std = _mean_std(ok)
        n = len(sub)
        extra = ""
        if math.isfinite(mean):
            extra = f"  mean T {mean:.2f}s  std {std:.2f}s"
        times, succ = timed_outcomes(sub, float(args.timeout))
        n_ok = int(np.sum(succ))
        ws = [
            float(r["wheelspeed_max"])
            for r in sub
            if math.isfinite(float(r.get("wheelspeed_max", float("nan"))))
        ]
        ws_extra = ""
        if ws:
            ws_mean, ws_std = _mean_std(ws)
            ws_extra = (
                f"  max wheelspeed mean {ws_mean:.2f} m/s  peak {max(ws):.2f} m/s"
            )
        print(
            f"{variant}: {n_ok}/{n} success by {args.timeout:.0f}s{extra}{ws_extra}"
        )

    if not args.no_plot:
        prefix = args.output or (root / "success_vs_time")
        plot_success_vs_time(results, float(args.timeout), Path(prefix))

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
