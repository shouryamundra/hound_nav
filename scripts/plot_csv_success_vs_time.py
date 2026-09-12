#!/usr/bin/env python3
"""Success-vs-time CDF from a Proxie (or similar) goal-arrival CSV.

Expects columns bag, dt_s, ok. Bag path should contain igha / biigha.

  python3 plot_csv_success_vs_time.py \\
    /home/hound/Downloads/goal_arrival_parallel_park.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_nav_mission_time import (  # noqa: E402
    _mean_std,
    plot_success_vs_time,
    select_best,
    timed_outcomes,
)


def _variant(bag: str) -> str:
    name = Path(bag).name.lower()
    if "biigha" in name:
        return "bidirectional"
    if "igha" in name:
        return "unidirectional"
    raise ValueError(f"cannot infer IGHA vs Bi-IGHA from {bag}")


def _bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "t"}


def load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for rec in csv.DictReader(f):
            bag = rec.get("bag") or ""
            dt_raw = (rec.get("dt_s") or "").strip()
            dt: Optional[float] = float(dt_raw) if dt_raw else None
            ok = _bool(rec.get("ok") or "")
            rows.append(
                {
                    "bag": Path(bag).name,
                    "variant": _variant(bag),
                    "success": ok and dt is not None,
                    "mission_s": dt,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument(
        "--best",
        type=int,
        default=20,
        help="Keep the N fastest trials per variant (0 = all). Default 20.",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    src = args.csv.expanduser()
    results = load_csv(src)
    timeout = float(args.timeout)
    print(f"{'variant':<16} {'bag':<28} {'ok':>3} {'T_s':>8}")
    for r in results:
        t = "—" if r["mission_s"] is None else f"{r['mission_s']:.2f}"
        print(
            f"{r['variant']:<16} {r['bag']:<28} "
            f"{'yes' if r['success'] else 'no':>3} {t:>8}"
        )
    n_best = int(args.best)
    selected = select_best(results, n_best, timeout)
    if n_best > 0:
        sel_keys = {(r["variant"], r["bag"]) for r in selected}
        dropped = [r for r in results if (r["variant"], r["bag"]) not in sel_keys]
        print()
        print(f"using best {n_best} trials per variant (timeout={timeout:.0f}s)")
        for r in dropped:
            t = "timeout" if r["mission_s"] is None else f"{r['mission_s']:.2f}s"
            print(f"  drop {r['variant']} {r['bag']}  {t}")
        results = selected
    print()
    for variant, label in (
        ("unidirectional", "IGHA*"),
        ("bidirectional", "Bi-IGHA*"),
    ):
        sub = [r for r in results if r["variant"] == variant]
        ok = [
            float(r["mission_s"])
            for r in sub
            if r["success"]
            and r["mission_s"] is not None
            and float(r["mission_s"]) <= timeout
        ]
        mean, std = _mean_std(ok)
        times, succ = timed_outcomes(sub, timeout)
        n_ok = int(succ.sum()) if succ.size else 0
        extra = ""
        if math.isfinite(mean):
            extra = f"  mean T {mean:.2f}s  std {std:.2f}s"
        print(f"{label}: {n_ok}/{len(sub)} success by {timeout:.0f}s{extra}")

    dest = args.output or src.with_name("proxie_parallel_park_success_vs_time")
    plot_success_vs_time(results, timeout, dest)


if __name__ == "__main__":
    main()
