#!/usr/bin/env python3
"""Shared helpers for nav-bag video names and experiment pickles."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import yaml

PACIFIC = ZoneInfo("America/Los_Angeles")

VARIANT_LABEL = {
    "unidirectional": "IGHA*",
    "bidirectional": "Bi-IGHA*",
}

VARIANT_PREFIX = {
    "unidirectional": "IGHA",
    "bidirectional": "Bi-IGHA",
}

DEFAULT_BAG_ROOTS = (
    Path("/root/colcon_ws/bags_nav"),
    Path("/home/hound/colcon_ws/bags_nav"),
)


def bag_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
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


def variant_of(bag: Path) -> str:
    parts = {p.name for p in bag.resolve().parents}
    if "unidirectional" in parts:
        return "unidirectional"
    if "bidirectional" in parts:
        return "bidirectional"
    raise ValueError(f"cannot infer uni/bi from {bag}")


def hud_label(variant: str) -> str:
    return VARIANT_LABEL[variant]


def file_prefix(variant: str) -> str:
    return VARIANT_PREFIX[variant]


def bag_start_ns(bag: Path) -> int:
    info = yaml.safe_load((bag / "metadata.yaml").read_text())
    return int(
        info["rosbag2_bagfile_information"]["starting_time"][
            "nanoseconds_since_epoch"
        ]
    )


def bag_start_utc(bag: Path) -> datetime:
    return datetime.fromtimestamp(bag_start_ns(bag) * 1e-9, tz=timezone.utc)


def bag_start_pacific(bag: Path) -> datetime:
    return bag_start_utc(bag).astimezone(PACIFIC)


def pacific_stamp(bag: Path) -> str:
    """YYYYMMDD_HHMMSS in America/Los_Angeles (PDT in September)."""
    return bag_start_pacific(bag).strftime("%Y%m%d_%H%M%S")


def export_stem(variant: str, bag: Path) -> str:
    return f"{file_prefix(variant)}_{pacific_stamp(bag)}"


def unique_stem(stem: str, used: set[str], bag: Path) -> str:
    if stem not in used:
        used.add(stem)
        return stem
    alt = f"{stem}_{bag.name}"
    used.add(alt)
    return alt
