"""Torch CUDA extension cache under ``colcon_ws/cache/`` (survives Docker restarts).

Set ``TORCH_EXTENSIONS_DIR`` before any ``torch.utils.cpp_extension.load``.
Override with that env var or ``ROS_WORKSPACE``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def repo_root() -> Path:
    # hound_nav/hound_nav/cuda_cache.py → hound_nav/
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    env = str(os.environ.get("ROS_WORKSPACE") or "").strip()
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    for cand in (Path("/root/colcon_ws"), Path("/home/hound/colcon_ws")):
        if cand.is_dir():
            return cand
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "colcon_ws":
            return parent
        if (parent / "src" / "hound_nav").is_dir() and (parent / "src" / "hound_core").is_dir():
            return parent
    return Path("/root/colcon_ws")


def cache_root() -> Path:
    return workspace_root() / "cache"


def torch_extensions_dir() -> Path:
    env = str(os.environ.get("TORCH_EXTENSIONS_DIR") or "").strip()
    if env:
        return Path(env)
    return cache_root() / "torch_extensions"


def _migrate_old_package_cache(dest: Path) -> None:
    """Copy leftover ``hound_nav/cache/torch_extensions`` if the workspace cache is empty."""
    old = repo_root() / "cache" / "torch_extensions"
    if not old.is_dir() or old.resolve() == dest.resolve():
        return
    if dest.exists() and any(dest.iterdir()):
        return
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(old, dest, dirs_exist_ok=True)
    except OSError:
        pass


def ensure_torch_extensions_dir(*, clean: bool = False) -> Path:
    """Point PyTorch JIT builds at ``$ROS_WORKSPACE/cache/torch_extensions``."""
    d = torch_extensions_dir()
    if clean and d.exists():
        shutil.rmtree(d)
    if not clean:
        _migrate_old_package_cache(d)
    d.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(d)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    return d


def _any_source_newer(sources: list[Path], so: Path) -> bool:
    so_m = so.stat().st_mtime
    for src in sources:
        try:
            if src.is_file() and src.stat().st_mtime > so_m + 1e-3:
                return True
        except OSError:
            return True
    return False


def _import_so(name: str, so: Path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, str(so))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_extension(name: str, sources, **kwargs):
    """``cpp_extension.load``, but reuse ``name.so`` if our sources are older.

    Fresh Docker overlays give ``/usr`` / Torch headers new mtimes, so ninja
    would otherwise rebuild every container start. Ignore those; only rebuild
    when a listed source is newer than the ``.so``, or
    ``TORCH_EXTENSIONS_FORCE_REBUILD=1``.
    """
    from torch.utils.cpp_extension import load

    root = ensure_torch_extensions_dir()
    build_dir = root / str(name)
    so = build_dir / f"{name}.so"
    srcs = [Path(s) for s in sources]
    force = str(os.environ.get("TORCH_EXTENSIONS_FORCE_REBUILD", "")).lower() in (
        "1",
        "true",
        "yes",
    )
    if so.is_file() and not force and not _any_source_newer(srcs, so):
        print(f"[cuda_cache] reuse {so}", flush=True)
        return _import_so(name, so)
    build_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(kwargs)
    kwargs.setdefault("build_directory", str(build_dir))
    kwargs.setdefault("verbose", False)
    print(f"[cuda_cache] compile {name} → {build_dir}", flush=True)
    return load(name=name, sources=list(map(str, sources)), **kwargs)
