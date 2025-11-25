"""
Centralized caching utilities for expensive calibration operations.

Provides common patterns for cache key computation, loading, and saving
while allowing each operation to define its own key hashing logic.
"""

import pickle
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def load_from_cache(
    cache_dir: Path | None,
    cache_key: str,
    cache_prefix: str,
    progress: bool = False,
) -> Any | None:
    """
    Load cached result from disk.

    Args:
        cache_dir: Directory containing cache files
        cache_key: MD5 hash identifying this cached result
        cache_prefix: Prefix for cache filename (e.g., "detection", "stereo_matching")
        progress: Whether to print progress message

    Returns:
        Cached data if available and loadable, None otherwise
    """
    if cache_dir is None:
        return None

    cache_file = cache_dir / f"{cache_prefix}_{cache_key}.pkl"
    if not cache_file.exists():
        return None

    try:
        with Path(cache_file).open("rb") as f:
            cached = pickle.load(f)
        if progress:
            print(f"Loaded cached {cache_prefix} results (key={cache_key[:8]}...)")
        return cached
    except Exception:
        # Silently fail and recompute
        return None


def save_to_cache(
    cache_dir: Path | None,
    cache_key: str,
    cache_prefix: str,
    data: Any,
    progress: bool = False,
) -> None:
    """
    Save result to cache on disk.

    Args:
        cache_dir: Directory to store cache files
        cache_key: MD5 hash identifying this cached result
        cache_prefix: Prefix for cache filename (e.g., "detection", "stereo_matching")
        data: Data to cache (must be pickle-serializable)
        progress: Whether to print progress message
    """
    if cache_dir is None:
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_prefix}_{cache_key}.pkl"

    try:
        with Path(cache_file).open("wb") as f:
            pickle.dump(data, f)
        if progress:
            print(f"Cached {cache_prefix} results (key={cache_key[:8]}...)")
    except Exception as e:
        print(f"Warning: Failed to cache {cache_prefix} results: {e}")
