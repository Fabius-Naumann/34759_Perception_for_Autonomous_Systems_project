# Caching Guide

This document explains the caching system implemented in the calibration pipeline to speed up repeated runs during experimentation.

## Overview

The calibration pipeline includes three expensive operations that now support caching:

1. **Chessboard Detection** (`detect_chessboards_parallel`)
2. **Stereo Pair Matching** (`build_stereo_pairs_from_detections`)
3. **Intrinsic Calibration** (`calibrate_intrinsics_robust`)

## How Caching Works

### Cache Directory

All cache files are stored in the directory specified by `data.cache_dir` in `config.yaml` (default: `calibration/cache/`).

### Cache Keys

Each cached result is identified by an MD5 hash of:
- Input data (image paths, pattern sizes, calibration parameters)
- File modification times (for detection)
- Configuration parameters

When parameters or input data change, a new cache key is generated and the operation is re-run.

### Cache Files

Cache files are stored as pickle files with names like:
- `detection_<hash>.pkl` - Chessboard detection results
- `stereo_matching_<hash>.pkl` - Stereo pair matching results
- `intrinsic_<hash>.pkl` - Intrinsic calibration results

## Using Caching

### In Notebook

Caching is automatically enabled when you define the cache directory:

```python
# Cell 2 (in notebook)
cache_dir = proj_root / "calibration" / "cache"
cache_dir.mkdir(exist_ok=True, parents=True)
```

Then pass `cache_dir` to functions that support caching:

```python
# Detection with caching
corners, objpoints, indices, size = detect_chessboards_parallel(
    image_paths,
    pattern_sizes,
    cache_dir=cache_dir  # Enable caching
)

# Stereo matching with caching
stereo_objpoints, stereo_imgpoints_left, stereo_imgpoints_right, metadata = (
    build_stereo_pairs_from_detections(
        corners_left, objpoints_left, indices_left,
        corners_right, objpoints_right, indices_right,
        cache_dir=cache_dir  # Enable caching
    )
)

# Intrinsic calibration with caching
result = calibrate_intrinsics_robust(
    corners, objpoints, image_size,
    cache_dir=cache_dir  # Enable caching
)
```

### In Scripts

```python
from pathlib import Path
from utils import detect_chessboards_parallel, calibrate_intrinsics_robust

# Define cache directory
cache_dir = Path("calibration/cache")
cache_dir.mkdir(exist_ok=True, parents=True)

# Use caching by passing cache_dir parameter
corners, objpoints, indices, size = detect_chessboards_parallel(
    image_paths,
    pattern_sizes,
    cache_dir=cache_dir
)
```

### Disabling Caching

To disable caching, simply omit the `cache_dir` parameter or pass `None`:

```python
# No caching
corners, objpoints, indices, size = detect_chessboards_parallel(
    image_paths,
    pattern_sizes,
    cache_dir=None  # or just omit this parameter
)
```

## Cache Invalidation

Caches are automatically invalidated when:

1. **Detection**: Image files are modified (based on mtime)
2. **Detection**: Pattern sizes or max_boards parameter changes
3. **Stereo matching**: Detection results change (different boards detected)
4. **Stereo matching**: Matching parameters change (dy_thresh, scale_ratio_thresh, method)
5. **Intrinsic calibration**: Any calibration parameter changes
6. **Intrinsic calibration**: Input corners/objpoints change

Manual cache clearing:
```powershell
# Clear all caches
Remove-Item -Recurse -Force calibration/cache/*

# Clear only detection caches
Remove-Item calibration/cache/detection_*.pkl

# Clear only stereo matching caches
Remove-Item calibration/cache/stereo_matching_*.pkl

# Clear only intrinsic calibration caches
Remove-Item calibration/cache/intrinsic_*.pkl
```

## Performance Benefits

Typical speedups with caching (on subsequent runs):

| Operation | Without Cache | With Cache | Speedup |
|-----------|--------------|------------|---------|
| Detection (200 images) | ~30-60s | ~0.5s | 60-120x |
| Stereo matching (200 pairs) | ~10-20s | ~0.2s | 50-100x |
| Intrinsic calibration | ~10-20s | ~0.1s | 100-200x |
| **Total pipeline** | ~60-120s | ~1-2s | **60-120x** |

## Implementation Details

### Detection Caching (`detection.py`)

The `_compute_detection_cache_key()` function creates a hash from:
- Sorted list of image paths
- File modification times for each image
- Pattern sizes (sorted)
- max_boards parameter

Cache hit/miss messages:
- Hit: `Loaded cached detection results (key=abc12345...)`
- Miss: Operation runs normally, then `Cached detection results (key=abc12345...)`

### Stereo Matching Caching (`matching.py`)

The `_compute_matching_cache_key()` function creates a hash from:
- Detection result counts and image indices
- Sample of corner data (every 10th board)
- Matching parameters (dy_thresh, scale_ratio_thresh, method)

This efficiently detects changes in detection results without hashing all corner data.

Cache hit/miss messages:
- Hit: `Loaded cached stereo matching results (key=abc12345...)`
- Miss: Operation runs normally, then `Cached stereo matching results (key=abc12345...)`

### Intrinsic Calibration Caching (`calibration_robust.py`)

The `_compute_cache_key()` function creates a hash from:
- All input corners and objpoints (serialized)
- Image size
- All calibration stage parameters (deduplicate, RANSAC settings, etc.)

Cache validation includes:
- Checking if pickle file exists and is readable
- Verifying result structure matches expected format
- Falls back to recomputation on any error

## Troubleshooting

### Cache Not Being Used

If you see the operation running every time despite caching:

1. **Check cache directory exists**: `cache_dir.mkdir(exist_ok=True, parents=True)`
2. **Verify parameter consistency**: Changing any parameter invalidates the cache
3. **Check file permissions**: Cache directory must be writable
4. **Look for cache messages**: Progress messages indicate cache hit/miss

### Cache Files Growing Too Large

Detection cache files are typically 1-10 MB depending on image count.
Stereo matching cache files are typically 1-5 MB depending on matched board count.
Intrinsic calibration cache files are typically 100-500 KB.

To manage cache size:
```powershell
# Show cache directory size
Get-ChildItem calibration/cache | Measure-Object -Property Length -Sum

# Remove old cache files (older than 7 days)
Get-ChildItem calibration/cache -Recurse | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item
```

### Cache Corruption

If you suspect cache corruption:

1. Delete the specific cache file
2. Or clear entire cache directory
3. Re-run the operation

The system will automatically regenerate valid cache files.

## Future Enhancements

Potential improvements:

1. **Stereo calibration cache**: Cache final stereo calibration results
2. **Cache expiration**: Automatic cleanup of old cache files
3. **Cache statistics**: Track hit/miss rates and performance gains
4. **Compression**: Use compressed pickle for smaller cache files
5. **Partial invalidation**: Smart cache updates when only some images change
