# Stereo Camera Calibration - Refactored

This directory contains a refactored, modular implementation of stereo camera calibration with robust RANSAC-based intrinsic calibration.

## Directory Structure

```
calibration/
├── calibration_refactored.ipynb    # Main calibration notebook (NEW)
├── calibration_backup.ipynb        # Backup of original notebook
├── config.yaml                     # Configuration parameters
├── cache/                          # Cached intermediate results
├── results/                        # Output directory (generated)
│   ├── stereo_calibration.yaml    # Saved calibration parameters
│   └── *.png                      # Visualization outputs
└── utils/                          # Calibration utilities (NEW)
    ├── __init__.py                 # Package exports
    ├── data_structures.py          # Type-safe data containers
    ├── detection.py                # Chessboard detection
    ├── matching.py                 # Stereo board matching
    ├── calibration_robust.py       # Robust calibration algorithms
    ├── visualization.py            # Plotting functions
    ├── io.py                       # Save/load functions
    └── cache.py                    # Caching utilities
```

## Quick Start

### 1. Configure Parameters

Edit `config.yaml` to adjust:
- Data paths
- Chessboard pattern sizes
- Calibration algorithm parameters
- Visualization settings

### 2. Run Calibration

Open and run `calibration_refactored.ipynb` cell by cell. The pipeline:
1. Loads configuration
2. Detects chessboards (parallel processing)
3. Performs robust intrinsic calibration per camera
4. Matches boards between stereo pairs
5. Computes stereo calibration and rectification
6. Exports results to YAML/JSON/NPZ

### 3. Use Calibration Results

```python
from calibration.utils import load_calibration

# Load saved parameters
rig = load_calibration("calibration/results/stereo_calibration.yaml")

# Access camera matrices
K_left = rig.left.K
dist_left = rig.left.dist
K_right = rig.right.K
dist_right = rig.right.dist

# Access stereo parameters
baseline = rig.baseline  # meters
Q = rig.Q  # Disparity-to-depth matrix

# Use rectification maps
import cv2
img_left_rectified = cv2.remap(img_left, rig.left.map1, rig.left.map2, cv2.INTER_LINEAR)
img_right_rectified = cv2.remap(img_right, rig.right.map1, rig.right.map2, cv2.INTER_LINEAR)
```

## Key Features

### Robust Calibration Pipeline

The robust calibration algorithm implements a multi-stage pipeline:

1. **Stage 0**: Per-image provisional filtering (optional)
2. **Geometric Deduplication**: Remove duplicate detections using spatial hashing
3. **Extrinsic Deduplication**: Cluster similar poses and keep best representatives
4. **Diversity Sampling**: Ensure good pose coverage across calibration volume
5. **RANSAC**: Find optimal intrinsic parameters via random subset sampling
6. **Post-Rejection**: Tighten inlier set for final calibration

This approach significantly improves calibration accuracy and robustness compared to naive `cv2.calibrateCamera`.

### Robust Stereo Calibration

Similar to the intrinsic pipeline, the stereo calibration now supports a RANSAC-like procedure (`calibrate_stereo_robust`) that:
1. Iteratively calibrates on random subsets of stereo pairs.
2. Evaluates the model on the full dataset.
3. Selects the subset that minimizes global RMS error.

### Parallel Processing

Chessboard detection supports parallel processing across multiple CPU cores:

```yaml
# config.yaml
detection:
  parallel_enabled: true
  num_workers: -1  # -1 = use all cores
```

### Result Caching

Enable caching to avoid re-running expensive computations:

```yaml
# config.yaml
data:
  cache_dir: "calibration/cache"
```

Cached results are automatically invalidated if input data changes.

### Flexible Visualization

Control visualization output mode:

```yaml
# config.yaml
visualization:
  mode: "save"  # "show", "save", or "both"
```

- **save**: Save figures to files (default, good for batch processing)
- **show**: Display interactive matplotlib windows
- **both**: Save AND display

## Module Documentation

### data_structures.py

Type-safe containers using dataclasses:

- **CameraIndex**: Enum for LEFT (0) and RIGHT (1)
- **CameraCalibration**: Single camera data
- **StereoRig**: Complete stereo rig
- **RobustCalibrationResult**: Calibration statistics
- **CalibrationConfig**: YAML configuration wrapper

### detection.py

Chessboard detection with multi-board support:

- `find_all_chessboards()`: Detect multiple boards in single image
- `detect_chessboards_parallel()`: Process multiple images in parallel
- `filter_detections_by_image_count()`: Quality filtering

### matching.py

Stereo board matching algorithms:

- `match_chessboards()`: Match detections between left/right images
  - Supports "greedy" (O(N log N)) and "hungarian" (O(N³)) methods
  - Uses epipolar constraints and scale consistency

### calibration_robust.py

Core calibration algorithms:

- `calibrate_intrinsics_robust()`: RANSAC-based intrinsic calibration
- `calibrate_stereo()`: Stereo calibration with rectification
- `calibrate_stereo_robust()`: RANSAC-based stereo calibration
- Automatic result caching support

### visualization.py

Plotting functions with save/show modes:

- `plot_undistortion_comparison()`: Before/after undistortion
- `plot_imagepoints_heatmap()`: Calibration point coverage
- `plot_matched_boards()`: Stereo matching visualization
- `visualize_stereo_matches()`: High-level stereo match visualization
- `plot_rectification_preview()`: Rectified epipolar alignment

### io.py

Serialization and deserialization:

- `save_calibration()`: Export to YAML/JSON/NPZ
- `load_calibration()`: Load from any supported format
- `load_kitti_calibration()`: Import KITTI-style `calib_cam_to_cam.txt`
- Human-readable YAML output for easy inspection

### cache.py

Centralized caching utilities:

- `save_to_cache()`: Save intermediate results to pickle
- `load_from_cache()`: Load cached results with invalidation checks
- `clear_cache()`: Utility to clean cache directory

## Configuration Reference

See `config.yaml` for full parameter documentation. Key sections:

- **data**: Paths and directories
- **detection**: Chessboard detection parameters
- **robust_calibration**: RANSAC and filtering thresholds
- **matching**: Stereo matching constraints
- **stereo**: Stereo calibration settings
- **visualization**: Plot output configuration
- **export**: Result export format

## Testing

Run unit tests:

```bash
pytest tests/test_calibration.py -v
```

## Comparison with Original Notebook

| Aspect | Original | Refactored |
|--------|----------|------------|
| **Lines of code** | ~1500 | ~300 (notebook) + ~1500 (utils) |
| **Cell count** | 41 | 15 |
| **Modularity** | Monolithic cells | Reusable modules |
| **Configuration** | Hardcoded | YAML file |
| **Parallel processing** | No | Yes |
| **Caching** | No | Yes |
| **Visualization control** | Manual | Configurable modes |
| **Type safety** | Dict-based | Dataclasses |
| **Documentation** | Minimal | Comprehensive |
| **Testability** | Difficult | Unit tests included |

## Migration from Original Notebook

If you have existing code using the old `cam` dictionary structure:

```python
# Old way
K_left = cam["K"][LEFT]
dist_left = cam["dist"][LEFT]

# New way
from calibration.utils import CameraIndex
K_left = rig.left.K
dist_left = rig.left.dist
# Or: rig.get_camera(CameraIndex.LEFT).K
```

## Troubleshooting

**Import errors**: Ensure you're running from project root or have added calibration to Python path.

**Detection finds no boards**: Check pattern_sizes in config.yaml match your physical chessboard.

**RANSAC takes too long**: Reduce `ransac_iterations` or `max_diverse_samples` in config.

**Out of memory**: Reduce `num_workers` or disable parallel processing.

## Citation

If you use this calibration pipeline in your research, please cite the original OpenCV calibration work and this refactored implementation.

## License

Same as parent project.
