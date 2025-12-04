"""
Visualization utilities for calibration results.

Provides plotting functions with support for both display and file saving modes.
"""

import random
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from .matching import group_detections_by_image


def _get_output_mode(config_mode: str | None, override_mode: str | None) -> str:
    """Determine output mode from config and override."""
    if override_mode is not None:
        return override_mode
    if config_mode is not None:
        return config_mode
    return "save"


def _handle_figure_output(fig: plt.Figure, output_path: Path | None, mode: str) -> None:
    """Handle figure display or saving based on mode."""
    if mode in ("save", "both") and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if mode in ("show", "both"):
        plt.show()
    else:
        plt.close(fig)


def plot_undistortion_comparison(
    image_path: str,
    K: np.ndarray,
    dist: np.ndarray,
    camera_name: str = "Camera",
    output_path: Path | None = None,
    mode: str = "save",
    figsize: tuple[float, float] = (12, 5),
) -> None:
    """
    Plot side-by-side comparison of original and undistorted image.

    Args:
        image_path: Path to test image
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        camera_name: Name for plot title
        output_path: Path to save figure (if mode includes 'save')
        mode: 'show', 'save', or 'both'
        figsize: Figure size
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    und = cv2.undistort(img, K, dist)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"{camera_name} Original")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"{camera_name} Undistorted")
    ax[1].axis("off")
    plt.tight_layout()

    _handle_figure_output(fig, output_path, mode)


def plot_imagepoints_heatmap(
    final_imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
    camera_name: str = "Camera",
    output_path: Path | None = None,
    mode: str = "save",
    bins: int = 50,
    figsize: tuple[float, float] = (36, 5),
) -> None:
    """
    Create heatmap visualization of calibration point distribution.

    Args:
        final_imgpoints: List of image point arrays from calibration
        image_size: Image dimensions (width, height)
        camera_name: Name for plot title
        output_path: Path to save figure
        mode: 'show', 'save', or 'both'
        bins: Number of bins for 2D histogram
        figsize: Figure size
    """
    if not final_imgpoints:
        print(f"No image points to visualize for {camera_name}")
        return

    # Collect all points
    all_points = []
    for corners in final_imgpoints:
        pts = corners.reshape(-1, 2)
        all_points.append(pts)

    all_points = np.vstack(all_points)
    x_coords = all_points[:, 0]
    y_coords = all_points[:, 1]

    fig = plt.figure(figsize=figsize)

    # 1. Heatmap
    ax1 = plt.subplot(131)
    h, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=bins, range=[[0, image_size[0]], [0, image_size[1]]])
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    im1 = ax1.imshow(h.T, extent=extent, origin="upper", cmap="hot", aspect="auto", interpolation="bilinear")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.set_title(f"{camera_name}: Point Density\n({len(all_points)} points from {len(final_imgpoints)} boards)")
    plt.colorbar(im1, ax=ax1, label="Point Count")
    ax1.invert_yaxis()

    # 2. Scatter
    ax2 = plt.subplot(132)
    ax2.scatter(x_coords, y_coords, c="cyan", s=2, alpha=0.6, edgecolors="none")
    ax2.set_xlim(0, image_size[0])
    ax2.set_ylim(image_size[1], 0)
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.set_title(f"{camera_name}: All Points")
    ax2.grid(True, alpha=0.3)

    # 3. Centroids
    ax3 = plt.subplot(133)
    centroids = np.array([corners.reshape(-1, 2).mean(axis=0) for corners in final_imgpoints])
    h_cent, xe_cent, ye_cent = np.histogram2d(
        centroids[:, 0], centroids[:, 1], bins=20, range=[[0, image_size[0]], [0, image_size[1]]]
    )
    xi = np.digitize(centroids[:, 0], xe_cent) - 1
    yi = np.digitize(centroids[:, 1], ye_cent) - 1
    xi = np.clip(xi, 0, h_cent.shape[0] - 1)
    yi = np.clip(yi, 0, h_cent.shape[1] - 1)
    density_values = h_cent[xi, yi]
    scatter = ax3.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c=density_values,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    ax3.set_xlim(0, image_size[0])
    ax3.set_ylim(image_size[1], 0)
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Y (pixels)")
    ax3.set_title(f"{camera_name}: Board Centroids ({len(centroids)} boards)")
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label="Local Density")

    plt.tight_layout()
    _handle_figure_output(fig, output_path, mode)

    # Print statistics
    print(f"\n{camera_name} Coverage Statistics:")
    print(f"  Total points: {len(all_points)}")
    print(f"  Total boards: {len(final_imgpoints)}")
    print(
        f"  X range: [{x_coords.min():.1f}, {x_coords.max():.1f}] "
        f"(coverage: {(x_coords.max() - x_coords.min()) / image_size[0] * 100:.1f}%)"
    )
    print(
        f"  Y range: [{y_coords.min():.1f}, {y_coords.max():.1f}] "
        f"(coverage: {(y_coords.max() - y_coords.min()) / image_size[1] * 100:.1f}%)"
    )


def plot_matched_boards(
    image_pair_index: int,
    imgL: np.ndarray,
    imgR: np.ndarray,
    dets_L: list[tuple[np.ndarray, tuple[int, int]]],
    dets_R: list[tuple[np.ndarray, tuple[int, int]]],
    matches: list[tuple[int, int]],
    unmatched_L: list[int],
    unmatched_R: list[int],
    output_path: Path | None = None,
    mode: str = "save",
    figsize: tuple[float, float] = (16, 7),
) -> None:
    """
    Visualize matched and unmatched chessboard centroids.

    Args:
        image_pair_index: Image pair number for title
        imgL, imgR: Left and right images
        dets_L, dets_R: Detection lists (corners, size) for each camera
        matches: List of (left_idx, right_idx) matches
        unmatched_L, unmatched_R: Lists of unmatched indices
        output_path: Path to save figure
        mode: 'show', 'save', or 'both'
        figsize: Figure size
    """
    visL = imgL.copy()
    visR = imgR.copy()
    cmap = cm.get_cmap("tab20")

    def centroid(c):
        pts = c.reshape(-1, 2)
        return pts.mean(axis=0)

    # Draw matched centers with consistent colors
    for idx, (iL, iR) in enumerate(matches):
        color = tuple(int(255 * v) for v in cmap(idx % 20)[:3])
        cL = centroid(dets_L[iL][0])
        cR = centroid(dets_R[iR][0])
        cv2.circle(visL, (int(cL[0]), int(cL[1])), 8, color, thickness=2)
        cv2.circle(visR, (int(cR[0]), int(cR[1])), 8, color, thickness=2)
        cv2.putText(
            visL, f"{idx}", (int(cL[0]) + 6, int(cL[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
        cv2.putText(
            visR, f"{idx}", (int(cR[0]) + 6, int(cR[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    # Draw unmatched: red on left-only, blue on right-only
    for i in unmatched_L:
        c = centroid(dets_L[i][0])
        cv2.circle(visL, (int(c[0]), int(c[1])), 8, (0, 0, 255), thickness=2)
    for j in unmatched_R:
        c = centroid(dets_R[j][0])
        cv2.circle(visR, (int(c[0]), int(c[1])), 8, (255, 0, 0), thickness=2)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(cv2.cvtColor(visL, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Left k={image_pair_index} — matched={len(matches)} unmatchedL={len(unmatched_L)}")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(visR, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Right k={image_pair_index} — matched={len(matches)} unmatchedR={len(unmatched_R)}")
    ax[1].axis("off")
    plt.tight_layout()

    _handle_figure_output(fig, output_path, mode)


def plot_rectification_preview(
    image_left_path: str,
    image_right_path: str,
    map1_left: np.ndarray,
    map2_left: np.ndarray,
    map1_right: np.ndarray,
    map2_right: np.ndarray,
    roi_left: tuple[int, int, int, int] | None = None,
    roi_right: tuple[int, int, int, int] | None = None,
    n_guides: int = 10,
    output_path: Path | None = None,
    mode: str = "save",
    figsize: tuple[float, float] = (16, 7),
) -> None:
    """
    Visualize rectified stereo pair with epipolar line guides.

    Args:
        image_left_path, image_right_path: Paths to test images
        map1_left, map2_left: Left rectification maps
        map1_right, map2_right: Right rectification maps
        roi_left, roi_right: Valid ROI tuples (x, y, w, h)
        n_guides: Number of horizontal guide lines
        output_path: Path to save figure
        mode: 'show', 'save', or 'both'
        figsize: Figure size
    """
    imgL = cv2.imread(image_left_path)
    imgR = cv2.imread(image_right_path)
    if imgL is None or imgR is None:
        print("Could not read images for rectification preview.")
        return

    # Remap
    rectL = cv2.remap(imgL, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

    h, _w = rectL.shape[:2]
    ys = np.linspace(0, h - 1, num=max(2, n_guides), dtype=int)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Rectified Left")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Rectified Right")
    ax[1].axis("off")

    # Draw horizontal guide lines
    for y in ys:
        ax[0].axhline(y=y, color="lime", linewidth=0.5, alpha=0.5)
        ax[1].axhline(y=y, color="lime", linewidth=0.5, alpha=0.5)

    # Draw valid ROI rectangles
    if roi_left is not None:
        x, y, w_roi, h_roi = roi_left
        rect = patches.Rectangle((x, y), w_roi, h_roi, linewidth=2, edgecolor="red", facecolor="none")
        ax[0].add_patch(rect)
    if roi_right is not None:
        x, y, w_roi, h_roi = roi_right
        rect = patches.Rectangle((x, y), w_roi, h_roi, linewidth=2, edgecolor="red", facecolor="none")
        ax[1].add_patch(rect)

    plt.tight_layout()
    _handle_figure_output(fig, output_path, mode)


def plot_stereo_pair_coverage(
    imgpoints_left: list[np.ndarray],
    imgpoints_right: list[np.ndarray],
    image_size: tuple[int, int],
    kept_indices: list[int] | None = None,
    output_path: Path | None = None,
    mode: str = "save",
    figsize: tuple[float, float] = (16, 8),
) -> None:
    """
    Visualize spatial coverage of stereo pairs showing centroids in both images.

    Shows where matched boards are located in left/right images, optionally
    highlighting which pairs were kept after subsampling.

    Args:
        imgpoints_left: List of left image point arrays
        imgpoints_right: List of right image point arrays
        image_size: Image dimensions (width, height)
        kept_indices: Optional list of indices that were kept after subsampling
        output_path: Path to save figure
        mode: 'show', 'save', or 'both'
        figsize: Figure size
    """
    # Compute centroids for all pairs
    centroids_left = []
    centroids_right = []
    for img_L, img_R in zip(imgpoints_left, imgpoints_right, strict=False):
        pts_L = img_L.reshape(-1, 2)
        pts_R = img_R.reshape(-1, 2)
        centroids_left.append(pts_L.mean(axis=0))
        centroids_right.append(pts_R.mean(axis=0))

    centroids_left = np.array(centroids_left)
    centroids_right = np.array(centroids_right)

    # Determine which pairs are kept vs discarded
    if kept_indices is not None:
        kept_set = set(kept_indices)
        kept_mask = np.array([i in kept_set for i in range(len(centroids_left))])
    else:
        kept_mask = np.ones(len(centroids_left), dtype=bool)

    discarded_mask = ~kept_mask

    # Create figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
    w, h = image_size

    # Plot LEFT camera
    ax_left.set_xlim(0, w)
    ax_left.set_ylim(h, 0)
    ax_left.set_aspect("equal")
    ax_left.set_title(
        f"Left Camera Coverage ({kept_mask.sum()}/{len(centroids_left)} pairs)", fontsize=12, fontweight="bold"
    )
    ax_left.set_xlabel("X (pixels)")
    ax_left.set_ylabel("Y (pixels)")
    ax_left.grid(True, alpha=0.3)

    # Plot discarded pairs (gray)
    if discarded_mask.any():
        ax_left.scatter(
            centroids_left[discarded_mask, 0],
            centroids_left[discarded_mask, 1],
            c="lightgray",
            s=60,
            alpha=0.4,
            edgecolors="gray",
            linewidths=0.5,
            label="Discarded",
            zorder=1,
        )

    # Plot kept pairs (colored)
    if kept_mask.any():
        ax_left.scatter(
            centroids_left[kept_mask, 0],
            centroids_left[kept_mask, 1],
            c="steelblue",
            s=80,
            alpha=0.7,
            edgecolors="darkblue",
            linewidths=1,
            label="Kept",
            zorder=2,
        )
        # Add indices
        for i in np.where(kept_mask)[0]:
            ax_left.text(
                centroids_left[i, 0],
                centroids_left[i, 1],
                str(i),
                fontsize=8,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                zorder=3,
            )

    ax_left.legend(loc="upper right")

    # Plot RIGHT camera
    ax_right.set_xlim(0, w)
    ax_right.set_ylim(h, 0)
    ax_right.set_aspect("equal")
    ax_right.set_title(
        f"Right Camera Coverage ({kept_mask.sum()}/{len(centroids_right)} pairs)", fontsize=12, fontweight="bold"
    )
    ax_right.set_xlabel("X (pixels)")
    ax_right.set_ylabel("Y (pixels)")
    ax_right.grid(True, alpha=0.3)

    # Plot discarded pairs (gray)
    if discarded_mask.any():
        ax_right.scatter(
            centroids_right[discarded_mask, 0],
            centroids_right[discarded_mask, 1],
            c="lightgray",
            s=60,
            alpha=0.4,
            edgecolors="gray",
            linewidths=0.5,
            label="Discarded",
            zorder=1,
        )

    # Plot kept pairs (colored)
    if kept_mask.any():
        ax_right.scatter(
            centroids_right[kept_mask, 0],
            centroids_right[kept_mask, 1],
            c="steelblue",
            s=80,
            alpha=0.7,
            edgecolors="darkblue",
            linewidths=1,
            label="Kept",
            zorder=2,
        )
        # Add indices
        for i in np.where(kept_mask)[0]:
            ax_right.text(
                centroids_right[i, 0],
                centroids_right[i, 1],
                str(i),
                fontsize=8,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                zorder=3,
            )

    ax_right.legend(loc="upper right")

    plt.tight_layout()
    _handle_figure_output(fig, output_path, mode)


def create_undistortion_gallery(
    image_paths: list[str],
    K: np.ndarray,
    dist: np.ndarray,
    camera_name: str,
    output_dir: Path,
    num_samples: int = 5,
    mode: str = "save",
) -> None:
    """
    Create a gallery of undistortion examples.

    Args:
        image_paths: List of calibration image paths
        K, dist: Camera parameters
        camera_name: Camera identifier
        output_dir: Directory to save individual images
        num_samples: Number of random samples to process
        mode: Output mode
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))

    for idx, path in enumerate(samples):
        output_path = output_dir / f"undistort_{camera_name}_{idx}.png"
        plot_undistortion_comparison(path, K, dist, camera_name, output_path, mode)


def visualize_stereo_matches(
    corners_left: list[np.ndarray],
    objpoints_left: list[np.ndarray],
    indices_left: list[int],
    corners_right: list[np.ndarray],
    objpoints_right: list[np.ndarray],
    indices_right: list[int],
    images_left: list[Path],
    images_right: list[Path],
    matching_metadata: dict,
    output_dir: Path,
    vis_mode: str = "save",
    max_pairs: int = 3,
) -> None:
    """
    Visualize matched stereo pairs.

    Args:
        corners_left, objpoints_left, indices_left: Left detection data
        corners_right, objpoints_right, indices_right: Right detection data
        images_left, images_right: Lists of image paths
        matching_metadata: Metadata returned by build_stereo_pairs_from_detections
        output_dir: Directory to save visualizations
        vis_mode: Visualization mode ('save', 'show', 'both')
        max_pairs: Maximum number of pairs to visualize
    """
    # Group detections for easy access
    left_by_image = group_detections_by_image(corners_left, objpoints_left, indices_left)
    right_by_image = group_detections_by_image(corners_right, objpoints_right, indices_right)

    # Select pairs to visualize (e.g., first, middle, last)
    pair_details = matching_metadata.get("pair_details", {})
    available_indices = sorted(pair_details.keys())

    if not available_indices:
        print("No matching details available for visualization")
        return

    # Pick indices
    if len(available_indices) <= max_pairs:
        vis_indices = available_indices
    else:
        # Always include first and last, and sample in between
        vis_indices = [available_indices[0]]
        if max_pairs > 2:
            step = (len(available_indices) - 1) / (max_pairs - 1)
            for i in range(1, max_pairs - 1):
                idx = int(i * step)
                if available_indices[idx] not in vis_indices:
                    vis_indices.append(available_indices[idx])
        if available_indices[-1] not in vis_indices:
            vis_indices.append(available_indices[-1])

    vis_indices = sorted(set(vis_indices))
    print(f"\nVisualizing matches for pairs: {vis_indices}")

    for k in vis_indices:
        details = pair_details[k]

        # Load images
        imgL = cv2.imread(str(images_left[k]))
        imgR = cv2.imread(str(images_right[k]))

        if imgL is None or imgR is None:
            print(f"Warning: Could not load images for pair {k}")
            continue

        vis_filename = f"matched_boards_{k}.png"

        plot_matched_boards(
            k,
            imgL,
            imgR,
            left_by_image[k],
            right_by_image[k],
            details["matches"],
            details["unmatched_L"],
            details["unmatched_R"],
            output_path=output_dir / vis_filename,
            mode=vis_mode,
        )


def plot_all_imgpoints_overlay(
    imgpoints_left: list[np.ndarray],
    imgpoints_right: list[np.ndarray],
    image_size: tuple[int, int],
    output_path: Path | None = None,
    mode: str = "save",
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Scatter all detected image points from left and right cameras in one overlay plot.

    Left points are drawn in blue, right points in red. Useful to visually spot
    mismatches or systematic offsets between the two views.

    Args:
        imgpoints_left: List of left image point arrays (Nx2 per board flattened as in OpenCV)
        imgpoints_right: List of right image point arrays
        image_size: (width, height) of images
        output_path: Path to save the figure
        mode: 'show', 'save', or 'both'
        figsize: Figure size
    """
    if not imgpoints_left and not imgpoints_right:
        print("No image points provided for overlay plot")
        return

    w, h = image_size

    xs_L = []
    ys_L = []
    xs_R = []
    ys_R = []

    # Process all left points
    for ptsL in imgpoints_left:
        if ptsL is not None:
            pL = ptsL.reshape(-1, 2)
            xs_L.extend(pL[:, 0].tolist())
            ys_L.extend(pL[:, 1].tolist())

    # Process all right points
    for ptsR in imgpoints_right:
        if ptsR is not None:
            pR = ptsR.reshape(-1, 2)
            xs_R.extend(pR[:, 0].tolist())
            ys_R.extend(pR[:, 1].tolist())

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs_L, ys_L, s=6, c="tab:blue", alpha=0.6, label="Left")
    ax.scatter(xs_R, ys_R, s=6, c="tab:red", alpha=0.6, label="Right")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("All Image Points Overlay (Left vs Right)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()

    _handle_figure_output(fig, output_path, mode)
