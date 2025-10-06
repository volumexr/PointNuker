#!/usr/bin/env python3
"""
PointNuker CLI - Headless point cloud cleaning for 3D Gaussian Splatting
"""

import argparse
import sys
import os
import json
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

# Preset management
PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pointnuker_cli_presets.json")

def load_presets():
    """Load presets from JSON file."""
    if os.path.exists(PRESET_FILE):
        try:
            with open(PRESET_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_presets(presets):
    """Save presets to JSON file."""
    try:
        with open(PRESET_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save presets: {e}")

def list_presets():
    """List all available presets."""
    presets = load_presets()
    if not presets:
        print("No presets saved.")
        return

    print("Available presets:")
    for name, params in presets.items():
        print(f"\n  {name}:")
        for key, value in params.items():
            print(f"    {key}: {value}")


def load_ply_gs_safe(path):
    """Load PLY with GS-safe index mapping."""
    ply = PlyData.read(path)
    if 'vertex' not in ply:
        raise ValueError("PLY has no 'vertex' element")

    vertex_data = ply['vertex'].data
    n = len(vertex_data)
    if n == 0:
        raise ValueError("PLY contains no points")

    xyz = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    idx = np.arange(n, dtype=np.int64)

    return pcd, idx, ply, vertex_data


def save_ply_gs_safe(vertex_data, idx_keep, ply_raw, out_path):
    """Save PLY preserving all GS attributes."""
    idx_keep = np.asarray(idx_keep, dtype=np.int64)
    idx_keep.sort()

    filtered = vertex_data[idx_keep]
    vertex_el = PlyElement.describe(filtered, 'vertex')

    new_ply = PlyData([vertex_el], text=ply_raw.text, byte_order=getattr(ply_raw, "byte_order", "<"))
    try:
        new_ply.comments = list(ply_raw.comments)
        new_ply.obj_info = list(ply_raw.obj_info)
    except:
        pass

    new_ply.write(out_path)


def dbscan_largest_cluster(pcd, idx, eps, min_points):
    """Keep only largest DBSCAN cluster."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    if labels.size == 0 or labels.max() < 0:
        print("[DBSCAN] No clusters detected (all noise)")
        return pcd, idx

    valid = labels >= 0
    if not np.any(valid):
        print("[DBSCAN] All points labeled as noise")
        return pcd, idx

    unique, counts = np.unique(labels[valid], return_counts=True)
    best_label = unique[np.argmax(counts)]
    best_count = counts.max()
    print(f"[DBSCAN] Largest cluster: label={best_label}, points={best_count}")

    idx_local = np.where(labels == best_label)[0].astype(np.int64)
    pcd = pcd.select_by_index(idx_local.tolist())
    idx = idx[idx_local]

    return pcd, idx


def radius_outlier_removal(pcd, idx, nb_points, radius):
    """Remove radius outliers with rollback on failure."""
    print(f"[RADIUS] nb_points={nb_points}, radius={radius}")
    pcd_tmp, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

    if len(ind) == 0:
        print("[RADIUS] Would remove ALL points - skipping")
        return pcd, idx

    print(f"[RADIUS] Kept {len(ind)} points")
    idx = idx[np.asarray(ind, dtype=np.int64)]
    return pcd_tmp, idx


def statistical_outlier_removal(pcd, idx, nb_neighbors, std_ratio):
    """Remove statistical outliers with rollback on failure."""
    print(f"[STAT] nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    pcd_tmp, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    if len(ind) == 0:
        print("[STAT] Would remove ALL points - skipping")
        return pcd, idx

    print(f"[STAT] Kept {len(ind)} points")
    idx = idx[np.asarray(ind, dtype=np.int64)]
    return pcd_tmp, idx


def aabb_crop(pcd, idx, min_bound, max_bound):
    """Crop to axis-aligned bounding box."""
    pts = np.asarray(pcd.points)
    mask = (
        (pts[:, 0] >= min_bound[0]) & (pts[:, 0] <= max_bound[0]) &
        (pts[:, 1] >= min_bound[1]) & (pts[:, 1] <= max_bound[1]) &
        (pts[:, 2] >= min_bound[2]) & (pts[:, 2] <= max_bound[2])
    )

    keep_local = np.where(mask)[0].astype(np.int64)
    if keep_local.size == 0:
        print("[CROP] Would remove ALL points - skipping")
        return pcd, idx

    print(f"[CROP] Kept {len(keep_local)} points")
    pcd = pcd.select_by_index(keep_local.tolist())
    idx = idx[keep_local]
    return pcd, idx


def main():
    parser = argparse.ArgumentParser(
        description="PointNuker CLI - Headless point cloud cleaning for 3DGS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input", nargs='?', help="Input PLY file")
    parser.add_argument("output", nargs='?', help="Output PLY file (GS-safe)")

    # Preset management
    parser.add_argument("--preset", type=str, help="Load parameters from saved preset")
    parser.add_argument("--save-preset", type=str, help="Save current parameters as a preset")
    parser.add_argument("--list-presets", action="store_true", help="List all saved presets")

    # DBSCAN
    parser.add_argument("--dbscan", action="store_true", help="Apply DBSCAN largest cluster")
    parser.add_argument("--eps", type=float, default=1.5, help="DBSCAN eps parameter")
    parser.add_argument("--min-points", type=int, default=50, help="DBSCAN min_points parameter")

    # Radius outlier
    parser.add_argument("--radius-outlier", action="store_true", help="Apply radius outlier removal")
    parser.add_argument("--radius-nb", type=int, default=16, help="Radius outlier nb_points")
    parser.add_argument("--radius", type=float, default=0.02, help="Radius outlier radius")

    # Statistical outlier
    parser.add_argument("--stat-outlier", action="store_true", help="Apply statistical outlier removal")
    parser.add_argument("--stat-nb", type=int, default=20, help="Statistical nb_neighbors")
    parser.add_argument("--stat-std", type=float, default=1.5, help="Statistical std_ratio")

    # AABB crop
    parser.add_argument("--crop", action="store_true", help="Apply AABB crop")
    parser.add_argument("--min-x", type=float, help="Min X bound")
    parser.add_argument("--min-y", type=float, help="Min Y bound")
    parser.add_argument("--min-z", type=float, help="Min Z bound")
    parser.add_argument("--max-x", type=float, help="Max X bound")
    parser.add_argument("--max-y", type=float, help="Max Y bound")
    parser.add_argument("--max-z", type=float, help="Max Z bound")

    args = parser.parse_args()

    # Handle list-presets
    if args.list_presets:
        list_presets()
        sys.exit(0)

    # Load preset if specified
    if args.preset:
        presets = load_presets()
        if args.preset not in presets:
            print(f"ERROR: Preset '{args.preset}' not found.")
            print("Use --list-presets to see available presets.")
            sys.exit(1)

        preset = presets[args.preset]
        print(f"[PRESET] Loading preset: {args.preset}")

        # Apply preset values (only override if not explicitly set by user)
        for key, value in preset.items():
            key_normalized = key.replace('-', '_')
            if key_normalized in args and getattr(args, key_normalized) == parser.get_default(key_normalized):
                setattr(args, key_normalized, value)
                print(f"  {key}: {value}")

    # Validate required arguments
    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)

    # Validate crop bounds
    if args.crop:
        if None in [args.min_x, args.min_y, args.min_z, args.max_x, args.max_y, args.max_z]:
            print("ERROR: --crop requires --min-x, --min-y, --min-z, --max-x, --max-y, --max-z")
            sys.exit(1)

    # Load PLY
    print(f"[LOAD] {args.input}")
    pcd, idx, ply_raw, vertex_data = load_ply_gs_safe(args.input)
    initial_points = len(idx)
    print(f"[LOAD] Loaded {initial_points:,} points")

    # Apply filters
    if args.dbscan:
        pcd, idx = dbscan_largest_cluster(pcd, idx, args.eps, args.min_points)

    if args.radius_outlier:
        pcd, idx = radius_outlier_removal(pcd, idx, args.radius_nb, args.radius)

    if args.stat_outlier:
        pcd, idx = statistical_outlier_removal(pcd, idx, args.stat_nb, args.stat_std)

    if args.crop:
        min_bound = [args.min_x, args.min_y, args.min_z]
        max_bound = [args.max_x, args.max_y, args.max_z]
        pcd, idx = aabb_crop(pcd, idx, min_bound, max_bound)

    # Save result
    final_points = len(idx)
    removed = initial_points - final_points
    pct = (removed / max(initial_points, 1)) * 100.0

    print(f"\n[SUMMARY]")
    print(f"  Initial: {initial_points:,} points")
    print(f"  Final:   {final_points:,} points")
    print(f"  Removed: {removed:,} ({pct:.2f}%)")

    save_ply_gs_safe(vertex_data, idx, ply_raw, args.output)
    print(f"[SAVE] {args.output}")
    print("[DONE] All GS attributes preserved")

    # Save preset if requested
    if args.save_preset:
        presets = load_presets()
        preset_params = {}

        # Save filter settings
        if args.dbscan:
            preset_params['dbscan'] = True
            preset_params['eps'] = args.eps
            preset_params['min_points'] = args.min_points

        if args.radius_outlier:
            preset_params['radius_outlier'] = True
            preset_params['radius_nb'] = args.radius_nb
            preset_params['radius'] = args.radius

        if args.stat_outlier:
            preset_params['stat_outlier'] = True
            preset_params['stat_nb'] = args.stat_nb
            preset_params['stat_std'] = args.stat_std

        if args.crop:
            preset_params['crop'] = True
            preset_params['min_x'] = args.min_x
            preset_params['min_y'] = args.min_y
            preset_params['min_z'] = args.min_z
            preset_params['max_x'] = args.max_x
            preset_params['max_y'] = args.max_y
            preset_params['max_z'] = args.max_z

        presets[args.save_preset] = preset_params
        save_presets(presets)
        print(f"\n[PRESET] Saved as '{args.save_preset}'")


if __name__ == "__main__":
    main()
