# PointNuker.py
# ===============================================================
# PointNuker — Point Cloud Cleaner for 3DGS (GS-safe)
# Version: 1.0
# Created: 25/09/25
# License: MIT (Open-source for the Radiance Fields community)
# ===============================================================

import os
import sys
import json
import time
import threading
import contextlib
import copy as _pycopy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.scrolledtext as scrolledtext

import numpy as np

# ---------- Optional modern theme ----------
USING_TTKBOOTSTRAP = False
try:
    import ttkbootstrap as tb  # pip install ttkbootstrap
    USING_TTKBOOTSTRAP = True
except Exception:
    USING_TTKBOOTSTRAP = False

# ---------- App metadata ----------
APP_NAME = "PointNuker"
APP_VERSION = "1.0"
CREATED_DATE = "25/09/25"

ABOUT_LIBS = (
    "Open3D, plyfile, numpy, tkinter"
    + (", ttkbootstrap" if USING_TTKBOOTSTRAP else "")
)

ABOUT_TEXT = (
    f"{APP_NAME} v{APP_VERSION}\n"
    f"Created: {CREATED_DATE}\n\n"
    "Open-source software destined to the Radiance Fields community.\n\n"
    "Libraries:\n"
    f" - {ABOUT_LIBS}\n\n"
    "License: MIT"
)

CHANGELOG_TEXT = """\
PointNuker — Changelog
----------------------
v1.0 (25/09/25)
- Initial public release.
- GS-safe saving that preserves all original PLY vertex properties
  (e.g., spherical harmonics, opacity, scale, rotation, etc.).
- Mode 1 (Pipeline): radius/statistical outlier removal with safe rollback,
  optional AABB crop, preview tools, progress bar & logs.
- Mode 2 (Cluster Finder): DBSCAN largest-cluster selection; one-click export
  of the current selection; persistent parameters.
- Real-time previewer windows (Open3D) for ORIGINAL/CURRENT/CLEANED.
- View-only orientation helpers (Flip X 180°, Swap Y/Z) to fix "up" direction.
- Modernized UI (ttkbootstrap if installed; fallback to clam theme).
"""

# ---------- Open3D ----------
try:
    import open3d as o3d
except Exception as e:
    raise SystemExit(
        f"[{APP_NAME}] Could not import open3d.\n"
        "Activate your env and install dependencies:\n"
        "  conda activate pointnuker\n"
        "  python -m pip install open3d plyfile ttkbootstrap\n"
        f"Detail: {e}"
    )

# ---------- plyfile (for GS-safe) ----------
try:
    from plyfile import PlyData, PlyElement
except Exception as e:
    raise SystemExit(
        f"[{APP_NAME}] Missing 'plyfile'. Install with:\n"
        "  python -m pip install plyfile\n"
        f"Detail: {e}"
    )

# ---------- Utils ----------
def draw_quiet(geoms, **kwargs):
    """Open Open3D viewer while silencing stdout/stderr."""
    with open(os.devnull, "w") as devnull, \
         contextlib.redirect_stderr(devnull), \
         contextlib.redirect_stdout(devnull):
        o3d.visualization.draw_geometries(geoms, **kwargs)

def clone_pcd(p):
    return p.clone() if hasattr(p, "clone") else _pycopy.deepcopy(p)

def pcd_n_points(pcd):
    return len(pcd.points) if pcd is not None else 0

def pcd_is_empty(pcd):
    return pcd_n_points(pcd) == 0

# ---------- Preferences ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_PATH = os.path.join(APP_DIR, "pointnuker_profile.json")

DEFAULT_PROFILE = {
    "dbscan_eps": 0.03,
    "dbscan_min_points": 30,
    "auto_apply_cluster": False,
    "orientation_preset": "none",   # "none" | "flip_x_180" | "swap_yz"
    "auto_apply_orientation": False,
    "gs_mode": True,                # GS mode ON by default
}

def load_profile():
    try:
        if os.path.isfile(PROFILE_PATH):
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {**DEFAULT_PROFILE, **data}
    except Exception:
        pass
    return DEFAULT_PROFILE.copy()

def save_profile(p):
    try:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ---------- Main GUI ----------
class PointNukerApp(tk.Tk):
    def __init__(self, initial_path=None):
        # Use ttkbootstrap window if available
        if USING_TTKBOOTSTRAP:
            super().__init__(className=APP_NAME)
            self.style = tb.Style(theme="cosmo")  # modern look
        else:
            super().__init__(className=APP_NAME)
            self.style = ttk.Style()
            try:
                self.style.theme_use("clam")
            except Exception:
                pass

        self.title(f"{APP_NAME} — v{APP_VERSION}")
        self.geometry("1220x840")
        self.minsize(1040, 760)

        # --- Global font / spacing tweaks (FIX for 'Segoe UI 10' tokenization) ---
        try:
            # Wrap family with braces so Tk treats it as one token
            self.option_add("*Font", "{Segoe UI} 10")
        except Exception:
            # Fallback: ignore if system lacks that family
            pass
        self.option_add("*TButton.Padding", 8)
        self.option_add("*TEntry.Padding", 4)

        # State
        self.src_path = None
        self.pcd_original = None
        self.pcd_current = None
        self.pcd_cleaned = None

        # Index mapping to original PLY rows
        self.idx_current = None
        self.idx_cleaned = None
        self.mapping_valid_cleaned = True

        # Original PLY store
        self.ply_raw = None
        self.vertex_data0 = None
        self.ply_text = True
        self.ply_byte_order = "<"

        # Profile
        self.profile = load_profile()

        # Build UI
        self._build_menu()
        self._build_main()

        # Status bar (footer)
        self._build_statusbar()

        # Initial file (optional)
        if initial_path and os.path.isfile(initial_path):
            self._load_path(initial_path)

    # ---------- Menus ----------
    def _build_menu(self):
        menubar = tk.Menu(self)

        # File
        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="Open .PLY...", command=self.open_file)
        m_file.add_separator()
        m_file.add_command(label="Save CURRENT (Open3D)...", command=self.save_current_as_o3d)
        m_file.add_command(label="Save CLEANED (Open3D)...", command=self.save_cleaned_as_o3d)
        m_file.add_separator()
        m_file.add_command(label="Save CURRENT (GS-safe)...", command=self.save_current_as_gs)
        m_file.add_command(label="Save CLEANED (GS-safe)...", command=self.save_cleaned_as_gs)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.on_exit)
        menubar.add_cascade(label="File", menu=m_file)

        # View
        m_view = tk.Menu(menubar, tearoff=False)
        m_view.add_command(label="Preview ORIGINAL", command=self.preview_original)
        m_view.add_command(label="Preview CURRENT", command=self.preview_current)
        m_view.add_command(label="Preview CLEANED", command=self.preview_cleaned)
        menubar.add_cascade(label="View", menu=m_view)

        # Help
        m_help = tk.Menu(menubar, tearoff=False)
        m_help.add_command(label="About", command=self.show_about)
        m_help.add_command(label="Changelog", command=self.show_changelog)
        menubar.add_cascade(label="Help", menu=m_help)

        self.config(menu=menubar)

    def show_about(self):
        messagebox.showinfo(
            f"About — {APP_NAME}",
            ABOUT_TEXT
        )

    def show_changelog(self):
        messagebox.showinfo("Changelog", CHANGELOG_TEXT)

    # ---------- Layout ----------
    def _build_main(self):
        # Top toolbar
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        self.lbl_file = ttk.Label(
            top,
            text="File: (none)",
            font=("Segoe UI", 10, "bold")
        )
        self.lbl_file.pack(side="left", padx=(0, 10))

        ttk.Button(top, text="Open .PLY...", command=self.open_file).pack(side="left")
        ttk.Button(top, text="Reset to ORIGINAL", command=self.reset_to_original).pack(side="left", padx=6)
        ttk.Button(top, text="Preview ORIGINAL", command=self.preview_original).pack(side="left", padx=6)
        ttk.Button(top, text="Preview CURRENT", command=self.preview_current).pack(side="left", padx=6)
        ttk.Button(top, text="Save CURRENT (GS-safe)", command=self.save_current_as_gs).pack(side="left", padx=8)
        ttk.Button(top, text="Save CLEANED (GS-safe)", command=self.save_cleaned_as_gs).pack(side="left", padx=8)

        # Main 2-column content
        main = ttk.Frame(self, padding=(10, 0))
        main.pack(fill="both", expand=True)

        # MODE 1: Pipeline
        mode1 = ttk.LabelFrame(main, text="Mode 1 — Pipeline", padding=10)
        mode1.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=10)

        # MODE 2: Cluster Finder
        mode2 = ttk.LabelFrame(main, text="Mode 2 — Cluster Finder", padding=10)
        mode2.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=10)

        # ----- Mode 1 controls -----
        # GS mode
        box_gs = ttk.LabelFrame(mode1, text="3DGS Mode (preserve attributes)", padding=8)
        box_gs.pack(fill="x", pady=6)
        self.var_gs_mode = tk.BooleanVar(value=self.profile.get("gs_mode", True))
        gs_chk = ttk.Checkbutton(
            box_gs,
            text="ON (only delete points; GS-safe saving keeps all vertex properties)",
            variable=self.var_gs_mode,
            command=self._on_toggle_gs_mode
        )
        gs_chk.pack(side="left", padx=6, pady=2)

        # Downsample
        box_ds = ttk.LabelFrame(mode1, text="Voxel Downsample (optional)", padding=8)
        box_ds.pack(fill="x", pady=6)
        ttk.Label(box_ds, text="voxel size [m] (0 = off):").pack(side="left", padx=6)
        self.var_voxel = tk.StringVar(value="0.0")
        self.ent_voxel = ttk.Entry(box_ds, textvariable=self.var_voxel, width=10)
        self.ent_voxel.pack(side="left")
        if self.var_gs_mode.get():
            self.ent_voxel.config(state="disabled")
            self.var_voxel.set("0.0")

        # Radius outlier
        box_rad = ttk.LabelFrame(mode1, text="Radius Outlier Removal", padding=8)
        box_rad.pack(fill="x", pady=6)
        self.var_use_rad = tk.BooleanVar(value=True)
        ttk.Checkbutton(box_rad, text="Enable", variable=self.var_use_rad).pack(side="left", padx=(6, 10))
        ttk.Label(box_rad, text="nb_points:").pack(side="left")
        self.var_rad_nb = tk.StringVar(value="16")
        ttk.Entry(box_rad, textvariable=self.var_rad_nb, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(box_rad, text="radius [m]:").pack(side="left")
        self.var_rad_r = tk.StringVar(value="0.02")
        ttk.Entry(box_rad, textvariable=self.var_rad_r, width=8).pack(side="left", padx=(2, 10))

        # Statistical outlier
        box_stat = ttk.LabelFrame(mode1, text="Statistical Outlier Removal", padding=8)
        box_stat.pack(fill="x", pady=6)
        self.var_use_stat = tk.BooleanVar(value=True)
        ttk.Checkbutton(box_stat, text="Enable", variable=self.var_use_stat).pack(side="left", padx=(6, 10))
        ttk.Label(box_stat, text="nb_neighbors:").pack(side="left")
        self.var_stat_nb = tk.StringVar(value="20")
        ttk.Entry(box_stat, textvariable=self.var_stat_nb, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(box_stat, text="std_ratio:").pack(side="left")
        self.var_stat_std = tk.StringVar(value="1.5")
        ttk.Entry(box_stat, textvariable=self.var_stat_std, width=6).pack(side="left", padx=(2, 10))

        # Crop AABB
        box_crop = ttk.LabelFrame(mode1, text="AABB Crop (axis-aligned box)", padding=8)
        box_crop.pack(fill="x", pady=6)
        grid = ttk.Frame(box_crop)
        grid.pack(fill="x", padx=4, pady=4)
        self.var_crop_vals = {
            "min_x": tk.StringVar(value=""),
            "min_y": tk.StringVar(value=""),
            "min_z": tk.StringVar(value=""),
            "max_x": tk.StringVar(value=""),
            "max_y": tk.StringVar(value=""),
            "max_z": tk.StringVar(value=""),
        }
        # Row 1
        ttk.Label(grid, text="min_x").grid(row=0, column=0, sticky="w", padx=3)
        ttk.Entry(grid, textvariable=self.var_crop_vals["min_x"], width=10).grid(row=0, column=1, padx=3)
        ttk.Label(grid, text="min_y").grid(row=0, column=2, sticky="w", padx=3)
        ttk.Entry(grid, textvariable=self.var_crop_vals["min_y"], width=10).grid(row=0, column=3, padx=3)
        ttk.Label(grid, text="min_z").grid(row=0, column=4, sticky="w", padx=3)
        ttk.Entry(grid, textvariable=self.var_crop_vals["min_z"], width=10).grid(row=0, column=5, padx=3)
        # Row 2
        ttk.Label(grid, text="max_x").grid(row=1, column=0, sticky="w", padx=3, pady=(4, 0))
        ttk.Entry(grid, textvariable=self.var_crop_vals["max_x"], width=10).grid(row=1, column=1, padx=3, pady=(4, 0))
        ttk.Label(grid, text="max_y").grid(row=1, column=2, sticky="w", padx=3, pady=(4, 0))
        ttk.Entry(grid, textvariable=self.var_crop_vals["max_y"], width=10).grid(row=1, column=3, padx=3, pady=(4, 0))
        ttk.Label(grid, text="max_z").grid(row=1, column=4, sticky="w", padx=3, pady=(4, 0))
        ttk.Entry(grid, textvariable=self.var_crop_vals["max_z"], width=10).grid(row=1, column=5, padx=3, pady=(4, 0))

        btns = ttk.Frame(box_crop)
        btns.pack(fill="x", padx=4, pady=(2, 2))
        ttk.Button(btns, text="Autofill bounds from CURRENT cloud", command=self.autofill_bounds).pack(side="left")
        ttk.Label(btns, text="(Leave fields empty to skip crop)").pack(side="left", padx=10)

        # Actions + progress (Mode1)
        actions = ttk.Frame(mode1, padding=(0, 8))
        actions.pack(fill="x")
        self.btn_run = ttk.Button(actions, text="RUN CLEAN (pipeline)", command=self.run_clean)
        self.btn_run.pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="determinate", length=280)
        self.progress.pack(side="left", padx=10)
        ttk.Label(actions, text="").pack(side="left", expand=True)

        # ----- Mode 2 controls -----
        # Orientation helpers (view only)
        box_or = ttk.LabelFrame(mode2, text="Orientation (view only; GS-safe doesn't modify xyz)", padding=8)
        box_or.pack(fill="x", pady=6)
        ttk.Button(box_or, text="Flip vertical (Rx=180°)", command=self.orient_flip_x).pack(side="left", padx=6, pady=2)
        ttk.Button(box_or, text="Swap Y/Z", command=self.orient_swap_yz).pack(side="left", padx=6, pady=2)
        self.var_save_orient = tk.BooleanVar(value=self.profile.get("auto_apply_orientation", False))
        ttk.Checkbutton(
            box_or,
            text="Save as default orientation and auto-apply on load",
            variable=self.var_save_orient,
            command=self._toggle_auto_orient
        ).pack(side="left", padx=6)

        # DBSCAN
        box_cluster = ttk.LabelFrame(mode2, text="DBSCAN Cluster Finder", padding=8)
        box_cluster.pack(fill="x", pady=6)
        ttk.Label(box_cluster, text="eps [m]:").pack(side="left", padx=(6, 2))
        self.var_db_eps = tk.StringVar(value=str(self.profile.get("dbscan_eps", 0.03)))
        ttk.Entry(box_cluster, textvariable=self.var_db_eps, width=8).pack(side="left", padx=(0, 10))
        ttk.Label(box_cluster, text="min_points:").pack(side="left", padx=(0, 2))
        self.var_db_minpts = tk.StringVar(value=str(self.profile.get("dbscan_min_points", 30)))
        ttk.Entry(box_cluster, textvariable=self.var_db_minpts, width=8).pack(side="left", padx=(0, 10))
        ttk.Button(box_cluster, text="Detect largest cluster NOW", command=self.detect_largest_cluster_now).pack(side="left", padx=6)

        self.var_keep_largest_on_clean = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box_cluster,
            text="Keep only largest cluster during CLEAN",
            variable=self.var_keep_largest_on_clean
        ).pack(side="left", padx=10)

        self.var_auto_apply_cluster = tk.BooleanVar(value=self.profile.get("auto_apply_cluster", False))
        ttk.Checkbutton(
            box_cluster,
            text="Remember & auto-apply on load",
            variable=self.var_auto_apply_cluster,
            command=self._toggle_auto_cluster
        ).pack(side="left", padx=6)

        # Param assistant
        box_auto = ttk.LabelFrame(mode2, text="Parameter Assistant", padding=8)
        box_auto.pack(fill="x", pady=6)
        ttk.Label(
            box_auto,
            text="Analyze kNN to suggest radius/eps/min_points (uses a random sample)."
        ).pack(side="left", padx=8)
        ttk.Button(box_auto, text="Auto-suggest params (kNN)", command=self.autosuggest_params).pack(side="left", padx=8)

        # Info card (community)
        info = ttk.LabelFrame(mode2, text="Open-source Notice", padding=8)
        info.pack(fill="x", pady=6)
        ttk.Label(
            info,
            text="This is open-source software intended for the Radiance Fields community."
        ).pack(side="left", padx=8, pady=2)

        # Logs area (bottom)
        logf = ttk.LabelFrame(self, text="Process console / Log")
        logf.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_log = scrolledtext.ScrolledText(logf, font=("Consolas", 10), height=16)
        self.txt_log.pack(fill="both", expand=True)

    def _build_statusbar(self):
        bar = ttk.Frame(self, padding=(10, 6))
        bar.pack(fill="x", side="bottom")
        self.lbl_status = ttk.Label(bar, text="Ready.")
        self.lbl_status.pack(side="left")
        right = ttk.Label(bar, text=f"v{APP_VERSION}  |  Open-source for the Radiance Fields community")
        right.pack(side="right")

    # ---------- UI helpers ----------
    def log(self, msg):
        self.txt_log.insert("end", f"{msg}\n")
        self.txt_log.see("end")
        self.update_idletasks()

    def set_status(self, text):
        self.lbl_status.config(text=text)
        self.update_idletasks()

    def set_progress(self, value):
        self.progress["value"] = value
        self.update_idletasks()

    def on_exit(self):
        self.destroy()

    # ---------- File I/O ----------
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open .PLY",
            filetypes=[("PLY", "*.ply"), ("All files", "*.*")]
        )
        if path:
            self._load_path(path)

    def _apply_saved_orientation_if_any(self):
        preset = self.profile.get("orientation_preset", "none")
        auto = self.profile.get("auto_apply_orientation", False)
        if not auto or preset == "none" or self.pcd_current is None:
            return
        self.log(f"[ORIENT] Applying default orientation: {preset}")
        if preset == "flip_x_180":
            self._apply_rotation_xyz(np.pi, 0.0, 0.0)
        elif preset == "swap_yz":
            self._apply_swap_yz()

    def _load_path(self, path):
        try:
            self.log(f"[INFO] Loading (plyfile): {path}")
            ply = PlyData.read(path)
            if 'vertex' not in ply:
                messagebox.showerror("Error", "PLY has no 'vertex' element.")
                return

            self.ply_raw = ply
            self.vertex_data0 = ply['vertex'].data
            n0 = len(self.vertex_data0)
            if n0 == 0:
                messagebox.showerror("Error", "The file contains no points.")
                return

            self.ply_text = ply.text
            self.ply_byte_order = getattr(ply, "byte_order", "<")

            xyz = np.stack([self.vertex_data0['x'], self.vertex_data0['y'], self.vertex_data0['z']], axis=1)
            self.pcd_original = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            self.pcd_current = clone_pcd(self.pcd_original)
            self.pcd_cleaned = None

            self.idx_current = np.arange(n0, dtype=np.int64)
            self.idx_cleaned = None
            self.mapping_valid_cleaned = True

            self.src_path = path
            self.lbl_file.config(text=f"File: {os.path.basename(path)}")
            self._report_cloud("ORIGINAL", self.pcd_original)

            self._apply_saved_orientation_if_any()

            if self.profile.get("auto_apply_cluster", False):
                eps = float(self.profile.get("dbscan_eps", 0.03))
                minpts = int(self.profile.get("dbscan_min_points", 30))
                self.log(f"[CLUSTER] Auto DBSCAN eps={eps}, min_points={minpts}")
                changed = self._keep_largest_cluster_in_current(eps, minpts)
                if changed:
                    self._report_cloud("After auto-cluster", self.pcd_current)

            self.set_status("File loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't load file.\n{e}")

    def _default_output_path(self, suffix):
        if not self.src_path:
            return os.path.join(os.getcwd(), f"{suffix}.ply")
        base, ext = os.path.splitext(self.src_path)
        return f"{base}{suffix}{ext}"

    # ---------- Save (Open3D preview format) ----------
    def save_current_as_o3d(self):
        if self.pcd_current is None:
            messagebox.showinfo("Info", "No CURRENT cloud to save.")
            return
        default = self._default_output_path("_current_o3d")
        path = filedialog.asksaveasfilename(
            title="Save CURRENT (Open3D)",
            initialfile=os.path.basename(default),
            defaultextension=".ply",
            filetypes=[("PLY", "*.ply")]
        )
        if path:
            try:
                o3d.io.write_point_cloud(path, self.pcd_current)
                self.log(f"[OK] Saved CURRENT (Open3D): {path}")
                messagebox.showinfo("Saved", f"Saved CURRENT (Open3D):\n{path}\n"
                                             "(Note: does NOT preserve extra GS attributes)")
            except Exception as e:
                messagebox.showerror("Error", f"Couldn't save.\n{e}")

    def save_cleaned_as_o3d(self):
        if self.pcd_cleaned is None:
            messagebox.showinfo("Info", "No CLEANED cloud yet. Run CLEAN first.")
            return
        default = self._default_output_path("_cleaned_o3d")
        path = filedialog.asksaveasfilename(
            title="Save CLEANED (Open3D)",
            initialfile=os.path.basename(default),
            defaultextension=".ply",
            filetypes=[("PLY", "*.ply")]
        )
        if path:
            try:
                o3d.io.write_point_cloud(path, self.pcd_cleaned)
                self.log(f"[OK] Saved CLEANED (Open3D): {path}")
                messagebox.showinfo("Saved", f"Saved CLEANED (Open3D):\n{path}\n"
                                             "(Note: does NOT preserve extra GS attributes)")
            except Exception as e:
                messagebox.showerror("Error", f"Couldn't save.\n{e}")

    # ---------- Save (GS-safe) ----------
    def _write_gs_preserving(self, idx_keep_sorted, out_path):
        if self.vertex_data0 is None or self.ply_raw is None:
            raise RuntimeError("No original PLY loaded.")
        if len(idx_keep_sorted) == 0:
            raise RuntimeError("Selection is empty; nothing to save.")

        idx_keep_sorted = np.asarray(idx_keep_sorted, dtype=np.int64)
        idx_keep_sorted.sort()

        filtered = self.vertex_data0[idx_keep_sorted]
        vertex_el = PlyElement.describe(filtered, 'vertex')

        # Many 3DGS PLYs only have 'vertex'. If other elements exist, they are omitted
        # to avoid inconsistent indices.
        elements = [vertex_el]

        new_ply = PlyData(elements, text=self.ply_text, byte_order=self.ply_byte_order)
        try:
            new_ply.comments = list(self.ply_raw.comments)
            new_ply.obj_info = list(self.ply_raw.obj_info)
        except Exception:
            pass

        new_ply.write(out_path)

    def save_current_as_gs(self):
        if self.idx_current is None or self.vertex_data0 is None:
            messagebox.showinfo("Info", "No CURRENT selection.")
            return

        if self.var_gs_mode.get() and float(self.var_voxel.get() or "0") > 0:
            messagebox.showwarning(
                "GS mode",
                "3DGS mode is ON: voxel downsample is disabled to preserve 1:1 mapping."
            )
            self.var_voxel.set("0.0")

        default = self._default_output_path("_current_gs")
        path = filedialog.asksaveasfilename(
            title="Save CURRENT (GS-safe; preserve attributes)",
            initialfile=os.path.basename(default),
            defaultextension=".ply",
            filetypes=[("PLY", "*.ply")]
        )
        if not path:
            return
        try:
            self._write_gs_preserving(self.idx_current, path)
            self.log(f"[OK] Saved CURRENT (GS-safe): {path}")
            messagebox.showinfo("Saved", "CURRENT (GS-safe) saved.\nAll original vertex properties preserved.")
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't save (GS-safe).\n{e}")

    def save_cleaned_as_gs(self):
        if self.idx_cleaned is None:
            messagebox.showinfo("Info", "No CLEANED GS-safe result yet. Run CLEAN.")
            return
        if not self.mapping_valid_cleaned:
            messagebox.showwarning(
                "GS-safe unavailable",
                "Pipeline included steps that break 1:1 mapping (e.g., voxel downsample).\n"
                "Re-run CLEAN with 3DGS Mode ON and without voxel downsample."
            )
            return

        default = self._default_output_path("_cleaned_gs")
        path = filedialog.asksaveasfilename(
            title="Save CLEANED (GS-safe; preserve attributes)",
            initialfile=os.path.basename(default),
            defaultextension=".ply",
            filetypes=[("PLY", "*.ply")]
        )
        if not path:
            return
        try:
            self._write_gs_preserving(self.idx_cleaned, path)
            self.log(f"[OK] Saved CLEANED (GS-safe): {path}")
            messagebox.showinfo("Saved", "CLEANED (GS-safe) saved.\nAll original vertex properties preserved.")
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't save (GS-safe).\n{e}")

    # ---------- Bounds ----------
    def _get_bounds_from_entries(self):
        vals = {}
        for k, var in self.var_crop_vals.items():
            s = var.get().strip()
            if s == "":
                return None
            try:
                vals[k] = float(s)
            except ValueError:
                raise ValueError(f"Invalid value in {k}: '{s}'")
        if not (vals["min_x"] < vals["max_x"] and vals["min_y"] < vals["max_y"] and vals["min_z"] < vals["max_z"]):
            raise ValueError("Bounds must satisfy min < max for each axis.")
        return vals

    def autofill_bounds(self):
        if self.pcd_current is None:
            messagebox.showinfo("Info", "Load a .PLY first.")
            return
        n = pcd_n_points(self.pcd_current)
        if n == 0:
            self.log("[INFO] CURRENT cloud empty; cannot compute bounds.")
            return
        pts = np.asarray(self.pcd_current.points)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        self.var_crop_vals["min_x"].set(f"{mn[0]:.6f}")
        self.var_crop_vals["min_y"].set(f"{mn[1]:.6f}")
        self.var_crop_vals["min_z"].set(f"{mn[2]:.6f}")
        self.var_crop_vals["max_x"].set(f"{mx[0]:.6f}")
        self.var_crop_vals["max_y"].set(f"{mx[1]:.6f}")
        self.var_crop_vals["max_z"].set(f"{mx[2]:.6f}")
        self.log("[INFO] Bounds autofilled from CURRENT cloud.")

    # ---------- Orientation (view only) ----------
    def _apply_rotation_xyz(self, rx, ry, rz):
        if self.pcd_current is None:
            return
        R = self.pcd_current.get_rotation_matrix_from_xyz((rx, ry, rz))
        self.pcd_current.rotate(R, center=(0, 0, 0))
        self.log(f"[ORIENT] Rotation applied (rx,ry,rz) = ({rx:.3f},{ry:.3f},{rz:.3f})")
        self._report_cloud("After orientation", self.pcd_current)

    def _apply_swap_yz(self):
        if self.pcd_current is None:
            return
        pts = np.asarray(self.pcd_current.points)
        if pts.size == 0:
            return
        pts[:, [1, 2]] = pts[:, [2, 1]]
        self.pcd_current.points = o3d.utility.Vector3dVector(pts)
        self.log("[ORIENT] Swap Y/Z applied")
        self._report_cloud("After swap Y/Z", self.pcd_current)

    def orient_flip_x(self):
        self._apply_rotation_xyz(np.pi, 0.0, 0.0)
        if self.var_save_orient.get():
            self.profile["orientation_preset"] = "flip_x_180"
            self.profile["auto_apply_orientation"] = True
            save_profile(self.profile)
            self.log("[ORIENT] Saved as default orientation: flip_x_180")

    def orient_swap_yz(self):
        self._apply_swap_yz()
        if self.var_save_orient.get():
            self.profile["orientation_preset"] = "swap_yz"
            self.profile["auto_apply_orientation"] = True
            save_profile(self.profile)
            self.log("[ORIENT] Saved as default orientation: swap_yz")

    def _toggle_auto_orient(self):
        self.profile["auto_apply_orientation"] = bool(self.var_save_orient.get())
        if not self.profile["auto_apply_orientation"]:
            self.profile["orientation_preset"] = "none"
        save_profile(self.profile)

    # ---------- Cluster (DBSCAN) ----------
    def _keep_largest_cluster_in_current(self, eps, min_points):
        if self.pcd_current is None:
            return False
        n = pcd_n_points(self.pcd_current)
        if n == 0:
            self.log("[DBSCAN] CURRENT cloud is empty.")
            return False

        self.log(f"[DBSCAN] eps={eps}  min_points={min_points}")
        labels = np.array(self.pcd_current.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        if labels.size == 0 or labels.max() < 0:
            self.log("[DBSCAN] No clusters detected (all noise).")
            return False

        valid = labels >= 0
        if not np.any(valid):
            self.log("[DBSCAN] All points labeled as noise.")
            return False

        unique, counts = np.unique(labels[valid], return_counts=True)
        best_label = unique[np.argmax(counts)]
        best_count = counts.max()
        self.log(f"[DBSCAN] Largest cluster label {best_label} with {best_count} points.")

        idx_local = np.where(labels == best_label)[0].astype(np.int64)
        self.pcd_current = self.pcd_current.select_by_index(idx_local.tolist())
        self.idx_current = self.idx_current[idx_local]
        return True

    def detect_largest_cluster_now(self):
        if self.pcd_current is None:
            messagebox.showinfo("Info", "Load a .PLY first.")
            return
        try:
            eps = float(self.var_db_eps.get())
            minpts = int(self.var_db_minpts.get())
        except ValueError:
            messagebox.showerror("Invalid parameters", "eps must be float and min_points must be integer.")
            return

        t0 = time.time()
        changed = self._keep_largest_cluster_in_current(eps, minpts)
        if changed:
            self._report_cloud("After largest cluster (NOW)", self.pcd_current)
        else:
            self.log("[DBSCAN] No changes.")
        self.profile["dbscan_eps"] = eps
        self.profile["dbscan_min_points"] = minpts
        if self.var_auto_apply_cluster.get():
            self.profile["auto_apply_cluster"] = True
        save_profile(self.profile)
        messagebox.showinfo("DBSCAN", f"Done in {time.time() - t0:.2f}s.\n"
                                      f"Params saved: eps={eps}, min_points={minpts}.\n"
                                      f"You can 'Save CURRENT (GS-safe)' to export this cluster.")

    def _toggle_auto_cluster(self):
        self.profile["auto_apply_cluster"] = bool(self.var_auto_apply_cluster.get())
        try:
            self.profile["dbscan_eps"] = float(self.var_db_eps.get())
            self.profile["dbscan_min_points"] = int(self.var_db_minpts.get())
        except ValueError:
            pass
        save_profile(self.profile)

    # ---------- Param assistant ----------
    def autosuggest_params(self, sample_size=20000, k=8):
        if self.pcd_current is None or pcd_is_empty(self.pcd_current):
            messagebox.showinfo("Info", "Load a cloud first.")
            return
        pts = np.asarray(self.pcd_current.points)
        n = len(pts)
        idxs = np.random.choice(n, size=min(sample_size, n), replace=False)

        kdtree = o3d.geometry.KDTreeFlann(self.pcd_current)
        nn_dists = []
        for i in idxs:
            k_found, idx_knn, d2 = kdtree.search_knn_vector_3d(self.pcd_current.points[i], max(k, 2))
            if k_found >= 2:
                nn_dists.append(np.sqrt(d2[1]))

        if not nn_dists:
            messagebox.showwarning("Auto-suggest", "Could not estimate kNN. Try another cloud.")
            return

        d_med = float(np.median(nn_dists))
        radius_sug = max(d_med * 2.5, 1e-6)
        eps_sug = max(d_med * 2.0, 1e-6)
        minpts_sug = max(8, int(k))

        self.var_rad_r.set(f"{radius_sug:.6f}")
        self.var_db_eps.set(f"{eps_sug:.6f}")
        self.var_db_minpts.set(str(minpts_sug))

        self.log(f"[AUTO] d_med≈{d_med:.6f} → radius≈{radius_sug:.6f}  dbscan eps≈{eps_sug:.6f}  min_points={minpts_sug}")
        messagebox.showinfo("Suggestions",
                            f"radius ≈ {radius_sug:.6f}\n"
                            f"DBSCAN eps ≈ {eps_sug:.6f}\n"
                            f"min_points = {minpts_sug}")

    # ---------- GS mode toggle ----------
    def _on_toggle_gs_mode(self):
        gs = bool(self.var_gs_mode.get())
        if gs:
            self.ent_voxel.config(state="disabled")
            self.var_voxel.set("0.0")
        else:
            self.ent_voxel.config(state="normal")
        self.profile["gs_mode"] = gs
        save_profile(self.profile)

    # ---------- CLEAN pipeline ----------
    def run_clean(self):
        if self.pcd_current is None:
            messagebox.showinfo("Info", "Load a .PLY first.")
            return
        self.btn_run.config(state="disabled")
        t = threading.Thread(target=self._clean_proc, daemon=True)
        t.start()

    def _clean_proc(self):
        try:
            t0 = time.time()
            self.set_status("Processing...")
            self.set_progress(0)
            self.log("=== CLEAN START ===")

            pcd = clone_pcd(self.pcd_current)
            idx_work = self.idx_current.copy()

            initial_points = pcd_n_points(pcd)
            gs_mode = bool(self.var_gs_mode.get())

            steps_total = 0
            voxel_size = float(self.var_voxel.get().strip() or "0")
            if voxel_size > 0: steps_total += 1
            if self.var_use_rad.get(): steps_total += 1
            if self.var_use_stat.get(): steps_total += 1
            do_crop = False
            try:
                bounds = self._get_bounds_from_entries()
                do_crop = bounds is not None
            except ValueError as e:
                self.log(f"[WARN] Invalid bounds; skipping crop: {e}")
                bounds = None
            if do_crop: steps_total += 1
            if self.var_keep_largest_on_clean.get(): steps_total += 1
            steps_total = max(steps_total, 1)
            steps_done = 0

            self.mapping_valid_cleaned = True

            # GS mode disables voxel step to preserve 1:1 mapping
            if gs_mode and voxel_size > 0:
                self.log("[GS] 3DGS mode ON → disabling voxel downsample.")
                voxel_size = 0.0

            # 1) voxel
            if voxel_size > 0:
                self.log(f"[STEP] Voxel downsample: size={voxel_size}")
                prev = clone_pcd(pcd)
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                if pcd_is_empty(pcd):
                    self.log("[WARN] Voxel left 0 points. Reverting.")
                    pcd = prev
                else:
                    self.mapping_valid_cleaned = False
                    steps_done += 1
                    self.set_progress(steps_done * 100 / steps_total)
                    self._report_cloud("After voxel", pcd)

            # 2) radius
            if self.var_use_rad.get():
                nb = int(self.var_rad_nb.get())
                rad = float(self.var_rad_r.get())
                self.log(f"[STEP] Radius outlier: nb_points>={nb}, radius={rad}")
                prev = clone_pcd(pcd); prev_idx = idx_work.copy()
                pcd_tmp, ind = pcd.remove_radius_outlier(nb_points=nb, radius=rad)
                if len(ind) == 0:
                    self.log("[WARN] Radius removed ALL points. Reverting. "
                             "Tip: increase 'radius' or reduce 'nb_points'.")
                else:
                    pcd = pcd_tmp
                    idx_work = prev_idx[np.asarray(ind, dtype=np.int64)]
                    self.log(f"[INFO] Points kept after radius: {len(ind)}")
                    steps_done += 1
                    self.set_progress(steps_done * 100 / steps_total)
                    self._report_cloud("After radius", pcd)

            # 3) statistical
            if self.var_use_stat.get():
                nb = int(self.var_stat_nb.get())
                std = float(self.var_stat_std.get())
                self.log(f"[STEP] Statistical outlier: nb_neighbors={nb}, std_ratio={std}")
                prev = clone_pcd(pcd); prev_idx = idx_work.copy()
                pcd_tmp, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
                if len(ind) == 0:
                    self.log("[WARN] Statistical removed ALL points. Reverting. "
                             "Tip: lower 'nb_neighbors' or raise 'std_ratio'.")
                else:
                    pcd = pcd_tmp
                    idx_work = prev_idx[np.asarray(ind, dtype=np.int64)]
                    self.log(f"[INFO] Points kept after statistical: {len(ind)}")
                    steps_done += 1
                    self.set_progress(steps_done * 100 / steps_total)
                    self._report_cloud("After statistical", pcd)

            # 4) DBSCAN largest
            if self.var_keep_largest_on_clean.get():
                eps = float(self.var_db_eps.get())
                minpts = int(self.var_db_minpts.get())
                self.log(f"[STEP] DBSCAN largest: eps={eps}, min_points={minpts}")
                labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=minpts, print_progress=False))
                if labels.size > 0 and labels.max() >= 0:
                    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
                    lab = unique[np.argmax(counts)]
                    keep_local = np.where(labels == lab)[0].astype(np.int64)
                    pcd_tmp = pcd.select_by_index(keep_local.tolist())
                    if pcd_is_empty(pcd_tmp):
                        self.log("[WARN] Largest cluster empty (unexpected). Reverting.")
                    else:
                        pcd = pcd_tmp
                        idx_work = idx_work[keep_local]
                        self.log(f"[INFO] Largest cluster kept, {len(keep_local)} pts.")
                        steps_done += 1
                        self.set_progress(steps_done * 100 / steps_total)
                        self._report_cloud("After DBSCAN", pcd)
                else:
                    self.log("[INFO] DBSCAN found no clusters (keeping cloud).")
                    steps_done += 1
                    self.set_progress(steps_done * 100 / steps_total)

            # 5) crop (manual mask to keep index mapping)
            if do_crop and bounds:
                self.log(f"[STEP] AABB crop with bounds: {bounds}")
                pts = np.asarray(pcd.points)
                mask = (
                    (pts[:, 0] >= bounds["min_x"]) & (pts[:, 0] <= bounds["max_x"]) &
                    (pts[:, 1] >= bounds["min_y"]) & (pts[:, 1] <= bounds["max_y"]) &
                    (pts[:, 2] >= bounds["min_z"]) & (pts[:, 2] <= bounds["max_z"])
                )
                keep_local = np.where(mask)[0].astype(np.int64)
                if keep_local.size == 0:
                    self.log("[WARN] Crop left 0 points. Reverting. Check your limits.")
                else:
                    pcd = pcd.select_by_index(keep_local.tolist())
                    idx_work = idx_work[keep_local]
                    steps_done += 1
                    self.set_progress(steps_done * 100 / steps_total)
                    self._report_cloud("After crop", pcd)

            # Result
            self.pcd_cleaned = pcd
            self.idx_cleaned = idx_work
            final_points = pcd_n_points(pcd)
            removed = initial_points - final_points
            pct = (removed / max(initial_points, 1)) * 100.0
            dt = time.time() - t0

            self.set_progress(100)
            self.set_status("Done.")
            self.log(f"[OK] CLEAN done in {dt:.2f}s")
            self.log(f"[SUMMARY] Initial: {initial_points:,} | Final: {final_points:,} | Removed: {removed:,} ({pct:.2f}%)")
            messagebox.showinfo(
                "CLEAN finished",
                f"Initial points: {initial_points:,}\n"
                f"Final points:   {final_points:,}\n"
                f"Removed:        {removed:,} ({pct:.2f}%)\n\n"
                f"To send to 3DGS, use 'Save CLEANED (GS-safe)'."
            )
        except Exception as e:
            self.set_status("Error")
            messagebox.showerror("CLEAN error", str(e))
            self.log(f"[ERROR] {e}")
        finally:
            self.btn_run.config(state="normal")

    # ---------- Previews ----------
    def preview_original(self):
        if self.pcd_original is None:
            messagebox.showinfo("Info", "Load a .PLY first.")
            return
        self.log("[PREV] ORIGINAL...")
        draw_quiet([self.pcd_original])

    def preview_current(self):
        if self.pcd_current is None:
            messagebox.showinfo("Info", "Load a .PLY first.")
            return
        self.log("[PREV] CURRENT...")
        draw_quiet([self.pcd_current])

    def preview_cleaned(self):
        if self.pcd_cleaned is None:
            messagebox.showinfo("Info", "Run CLEAN first.")
            return
        self.log("[PREV] CLEANED...")
        draw_quiet([self.pcd_cleaned])

    def reset_to_original(self):
        if self.pcd_original is None:
            return
        self.pcd_current = clone_pcd(self.pcd_original)
        self.pcd_cleaned = None
        if self.vertex_data0 is not None:
            self.idx_current = np.arange(len(self.vertex_data0), dtype=np.int64)
        self.idx_cleaned = None
        self.mapping_valid_cleaned = True
        self.set_progress(0)
        self.set_status("Reset to ORIGINAL.")
        self.log("[INFO] Reset to ORIGINAL.")
        self._apply_saved_orientation_if_any()

    # ---------- Report ----------
    def _report_cloud(self, title, pcd):
        n = pcd_n_points(pcd)
        if n == 0:
            self.log(f"[{title}] points=0 (empty)")
            return
        pts = np.asarray(pcd.points)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        self.log(f"[{title}] points={n:,}  bounds min={mn}  max={mx}")

# ---------- Main ----------
def main():
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = PointNukerApp(initial_path=initial)
    app.mainloop()

if __name__ == "__main__":
    main()
