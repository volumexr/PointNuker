# PointNuker

**PointNuker** is a GUI tool to clean unwanted “floaters” or noisy points that often appear after training a **3D Gaussian Splatting (3DGS)** scene.  
It provides **cluster-based filtering (DBSCAN)**, a **multi-step cleaning pipeline**, and a **real-time viewer** to validate every step.  
Safe saving options preserve all Gaussian attributes (GS-safe) so your 3DGS data remains intact.

---

## ✨ Features
- Remove floating/noisy points from trained 3DGS point clouds.  
- **Cluster Finder (DBSCAN):** keep only the main cluster by adjusting `eps` and `min_points`.  
- **Pipeline Cleaning:**  
  - Radius outlier removal  
  - Statistical outlier removal  
  - Axis-Aligned Bounding Box (AABB) crop  
- **3DGS Mode:** ensures 1:1 mapping, preserving SH, opacity, scale, etc.  
- **Real-time visualization:** compare **ORIGINAL**, **CURRENT**, and **CLEANED** clouds.  
- **GS-safe saving:** keeps extra attributes required for 3D Gaussian Splatting.  

---

## 🛠 Requirements

- Python 3.8+  
- Modules:  
  ```text
  open3d
  plyfile
  numpy
  ttkbootstrap  # optional (for modern UI theme)
  ```

### Installation
```bash
# Create and activate environment (recommended)
conda create -n pointnuker python=3.10 -y
conda activate pointnuker

# Install required modules
python -m pip install open3d plyfile numpy ttkbootstrap
```

Or, install from a `requirements.txt`:
```bash
python -m pip install -r requirements.txt
```

---

## 🚀 Usage

### Launch GUI
```bash
python PointNuker.py
```

### Auto-load a point cloud
```bash
python PointNuker.py path/to/your_cloud.ply
```

---

## 🧹 Suggested Workflow

1. **Open .PLY** file → check in **Preview ORIGINAL**.  
2. (Optional) **Enable 3DGS Mode** → prevents voxel downsampling and keeps full Gaussian attributes.  
3. **Cluster Finder (DBSCAN)** → adjust `eps` & `min_points` → click *Detect largest cluster NOW*.  
   - Option: “Keep only largest cluster during CLEAN.”  
   - Use *Parameter Assistant* for auto-suggestions.  
4. **Pipeline CLEAN** →  
   - Apply **Radius Outlier Removal**  
   - Apply **Statistical Outlier Removal**  
   - Apply **AABB Crop** (if needed)  
   - Optionally keep only the largest cluster  
   - Validate with **Preview CURRENT / CLEANED**.  
5. **Save Results** →  
   - **Save CLEANED (GS-safe)** or **Save CURRENT (GS-safe)** → preserves SH/opacity/scale.  
   - *Open3D save options* strip extra attributes, useful only for previews.  

---

## 🔧 Extra Tools
- **Orientation helpers** (view only): Flip vertical (Rx=180°), Swap Y/Z.  
- **Help menu** includes About & Changelog.  

---

## ❗ Troubleshooting
- **`ModuleNotFoundError: No module named 'open3d'`**  
  → Make sure your environment is active and install requirements.  
- **“GS-safe unavailable” when saving**  
  → Some steps broke the 1:1 mapping (e.g., voxel). Try again with **3DGS Mode ON** and without voxel downsampling.  

---

## 📜 License
This tool is provided as-is, free for research and production workflows.  
