"""
Initialization utilities for the DfW processing pipeline.

------
Byunguk Kim, PhD
https://bkim-git.github.io/

"""
import time
from pathlib import Path
from datetime import datetime

import cv2
import yaml
import rasterio
import numpy as np

from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------
# Execution time reference
# ---------------------------------------------------------------------
t_start = time.time()

def load_params(
    working_dir,
    site,
    case,
    run_id=None,
    verbose=True,
):
    """
    Load configuration from config.yaml, initialize output directories,
    and return a unified parameter dictionary.

    Parameters
    ----------
    working_dir : str or Path
        Project root directory.
    site : str
        Site identifier.
    case : str
        Case identifier.
    run_id : str or None, optional
        Output subdirectory name. If None, uses current date (YYYY-MM-DD).
    verbose : bool, optional
        If True, print parameters and write a configuration report.

    Returns
    -------
    dict
        Dictionary containing configuration parameters and runtime metadata.
    """
    
    base_dir = Path("data") / site / case
    config_path = base_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d")

    savedir = base_dir / "res" / run_id
    cache_dir = savedir / "__cache__"

    savedir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load YAML configuration
    # ------------------------------------------------------------------
    try:
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration: {exc}")

    # ------------------------------------------------------------------
    # Augment with runtime metadata
    # ------------------------------------------------------------------
    params = {
        **cfg,
        "working_dir": Path(str(working_dir)),
        "savedir": Path(str(savedir)),
        "site": site,
        "case": case,
    }

    # ------------------------------------------------------------------
    # Logging and reporting
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 50)
        print("Parameter Configuration")
        print("-" * 50)
        for k, v in params.items():
            print(f"{k:20s}: {v}")
        print("=" * 50 + "\n")

        report_path = savedir / "Report_configurations.txt"
        with report_path.open("w") as f:
            f.write("========== Parameter Configuration ==========\n\n")
            f.write(f"Site   : {site}\n")
            f.write(f"Case   : {case}\n")
            f.write(f"savedir: {savedir}\n\n")
            f.write("--------------- Parameters ----------------\n")
            for k, v in params.items():
                f.write(f"{k} : {v}\n")
            f.write("-------------------------------------------\n")

    return params


def load_vid(params):
    """
    Load a single input video as a 3D NumPy array of grayscale frames
    and read the corresponding spatial extent.

    Parameters
    ----------
    params : dict
    
    Returns
    -------
    Vid : ndarray (T, H, W), uint8
        Grayscale video frames ordered in time.
    extent : list
        Spatial extent loaded from extent.txt.
    """
    site = params.get("site")
    case = params.get("case")

    data_dir = Path("data") / site / case
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # ------------------------------------------------------------------
    # Locate video file
    # ------------------------------------------------------------------
    video_files = []
    for pattern in ("*.mp4", "*.mov"):
        video_files.extend(data_dir.glob(pattern))

    if len(video_files) == 0:
        raise FileNotFoundError("No video file (.mp4 or .mov) found.")
    if len(video_files) > 1:
        raise RuntimeError(f"Multiple video files found: {video_files}")

    vid_path = video_files[0]

    # ------------------------------------------------------------------
    # Read video frames
    # ------------------------------------------------------------------
    print("- Loading video data ...", end="", flush=True)
    t0 = time.time()

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {vid_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video = np.empty((nframes, height, width), dtype=np.uint8)

    for i in range(nframes):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Frame read failed at index {i}")
        video[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    # ------------------------------------------------------------------
    # Load spatial extent
    # ------------------------------------------------------------------
    extent_path = data_dir / "extent.txt"
    if not extent_path.exists():
        raise FileNotFoundError(f"extent.txt not found: {extent_path}")

    extent = np.loadtxt(extent_path).tolist()
    
    print(f"\r- Video data imported [{time.time()-t0:.1f} s]")

    return video, extent


def load_gt(video, extent, params):
    """
    Load ground-truth bathymetry and map it onto the video grid.

    The bathymetry provided as a GeoTIFF is interpolated onto the
    spatial domain of the video. The resulting field is stored for
    later reuse to avoid repeated interpolation.
    """

    site = params.get("site")
    case = params.get("case")

    gt_dir = Path("data") / site / case
    if not gt_dir.exists():
        return None

    print("- Loading ground-truth data ...", end="", flush=True)
    t0 = time.time()

    # --------------------------------------------------------------
    # Video grid definition
    # --------------------------------------------------------------
    video_mean = np.mean(video, axis=0)
    Ny, Nx = video_mean.shape

    # --------------------------------------------------------------
    # Cached result
    # --------------------------------------------------------------
    npy_path = gt_dir / "__groundtruth.npy"
    if npy_path.exists():
        gt = np.load(npy_path)
        print(f"\r- Ground-truth data imported [{time.time() - t0:.1f} s]")
        return gt

    # --------------------------------------------------------------
    # Load bathymetry GeoTIFF
    # --------------------------------------------------------------
    tif_path = gt_dir / "groundtruth.tif"
    if not tif_path.exists():
        print("\r- Ground-truth data not found")
        return None

    with rasterio.open(tif_path) as ds:
        z = ds.read(1).astype(float)

        Ny_t, Nx_t = z.shape
        tr = ds.transform

        dx = tr.a
        dy = tr.e
        x0_t = tr.c
        y0_t = tr.f

        x_t = x0_t + dx * (0.5 + np.arange(Nx_t))
        y_t = y0_t + dy * (0.5 + np.arange(Ny_t))

    # --------------------------------------------------------------
    # Remove unrealistic depth values
    # --------------------------------------------------------------
    z[(np.abs(z) < 1e-2) | (np.abs(z) > 1e2)] = np.nan

    # --------------------------------------------------------------
    # Interpolate bathymetry onto video domain
    # --------------------------------------------------------------
    interp = RegularGridInterpolator(
        (y_t, x_t),
        z,
        bounds_error=False,
        fill_value=np.nan
    )

    x_min, x_max, y_min, y_max = extent
    x_grid = np.linspace(x_min, x_max, Nx)
    y_grid = np.linspace(y_min, y_max, Ny)
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    pts = np.column_stack((Yg.ravel(), Xg.ravel()))
    gt = interp(pts).reshape(Ny, Nx)

    # Align orientation with video coordinates
    gt = np.flipud(gt)

    # --------------------------------------------------------------
    # Mask outside video footprint
    # --------------------------------------------------------------
    threshold = np.nanquantile(np.abs(gt), 0.95)
    video_mask = ~np.isnan(np.flip(video_mean, axis=0))
    valid = video_mask & (np.abs(gt) < threshold)
    gt[~valid] = np.nan

    # --------------------------------------------------------------
    # Save for later use
    # --------------------------------------------------------------
    np.save(npy_path, gt)

    print(f"\r- Ground-truth data imported and aligned "
          f"[{time.time() - t0:.2f} s]")

    return gt

