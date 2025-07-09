"""
* Initialization module for DfW
* @author: Byunguk Kim, Seoul National University
"""

print('- Initiating...')
# === Imports ===
import cv2, os, sys, time, rasterio, yaml, glob
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata

# === Global start time for execution time tracking ===
global start_time
start_time = time.time()

# === Parameter Initialization Function ===
def params(working_dir, Site, Case, idx=None, dirname=None):
    """
    Load and configure input parameters from config.yaml
    and create necessary output directories.
    """
    data_path = os.path.join(working_dir, 'data', Site, Case)
    config_path = os.path.join(data_path, 'config.yaml')

    # Default result folder name by timestamp
    if dirname is None:
        dirname = datetime.now().strftime("%Y-%m-%d")

    savedir = os.path.join(data_path, 'res', dirname)

    # Load YAML configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            try:
                params = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise ValueError(f"YAML parsing error: {e}")
    else:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Update with additional parameters
    params.update({
        'working_dir': working_dir,
        'savedir': savedir,
        'Site': Site,
        'Case': Case
    })

    # Ensure output directories exist
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, '__cache__'), exist_ok=True)

    # Print loaded parameters
    print('----------------- Parameters -----------------')
    for key, value in params.items():
        print(f'{key} : {value}')
    print('----------------------------------------------\n')
    print(f'- Exporting results to: {savedir}')

    return params
    

def load_inputfile(params):
    """
    Load input video as 3D numpy array (gray frames),
    and read spatial extent from associated metadata.
    """
    site, case = params['Site'], params['Case']
    data_location = os.path.join('data', site, case)
        
    # Load raw video file
    video_files = []
    for ext in ['*.mov', '*.mp4']:
        video_files.extend(glob.glob(os.path.join(data_location, ext)))

    if len(video_files) == 0:
        raise FileNotFoundError("No .mov or .mp4 file found.")
    elif len(video_files) > 1:
        raise RuntimeError(f"Multiple video files found: {video_files}")

    vid_path = video_files[0]
    cap = cv2.VideoCapture(vid_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Vid = np.zeros((height, width, frames), dtype=np.uint8)

    for i in range(frames):
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Frame {i} could not be read.")
        Vid[:, :, i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    # Remove invalid pixels (black columns/rows)
    Vid[np.any(Vid == 0, axis=2), :] = 0
    Vid = np.flip(Vid, axis=0)  # Flip vertically

    # Load spatial extent
    extent_path = os.path.join(data_location, 'extent.txt')
    if not os.path.exists(extent_path):
        raise FileNotFoundError('extent.txt file not found.')
    extent = np.loadtxt(extent_path).tolist()

    return Vid, extent


def load_gt(basemap, extent, params):
    site, case = params['Site'], params['Case']
    gt_dir = os.path.join('data', site, case, 'gt')
    
    filetype = next((ext for ext in ['tif', 'csv', 'xyz']
                     if os.path.exists(os.path.join(gt_dir, f'groundtruth.{ext}'))), None)
    gt_file = os.path.join(gt_dir, f'groundtruth.{filetype}')
    
    if filetype == 'tif':
        with rasterio.open(gt_file) as dataset:
            groundtruth = dataset.read(1)
            transform = dataset.transform
            width, height = dataset.width, dataset.height

            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
            xs = np.array(xs).reshape((height, width))
            ys = np.array(ys).reshape((height, width))

            # Crop to extent
            min_x_idx = np.argmin(np.abs(xs[0, :] - extent[0]))
            max_x_idx = np.argmin(np.abs(xs[0, :] - extent[1]))
            min_y_idx = np.argmin(np.abs(ys[:, 0] - extent[3]))
            max_y_idx = np.argmin(np.abs(ys[:, 0] - extent[2]))

            groundtruth = groundtruth[min_y_idx:max_y_idx, min_x_idx:max_x_idx]
            xs = xs[min_y_idx:max_y_idx, min_x_idx:max_x_idx]
            ys = ys[min_y_idx:max_y_idx, min_x_idx:max_x_idx]

            # Resize to match basemap
            target_shape = (basemap.shape[1], basemap.shape[0])
            groundtruth = cv2.resize(groundtruth, target_shape)
            xs = cv2.resize(xs, target_shape)
            ys = cv2.resize(ys, target_shape)    
    else:
        delimiter = ',' if filetype == 'csv' else None
        try:
            data = np.loadtxt(gt_file, delimiter=delimiter)
        except Exception as e:
            raise ValueError(f"Failed to load point cloud from {gt_file}: {e}")

        if data.shape[1] < 3:
            raise ValueError("Groundtruth CSV/XYZ must have at least 3 columns (X Y Z)")

        X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

        # Crop by extent (same logic as with TIFF)
        in_bounds = (
            (X >= extent[0]) & (X <= extent[1]) &
            (Y >= extent[2]) & (Y <= extent[3])
        )
        X, Y, Z = X[in_bounds], Y[in_bounds], Z[in_bounds]

        # Build target grid
        height, width = basemap.shape
        xi = np.linspace(extent[0], extent[1], width)
        yi = np.linspace(extent[2], extent[3], height)
        xs, ys = np.meshgrid(xi, yi)

        # Interpolate to grid
        points = np.column_stack((X, Y))
        groundtruth = griddata(points, Z, (xs, ys), method='linear')

    # === Final cleaning ===
    groundtruth[np.flip(basemap, axis=0) == 0] = np.nan
    groundtruth[(np.abs(groundtruth) > 1e2) | (groundtruth < 0)] = np.nan      
    
    return (xs, ys, groundtruth)
        
class logger:
    def __init__(self, params):
        self.terminal = sys.stdout
        self.filename = os.path.join(params['savedir'], r'console_log.txt')
    def write(self, message):
        self.terminal.write(message)
        try:
            with open(self.filename, "a", encoding="utf-8") as logfile:
                logfile.write(message)
        except Exception as e:
            self.terminal.write(f"\n[Logger Error] Could not write to log file: {e}\n")
    def flush(self):
        pass


