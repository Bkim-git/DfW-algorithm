# Depth-from-Wave (DfW) algorithm

This is a Python implementation of the **DfW** algorithm for water depth mapping from coastal surface wave video.

### Description
This tool retrieves nearshore water depth map from surface wave video by analyzing dynamic wave patterns and solving the nonlinear dispersion relation.

### Project Structure
```
DfW/  
├── runfile.py                   # Entry point to execute the DfW algorithm  
├── Libs/                        # Library folder containing core modules  
│   ├── init.py                  # Initialization and input loading  
│   ├── modules.py               # Utility functions and core wave analysis tools  
│   └── DfW.py                   # Main DfW operator and visualization modules  
└── data/                        # Example dataset  
    └── Site_example/
	    └── Case_example/  
			├── input.mp4        # Input video file
			├── extent.txt       # ROI range in local coordinate (xmin, xmax, ymin, ymax)
			├── config.yaml      # Configuration parameters
			├── groundtruth.tif  # Groundtruth data in .tif or .csv format.
			└── res/             # Results will be saved here  
```


### Getting Started
1. Clone the repository or download the code.
2. Ensure you have the required Python packages:
   ```bash
   pip install numpy opencv-python matplotlib joblib tqdm scikit-image scipy pydmd

3. Edit the working_dir path in runfile.py to point to your project directory.
4. Run the script:

   python runfile.py

### Parameters
Parameters can be adjusted in `config.yaml`.

- `dx`: Target spatial grid spacing (m)
- `dt`: Temporal sampling interval (s)
- `VidSeg`: Video segmentation length per batch (s)
- `step`: Spatial stride for the interrogation grid (m)
- `n_jobs`: Number of parallel worker processes
<!-- Spectral / modal -->
- `win_coef`: Window size scaling factor for local wavenumber estimation (default = 1.5)
- `modenum`: Number of retained dominant modes in DMD (default = 6)
- `Sth`: Threshold for outlier rejection based on local slope criteria (m/m, default = 0.3)
- `fp`: Peak frequency (Hz) (If available)
<!-- Flags -->
- `flag_masking`: Enable user-defined spatial mask
- `flag_nonlinearity`: Enable nonlinear wave correction module
- `flag_bandpass`: Enable temporal bandpass filtering


### Output

```
data/Example/res/YYYY-MM-DD/
├── output.npy        # Pickled file containing all intermediate DfW results
└── Estimate.png      # Estimated depth map
```


## Citation
If you find this tool helpful in your research, please consider citing the following works:
- Kim, B., Park, Y. S., Noh, H., & Baek, S. (2026). Nonlinearity-corrected kinematic depth inversion from UAV imagery in irregular tidal flats: Application to Byeonsan Beach, South Korea. Coastal Engineering, 204, 104904.
- Kim, B., Park, Y. S., Noh, H., & Lee, M. (2025). Improving accuracy of image-based depth inversion with an adaptive window optimization. Coastal Engineering Journal, 67(2), 306-318.
- Kim, B., Noh, H., Park, Y. S., & Lee, M. (2023). Non-spectral linear depth inversion using drone-acquired wave field imagery. Applied Ocean Research, 138, 103625.

## Contact
Feel free to contact me via "https://bkim-git.github.io/"
