# Depth-from-Wave (DfW) algorithm

This is a Python implementation of the **DfW** algorithm for water depth mapping from coastal surface wave video.

### Description
This tool retrieves nearshore water depth map from surface wave video by analyzing dynamic wave patterns and solving the nonlinear dispersion relation.

### Project Structure
```
DfW/  
├── runfile.py               # Entry point to execute the DfW algorithm  
├── Libs/                    # Library folder containing core modules  
│   ├── init.py              # Initialization and input loading  
│   ├── modules.py           # Utility functions and core wave analysis tools  
│   └── DfW.py               # Main DfW operator and postprocessing modules  
└── data/                    # Example dataset  
    └── Example/
        └── Case1/  
            ├── input.mov            # Input video file
            ├── extent.txt           # Local coordinates
            ├── basemap.png          # Sattelite image within the extent
            ├── config.yaml          # Input parameters
            ├── groundtruth/    
            │   ├── groundtruth.tif  # Ground-truth data
            └── res/                 # Results will be saved here
```


### Getting Started
1. Clone the repository or download the code.
2. Ensure you have the required Python packages:
   ```bash
   pip install numpy scipy matplotlib opencv-python scikit-image rasterio pydmd
3. Edit the working_dir path in runfile.py to point to your project directory.
4. Run the script:
   python runfile.py

### Parameters
You can adjust parameters in init.py → params():
- dt: Frame interval (seconds)
- VidSeg: Video segment length (seconds)
- jump: Grid spacing for interrogation points (meter)
- windowing: Adaptive or fixed interrogation window
- H_ref, h_ref, theta_ref: Reference wave parameters
- Masking, NL_flag: Toggle masking or nonlinearity correction

### Output

```
data/Example/res/YYYY-MM-DD/
├── results.csv       # X, Y, ground-truth, estimation, std
├── report.txt        # Parameters used
├── Estimate.png      # Estimated bathymetry map
├── Groundtruth.png   # Ground truth depth map
└── Diff.png          # Difference map
```


## Citation
If you use this tool in your research, please cite:
Kim, B., Park, Y. S., Noh, H., & Lee, M. (2025). Improving accuracy of image-based depth inversion with an adaptive window optimization. Coastal Engineering Journal, 67(2), 306-318.
Kim, B., Noh, H., Park, Y. S., & Lee, M. (2023). Non-spectral linear depth inversion using drone-acquired wave field imagery. Applied Ocean Research, 138, 103625.
## Contact
Feel free to contact me via **https://bkim-git.github.io/**
