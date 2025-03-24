# Depth-from-Wave (DfW) Algorithm

A Python-based implementation of the **Depth-from-Wave (DfW)** algorithm for water depth mapping from surface wave video.

## 📌 Overview

This tool extracts bathymetry from surface wave video by analyzing the dynamic wave modes and solving the dispersion relationship. It includes preprocessing, adaptive windowing, wave parameter estimation, nonlinearity correction, and result visualization.

## 📁 Project Structure
DfW/ ├── runfile.py # Entry point to execute the DfW algorithm ├── init.py # Initialization and input loading ├── modules.py # Utility functions and core wave analysis tools ├── DfW.py #   Main DfW operator and postprocessing modules ├── data/ │ └── Example/ │ ├── input.npz # Input wave video and spatial coordinates │ └── gt/ │ └── groundtruth.tif # Ground truth for validation
