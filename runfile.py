# -*- coding: utf-8 -*-
"""
DfW Algorithm (v1.0)

Author: Byunguk Kim
Affiliation: Seoul National University

This script executes the Depth-from-Wave (DfW) algorithm

Modules:
- init: Initializes parameters, loads inputs and ground truth
- DfW: Core algorithm, post-processing, and visualization

Usage:
    Update the `working_dir` and ensure all dependencies are available.
"""

import os, sys

def main():    
    # Set working directory (update as needed)    
    working_dir =  r'A:/05_Codes/DfW_v1.0_package'
    os.chdir(working_dir)
    sys.path.append(working_dir+'/Libs')
    
    # Local imports
    import init, DfW
    
    # Initialize parameters and input 
    params         =  init.params(working_dir)
    Vid, extent, _ =  init.load_inputfile(params['Case'])
    groundtruth    =  init.load_gt(Vid[:,:,0], extent, params['Case'])
    
    # Run DfW processing
    result =  DfW.Operator(Vid, extent, params).run()
    DfW.postprocessing(params).run(result, groundtruth)    
    DfW.visualization(Vid, extent, params).run()

if __name__ == "__main__":
    main()