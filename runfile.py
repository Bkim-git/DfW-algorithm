<<<<<<< HEAD
"""
@author: Byunguk Kim, Seoul National University

This script executes the Depth from Wave (DfW) algorithm.
It includes loading input data, running the core operator,
and performing postprocessing and visualization.

Ensure the directory structure and module dependencies are correctly set before running.
"""

# === Standard Library Imports ===
import os, sys

# === Set Working Directory ===
# Specify the main directory where the DfW code and libraries reside
working_dir = r'E:\05_Codes\DfW_package\DfW_1.1'
os.chdir(working_dir)
sys.path.append(os.path.join(working_dir, 'Libs'))

# === Local Module Imports ===
import init, DfW

# === Configuration ===
Site  = 'Example'  # Site identifier (used for file naming and folder structures)
Cases = ['Case0']

# === Main Execution ===
def main():
    for Case in Cases:
        # Load input video, validation data, and configuration parameters for each case.
        params = init.params(working_dir, Site, Case, dirname=None)
        Vid, extent = init.load_inputfile(params)
        groundtruth = init.load_gt(Vid[:, :, 0], extent, params)
        
        # Redirect standard output to a log file
        sys.stdout = init.logger(params)
        
        # Perform the DfW algorithm and visualize the results.
        DfW.operator(Vid, extent, params, gt=groundtruth).run()
        DfW.postprocessing(params).run(groundtruth)
        DfW.visualization(Vid, extent, params, max_depth=6, interval=0.5).run()

if __name__ == "__main__":
    main()









=======
"""
@author: Byunguk Kim, Seoul National University

This script executes the Depth from Wave (DfW) algorithm.
It includes loading input data, running the core operator,
and performing postprocessing and visualization.

Ensure the directory structure and module dependencies are correctly set before running.
"""

# === Standard Library Imports ===
import os, sys

# === Set Working Directory ===
# Specify the main directory where the DfW code and libraries reside
working_dir = r'E:\05_Codes\DfW_package\DfW_1.1'
os.chdir(working_dir)
sys.path.append(os.path.join(working_dir, 'Libs'))

# === Local Module Imports ===
import init, DfW

# === Configuration ===
Site  = 'Example'  # Site identifier (used for file naming and folder structures)
Cases = ['Case1']
Mode = 'Mapping' # 'Point' or 'Mapping'

# === Main Execution ===
def mapping():
    for Case in Cases:
        # Load input video, validation data, and configuration parameters for each case.
        params = init.params(working_dir, Site, Case, dirname=None)
        Vid, extent = init.load_inputfile(params)
        groundtruth = init.load_gt(Vid[:, :, 0], extent, params)
        
        # Redirect standard output to a log file
        sys.stdout = init.logger(params)
        
        # Perform the DfW algorithm and visualize the results.
        DfW.operator(Vid, extent, params, gt=groundtruth).run()
        DfW.postprocessing(params).run(groundtruth)
        DfW.visualization(Vid, extent, params, max_depth=6, interval=0.5).run()



if __name__ == "__main__":
    main()









>>>>>>> 9b2a775 (Initial commit with full project and Git LFS tracking)
