"""
@author: Byunguk Kim, Seoul National University
Ensure the directory structure and module dependencies are correctly set before running.
"""
import os, sys
# WORKING_DIR = r"Path to the working directory containing this runfile"
WORKING_DIR = r"E:\05_Codes\01_Published\DfW_package\DfW_v1.2"
LIB_DIR = os.path.join(WORKING_DIR, "Libs")
sys.path.append(LIB_DIR); os.chdir(WORKING_DIR)

import init, DfW

SITE = "Site_example"
CASES = ("Case_example",)
RUN_ID = None

def execute(case):
    params = init.load_params(WORKING_DIR, SITE, case, run_id=RUN_ID)
    vid, extent = init.load_vid(params)
    gt = init.load_gt(vid, extent, params)

    DfW.Operator(vid, extent, params, gt=gt).run()
    DfW.Visualization(vid, extent, params).run()

if __name__ == "__main__":
    for case in CASES:
        execute(case)
