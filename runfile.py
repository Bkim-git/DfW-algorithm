"""
@author: Byunguk Kim, Seoul National University
Ensure the directory structure and module dependencies are correctly set before running.
"""
import os, sys
WORKING_DIR = r"E:\05_Codes\02_DEV\DfW_dev"
LIB_DIR = os.path.join(WORKING_DIR, "Libs")
sys.path.append(LIB_DIR); os.chdir(WORKING_DIR)

import init, DfW

SITE = "Site"
# CASES = ("Case1", "Case2", "Case3")
CASES = ("Case1",)
RUN_ID = "run_win1.3"

def execute(case, valid = True):
    params = init.load_params(WORKING_DIR, SITE, case, run_id=RUN_ID)
    log_handle = init.setup_logging(params)
    try:
        vid, extent = init.load_vid(params)
        DfW.Operator(vid, extent, params).run()
        gt = init.load_gt(vid, extent, params)
        DfW.Visualization(vid, extent, params).run(gt)
    finally:
        init.close_logging(log_handle)

if __name__ == "__main__":
    for case in CASES:
        execute(case, valid = True)
