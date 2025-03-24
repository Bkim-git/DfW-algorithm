"""
* Initiation
* Updated on January 10, 2024
* @author: Byunguk Kim

"""

print('- Initiating...')
import os
import time
import cv2
import rasterio
import numpy as np
from datetime import datetime

start_time = time.time()

def params(working_dir):
    parameter_config = {'Working_dir'     :  working_dir,                       #  Location of the working directory
                        'Case'            :  'Example',                         #  Case configuration
                        'dt'              :  0.5,
                        'VidSeg'          :  40,                                #  Video segmentation [s]
                        'jump'            :  20,                                #  (m), Interogation interval
                        'windowing'       :  "Adaptive",                        #  'Adaptive'(default) or specific window size in meter
                        'win_coef'        :  1.5,                               #  User-specified values
                        'r_win'           :  0.5,                               #  Resampling rate within an interrogation window (default = 0.5)
                        'Sth'             :  0.03,                              #  Threshold for outlier exclusion (default = 0.03)
                        'Masking'         :  'on',                              #  Flag for masking (on or off)
                        'NL_flag'         :  'on',                              #  Flag for nonlinearity correction (on or off)
                        'H_ref'           :  0.54,                              # (m), Reference wave height
                        'h_ref'           :  2.02,                              # (m), Reference water depth
                        'theta_ref'       :  np.radians(29.8)                   # (rad), Reference wave direction (clockwise from east)
                        }

    savedir = os.path.join(parameter_config.get('Working_dir'), 
                           'data', 
                           parameter_config.get('Case'), 
                           'res',  
                           datetime.now().strftime("%Y-%m-%d"))
    
    parameter_config['savedir'] = savedir
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, '__cache__'), exist_ok=True)
    
    print('----------------- Parameters -----------------')
    for key, value in parameter_config.items():
        print(f'{key} : {value}')
    print('---------------------------------------------- \n')    
    print(f'- Exporting results to: {savedir}')
                
    return parameter_config

        
def load_inputfile(Case): 
    data_location = f'data/{Case}/' 
    load_npz =  np.load(data_location + r'input.npz')
    Vid      =  load_npz['arr_0']              
    xi, yi   =  load_npz['arr_1'], load_npz['arr_2']      
    extent   =  [np.nanmin(xi), np.nanmax(xi), np.nanmin(yi), np.nanmax(yi)]
    report   =  load_npz['arr_3']
    
    return Vid, extent, report

def load_gt(basemap, extent, Case, MWL = 0): 
    data_location = f'data/{Case}/gt/'
    filepath = data_location + r'groundtruth.tif'

    dformat = filepath[-3:]
    if dformat == 'csv':
        groundtruth =  np.genfromtxt(filepath, 
                                     delimiter=',',
                                     skip_header=1)
        xs = groundtruth[:,0]
        ys = groundtruth[:,1]
        groundtruth = groundtruth[:,2]
        
    elif dformat == 'tif':
        with rasterio.open(filepath) as dataset:
            groundtruth = dataset.read(1)
            transform = dataset.transform
            
            width = dataset.width
            height = dataset.height
                           
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
            xs = np.array(xs).reshape((height, width))
            ys = np.array(ys).reshape((height, width))
            
            min_x_idx = np.argmin(np.abs(xs[0, :] - extent[0]))
            max_x_idx = np.argmin(np.abs(xs[0, :] - extent[1]))
            min_y_idx = np.argmin(np.abs(ys[:, 0] - extent[3]))
            max_y_idx = np.argmin(np.abs(ys[:, 0] - extent[2]))
            
            min_x_idx = np.argmin(np.abs(xs[0, :] - extent[0]))
            max_x_idx = np.argmin(np.abs(xs[0, :] - extent[1]))
            min_y_idx = np.argmin(np.abs(ys[:, 0] - extent[3]))
            max_y_idx = np.argmin(np.abs(ys[:, 0] - extent[2]))
                    
            groundtruth = groundtruth[min_y_idx:max_y_idx, min_x_idx:max_x_idx]
            xs = xs[min_y_idx:max_y_idx, min_x_idx:max_x_idx]
            ys = ys[min_y_idx:max_y_idx, min_x_idx:max_x_idx]
            
            reshape = (basemap.shape[1], basemap.shape[0])
            groundtruth = cv2.resize(groundtruth, reshape)
            xs, ys = cv2.resize(xs, reshape), cv2.resize(ys, reshape)
            groundtruth[np.flip(basemap, axis=0) == 0] = np.nan
            groundtruth[np.abs(groundtruth) > 1e2] = np.nan

    return (xs, ys, groundtruth)
