import os, time
import numpy as np
from scipy.interpolate import griddata
import modules as md
import pickle
from scipy.optimize import fsolve
from tqdm import tqdm
from contextlib import redirect_stderr

import matplotlib.pyplot as plt

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"    Subprocess: {func.__name__} [{elapsed_time:.2f} seconds]")
        return result
    return wrapper

class operator():
    def __init__(self, Vid, extent, parameters, gt = None):
        self.params   = parameters
        self.I        = self._get_cached_lanczos_resem(Vid, extent)
        self.rows, self.cols, self.frames = self.I.shape
        self.x_coords = np.linspace(extent[0], extent[1], self.cols)
        self.y_coords = np.linspace(extent[2], extent[3], self.rows)
        self.dx       = self.params['dx']
        self.dt       = self.params['dt']
        self.gt       = gt
        
        if self.params['flag_Logging']:
            self.log      = {
                'intergation_lc': None,     #  (x, y) 
                'intergation_id': None,     #  (x, y)
                'Umap': None,
                'I_interg': None,           #  (Space, Time)
                'dynamic_modes': [],        #  (Segment, Mode, Space, Time)
                'frequency': [],            #  (Segment, Mode)
                'contribution': [],         #  (Segment, Mode)
                'wavenumber': [],         #  (Segment, Mode, Space, 2)
                'depth': [],              #  (Segment, Mode, Space)
                'epsilon': []              #  (Segment, Mode, Space)
            }
        
    @time_it    
    def _get_cached_lanczos_resem(self, Vid, extent):
        dx = (extent[1] - extent[0])/Vid.shape[1]
        if dx < 0.5:
            return md.LanczosResem(Vid, dx, self.params.get('dx'))
        else:
            return Vid
        
    @time_it            
    def preprocessing(self):
        if self.params['flag_whitecapping']:
            self.I = md.WhitecappingRemoval(self.I)
        self.I = md.FreeFormMasker().run(self.I, self.params)
        self.generate_sparse_mesh()
        
        if not self.gt is None:
            points = np.column_stack((self.gt[0].ravel(), self.gt[1].ravel()))
            values = self.gt[2].ravel()
            valid = ~(np.isnan(values) | (values == 0))
            self.gt = griddata(points[valid], values[valid], self.interg_pts, method='linear')
            valid = ~np.isnan(self.gt)
            self.interg_pts = self.interg_pts[valid, :]
            self.indices  = self.indices[valid, :]    
            self.gt = self.gt[valid]
        else:
            print('No groundtruth data is given.')
        
        if hasattr(self, 'log'):
            self.log['intergation_lc'] = self.interg_pts
            self.log['intergation_id'] = self.indices
            self.log['I_interg'] = self.I[self.indices[:,1], self.indices[:,0], :]

    @time_it    
    def generate_sparse_mesh(self):
        step = int(self.params['jump'] / self.dx)
        x_coords_int, y_coords_int = np.meshgrid(self.x_coords[::step], self.y_coords[::step])
        x_indices, y_indices = np.meshgrid(np.arange(0, self.cols, step), np.arange(0, self.rows, step))
        intrgpoints = self.I[::step, ::step, 0]
        valid_pixels = ~intrgpoints.mask
        
        self.interg_pts = np.vstack((x_coords_int[valid_pixels].ravel(), y_coords_int[valid_pixels].ravel())).T
        self.indices = np.vstack((x_indices[valid_pixels].ravel(), y_indices[valid_pixels].ravel())).T 
        
    def initial_window_config(self, T):
        winsiz = self.params['win_coef'] * 1.56 * (T ** 2) / self.dx
        return np.array((winsiz, winsiz), dtype=np.int32)
      
    def mode_decomposition(self, I_seg, seg_idx):        
        I_seg  =  md.ImageEnhancement(I_seg, self.params['dt'])
        DMs, T, cont = md.DMD(I_seg, self.params, seg_idx).run()
        if hasattr(self, 'log'):
            self.log['dynamic_modes'].append(DMs)
            self.log['frequency'].append(1/T)
            self.log['contribution'].append(cont)       
        return DMs, T, cont, len(T)
    
    def nonlinearity_correction(self, h_lin, k, theta, T, loc_idx):
        " Parameters "
        H_ref     = self.params.get('H_ref')
        h_ref     = self.params.get('h_ref')
        theta_ref = self.params.get('theta_ref')
        alpha     = self.params.get('alpha')
        beta      = self.params.get('beta')

        k_deep = (2*np.pi/T)**2 / 9.81  # Vectorized    
        if self.params.get('flag_Activation'):
            activation = 1 / (1 + np.exp(beta*(k_deep*h_lin - 0.5)))
        else:
            activation = np.ones(len(k_deep))
        
        if isinstance(theta_ref, (int, float)):
            refraction = 1/np.cos(abs(theta_ref-theta))
            refraction[refraction > 2] = 1
        else: refraction = np.ones(len(k_deep))
        
        h_updated = np.full_like(h_lin, np.nan)

        for jj in range(len(h_lin)):
            if not np.isfinite(h_lin[jj]) or h_lin[jj] <= 1e-6:
                continue    
            
            def updating(h):
                if self.params.get('flag_Activation') == 'full':     
                    Gamma = lambda k, h: np.where(
                        (k > 1e-8) & (h > 1e-6),
                        np.sqrt(np.tanh(k*h) / k) * (1 + 2*k*h / np.sinh(2*k*h)),
                        np.nan
                    )
                    k_ref = (2*np.pi/T[jj])/np.sqrt(9.81*h_ref)
                    G = Gamma(k[jj], h)
                    G_ref = Gamma(k_ref, h_ref)
                    ratio = G/G_ref
                    # H = min(H_ref * ratio**0.5 * refraction[jj]**0.5, h * 0.78) if np.isfinite(ratio) else np.nan
                    H = min(H_ref * ratio**0.5 * refraction[jj]**0.5, h * 0.78)
                else:
                    H = min(H_ref * (h_ref/h)**0.25 * refraction[jj]**0.5, h*0.78)
                
                epsilon = alpha*H/h * activation[jj]
                updating.last_epsilon = epsilon
                return h - h_lin[jj] / (1 + epsilon)
            
            with open(os.devnull, 'w') as f, redirect_stderr(f):
                h_updated[jj] = fsolve(updating, h_lin[jj])[0]
            # try: h_updated[jj] = fsolve(updating, h_lin[jj])[0]
            # except: 
            #     print(f"Cannot solve the objective function at loc_idx:{loc_idx} / mode:{jj+1}th")
            #     continue
            
            if hasattr(self, 'log'):
                self.log['epsilon'][-1][jj, loc_idx] = updating.last_epsilon

        return h_updated

    @time_it            
    def depth_estimator(self, seg_idx, nloc, T, DMs, cont):
        # print('       Subprocess: depth_estimator', end="")
        h_est = np.empty(nloc, np.float32)
        winsiz_init = self.initial_window_config(T)
        nmode = len(T)
        
        h_est = np.full(nloc, np.nan)
        for loc_idx in tqdm(range(nloc), desc="    Interrogation in progress"):
            k_modes = np.full(nmode, np.nan)
            theta_modes = np.full(nmode, np.nan)
            h_modes = np.full(nmode, np.nan)
            for j, T_each in enumerate(T):
                k,  theta = np.nan, np.nan
                subwindow = winsiz_init[:, j]
                tol, count, k_old = 1, 0, None
                while tol > 1e-2 and count < 10:
                    count += 1
                    DM_win = md.subwindowing(DMs[:,:,j], self.indices[loc_idx,:], subwindow)

                    if isinstance(DM_win, np.ndarray) and np.sum(DM_win.mask)/DM_win.size < 0.5:
                        Proj, theta, dx_proj = md.RadonTransform(DM_win.real, dx=self.dx)
                        k  = md.wavenumest(Proj, dx_proj)
                        if np.isnan(k): 
                            break
                        if k_old is not None:
                            tol = abs((k - k_old) / k_old)     
                        k_old = k
                        
                        wavelength = np.clip(np.array([
                            2 * np.pi / (np.maximum(k, 1e-10) * np.maximum(np.abs(np.cos(theta)), 1e-10)),
                            2 * np.pi / (np.maximum(k, 1e-10) * np.maximum(np.abs(np.sin(theta)), 1e-10))
                        ]), 20, 120)

                        subwindow = self.params['win_coef'] * wavelength / self.dx                    
                    else: 
                        break
                
                k_modes[j], theta_modes[j] = k, np.mod(theta, 2*np.pi)
                argument = (2*np.pi/T[j])**2/k/9.81
                h_modes[j] = 1/k*np.arctanh(argument) if not np.isnan(argument) and argument < 1 else np.nan
                
                if hasattr(self, 'log'):        
                    self.log['wavenumber'][-1][j, loc_idx, :] = [k*np.cos(theta), k*np.sin(theta) ]
            
            if self.params['flag_Nonlinearity']:
                h_modes = self.nonlinearity_correction(h_modes, k_modes, theta_modes, T, loc_idx)

            h_est[loc_idx] = np.nansum(h_modes * cont) / np.sum(cont[~np.isnan(h_modes)]) if np.sum(cont[~np.isnan(h_modes)]) != 0 else np.nan

        h_est = md.Excludingoutlier(self.params['Sth'], self.interg_pts, h_est, self.params['jump'])            
        return h_est
 
    @time_it                   
    def postprocessing(self, result):        
        path = os.path.join(self.params['savedir'], '__cache__', 'log.pkl')
        if hasattr(self, 'log'):    
            with open(path, "wb") as file:
                pickle.dump(self.log, file)
        return result[~np.isnan(result[:, 2])]

    def run(self):
        start_time = time.time()
        self.preprocessing()
        nperseg    = int(self.params['VidSeg']/self.params['dt'])
        nseg, nloc = int(self.frames/nperseg), len(self.indices)
        
        h_seg = []
        # nseg = 1
        for seg_idx in range(nseg): # Video Segmentation      
            print(f'[ DfW in progress: {seg_idx + 1}/{nseg} ]')    
            I_seg = self.I[:, :, seg_idx*nperseg : (seg_idx + 1)*nperseg]            
            DMs, T, cont, nmode = self.mode_decomposition(I_seg, seg_idx)

            if hasattr(self, 'log'):
                self.log['wavenumber'].append(np.full((nmode, nloc, 2), np.nan, dtype = np.float32))
                self.log['epsilon'].append(np.full((nmode, nloc), np.nan, dtype = np.float32))            

            h_est = self.depth_estimator(seg_idx, nloc, T, DMs, cont)
            h_seg.append(h_est)
        
        h_mean = np.nanmean(np.array(h_seg), axis=0)
        h_std = np.nanstd(np.array(h_seg), axis=0)
        
        result = np.column_stack((self.interg_pts, h_mean, h_std))   
        result = self.postprocessing(result)
        
        path = os.path.join(self.params['savedir'], '__cache__', 'DfW-results.npy')
        np.save(path, result)

        print(f' Total elapsed time: [{(time.time()-start_time)/60:.2f} min]')

        self.__dict__.clear()

#%%
class postprocessing():
    def __init__(self, params):
        self.params = params
        
    def Groundtruth_interpolation(self, coords, groundtruth):
        x = groundtruth[0].flatten()
        y = groundtruth[1].flatten() 
        gt = groundtruth[2].flatten()
        gt = griddata((x, y), gt, coords, method='linear')   
        return gt
    
    def Stacking(self, result, gt):
        interg_pts = result[:,:2]
        DfW_results = result[:,2:]
        result_stack = np.hstack((interg_pts, gt[:,np.newaxis], DfW_results))
        return result_stack
           
    def Exporting(self, result_stack):
        header = r'X(m), Y(m), GT(m), h(m), std(m), Ux(m/s), Uy(m/s)'
        path = os.path.join(self.params['savedir'], r'results.csv')
        np.savetxt(path, 
                   result_stack, 
                   delimiter=',', 
                   header=header, 
                   comments='', 
                   fmt='%.6f')
        
        path = os.path.join(self.params['savedir'], r'report.txt')
        with open(path, 'w') as file:
            file.write(r"Parameters \n")
            for key, value in self.params.items():
                file.write(f"{key} : {value}\n")
                
    def run(self, groundtruth):
        start_time = time.time()
        print('- Exporting the DfW results...', end="")
        result = np.load(os.path.join(self.params['savedir'], '__cache__', r'DfW-results.npy'))
        gt = self.Groundtruth_interpolation(result[:,:2], groundtruth)
        result_stack = self.Stacking(result, gt)
        self.Exporting(result_stack)
        print(f'\r- Exporting the DfW results [{time.time()-start_time:.2f}s]')
    
#%%
import matplotlib.pyplot as plt
import cv2
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["figure.dpi"]  = 300

class visualization():        
    def __init__(self, Vid, extent, params, max_depth = 10, interval = 1.0):
        # Basemap generation
        try:
            path  = os.path.join(params['savedir'], r'..\..\basemap.png')
            self.basemap = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.basemap = np.flipud(np.array(cv2.cvtColor(self.basemap, cv2.COLOR_BGR2RGB)))
        except:
            self.basemap = Vid[:,:,0].astype(np.float32)
            self.basemap[self.basemap == 0] = np.nan
        
        # Plotting parameters
        self.hran = (0, max_depth)
        self.interval = interval
        self.cmap = plt.cm.get_cmap('Spectral', 15)
        self.savedir = params['savedir']
        self.current_flag = params.get('flag_Currents')
        self.params = params
        self.mask = np.where(np.load(os.path.join(self.savedir, '..\..\mask.npy')), 1.0, np.nan)
        
        # self.savedir = r'E:\05_Codes\DfW_v1.0\data\MLP\D1T3\res\2025-06-21'
        
        self.NL_flag = params.get('flag_Nonlinearity')
        self.figsize = (4,4)
        self.er = (-2, 2)
        self.offset = [extent[0] - (extent[0] % 100), extent[2] - (extent[2] % 100)]
        self.extent = [extent[0] - self.offset[0], 
                       extent[1] - self.offset[0], 
                       extent[2] - self.offset[1], 
                       extent[3] - self.offset[1]]  
        # self.er = (-1.5, 1.51)
        
    def getDfWresult(self):
        load_path = os.path.join(self.savedir, r'results.csv')
        return np.loadtxt(load_path, delimiter=',', skiprows=1)
    
    def plotting_parameter(self, z):
        if self.hran[1] is None:        
            level_filled = np.linspace(0, np.ceil(np.nanmax(z)) + 1e-3, 30)
            level_lines = np.arange(0, np.ceil(np.nanmax(z)) + 1e-3, self.interval)
        else:
            level_filled = np.linspace(self.hran[0], self.hran[1] + 1e-3, 30)
            level_lines = np.arange(self.hran[0], self.hran[1] + 1e-3, self.interval)        
        ticks =  np.arange(level_filled.min(),level_filled.max(), 1)
        # cmap = plt.cm.get_cmap('gist_earth_r', 30)
        # cmap = plt.cm.get_cmap('RdYlBu', 30)
        cmap = plt.cm.get_cmap('Spectral', 30)
        return level_filled, level_lines, ticks, cmap

    def MappingEstimate(self, result_stack):
        print("Plotting DfW estimates...")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin='lower', extent = self.extent)
        
        x, y, z = result_stack[:,0], result_stack[:,1], result_stack[:,3]
        mask_valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        x, y, z = x[mask_valid], y[mask_valid], z[mask_valid]
        x -= self.offset[0]
        y -= self.offset[1]
        
        lv_f, lv_l, ticks, cmap = self.plotting_parameter(z)       
        
        ct = ax.tricontourf(x, y, z, 
                            levels = lv_f, cmap = cmap)

        cbar = plt.colorbar(ct, ax = ax, label='Water depth (m)', shrink = 0.7)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        
        ctl = ax.tricontour(x, y, z,
                            levels = lv_l, colors = 'black', linewidths = 0.3)       
        plt.clabel(ctl, inline=True, fmt='%.1f', fontsize=7)
        
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
        ax.xaxis.offsetText.set_visible(True)
        ax.yaxis.offsetText.set_visible(True)
        ax.yaxis.get_offset_text().set_position((0, 1.02))
        ax.xaxis.get_offset_text().set_position((1.02, 0))
        
        plt.tight_layout()    
        full_path = os.path.join(self.savedir, r'Estimate')
        plt.savefig(full_path)
        
    def MappingGroundTruth(self, result_stack):  
        print("Plotting groundTruth map...")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin='lower', extent = self.extent, cmap='bone')
        
        x, y, z = result_stack[:,0], result_stack[:,1], result_stack[:,2]
        x, y, z = x[~np.isnan(z)], y[~np.isnan(z)], z[~np.isnan(z)]
                
        lv_f, lv_l, ticks, cmap = self.plotting_parameter(z)   
        
        ct = ax.tricontourf(x - self.offset[0], y - self.offset[1], z, levels = lv_f, cmap = cmap, alpha = 1.0)
        cbar = plt.colorbar(ct, ax = ax, label='Water depth (m)', shrink = 0.7)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        
        ctl = ax.tricontour(x - self.offset[0], y - self.offset[1], z, levels = lv_l, colors = 'black', linewidths = 0.3)
        plt.clabel(ctl, inline=True, fmt='%.1f', fontsize=7)
        
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')       
        
        plt.tight_layout()    
        full_path = os.path.join(self.savedir, r'Groundtruth')
        plt.savefig(full_path)

    def MappingDiff(self, result_stack):
        print("Plotting errors...")
        Diff = (result_stack[:,3] - result_stack[:,2])
        Diff[Diff > self.er[1]] = self.er[1]

        x, y, z = result_stack[:,0], result_stack[:,1], Diff
        x, y, z = x[~np.isnan(z)], y[~np.isnan(z)], z[~np.isnan(z)]
        
        # z -= np.nanmedian(Diff)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin='lower', extent = self.extent, cmap='bone')
        
        ct = ax.tricontourf(x - self.offset[0], y - self.offset[1],z, 
                            levels = np.linspace(self.er[0], self.er[1] + 1e-3, 20), 
                            cmap = 'bwr',
                            alpha = 1.0)
        cbar = plt.colorbar(ct, ax = ax, label='Difference (m)', shrink = 0.7)
        major_ticks = np.arange(self.er[0], self.er[1] + 1e-3, 0.5)
        cbar.set_ticks(major_ticks)
        cbar.set_ticklabels(major_ticks)
        ax.tricontour(x - self.offset[0], y - self.offset[1], z,
                      levels = np.linspace(self.er[0], self.er[1] + 1e-3, 15), 
                      colors = 'black',
                      linewidths = 0.2)
        
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.tight_layout()    
        full_path = os.path.join(self.savedir, r'Diff')
        plt.savefig(full_path)
        
    def MappingNonlinearity(self):
        print("Plotting Nonlinearity...")
        
        path = os.path.join(self.savedir, '__cache__', 'log.pkl')
        with open(path, "rb") as file:
            log = pickle.load(file)
            
        dirpath = os.path.join(self.savedir, 'Nonlinearity_parameters')
            
        intergation_lc = log['intergation_lc']
        epsilon = log['epsilon']
        # frequency = log['frequency']

        epsilon_stack = []
        for seg, inepsilon_pseg in enumerate(epsilon):
            for mode in range(inepsilon_pseg.shape[0]): 
                # T = 1/frequency[seg][mode]
                epsilon_stack.append(inepsilon_pseg[mode, :])
  
        " Average "
        epsilon_stack = np.array(epsilon_stack)
       
        x, y, z = intergation_lc[:,0], intergation_lc[:,1], np.nanmean(epsilon_stack, 0)
        x, y, z = x[~np.isnan(z)], y[~np.isnan(z)], z[~np.isnan(z)]

        ran = [0, np.round(np.nanmax(z),1)]
        z = np.clip(z, ran[0], ran[1])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin='lower', extent = self.extent, cmap='bone')
        
        ct = ax.tricontourf(x - self.offset[0], y - self.offset[1], z, 
                            levels = np.linspace(ran[0],ran[1], 15), 
                            cmap = 'pink_r',
                            alpha = 1.0)
        cbar = plt.colorbar(ct, ax = ax, label='Nonlinearity parameter (m)', shrink = 0.7)

        # major_ticks = np.round(np.linspace(ran[0], ran[1], 5), 2)
        # cbar.set_ticks(major_ticks)
        # cbar.set_ticklabels(major_ticks)
        
        ax.tricontour(x - self.offset[0], y - self.offset[1], z,
                      levels = np.linspace(ran[0],ran[1], 11), 
                      colors = 'black',
                      linewidths = 0.2)
        
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])        
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        plt.tight_layout()    
            
        os.makedirs(dirpath, exist_ok=True)
        full_path = os.path.join(dirpath, r'Averaged.png')
        plt.savefig(full_path)
        plt.close('all')

        full_path = os.path.join(dirpath, 'epsilon_stack_average.csv')
        epsilon_stack_average = np.stack((x, y, z), axis = 1)
        np.savetxt(full_path, epsilon_stack_average, fmt='%.6f', delimiter=',')

    def run(self):
        result = self.getDfWresult()
        self.MappingEstimate(result)
        self.MappingGroundTruth(result)
        self.MappingDiff(result)

        if self.params['flag_Nonlinearity']:
            self.MappingNonlinearity()

        plt.close('all')
        
