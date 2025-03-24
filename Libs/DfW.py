import os, time
import numpy as np
from scipy.interpolate import griddata
import modules as md
import pickle
from scipy.optimize import fsolve

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"       Subprocess: {func.__name__} [{elapsed_time:.2f} seconds]")
        return result
    return wrapper

class Operator():
    def __init__(self, Vid, extent, parameters):
        self.params   = parameters
        self.I        = self._get_cached_lanczos_resem(Vid)
        self.rows, self.cols, self.frames = self.I.shape
        self.x_coords = np.linspace(extent[0], extent[1], self.cols)
        self.y_coords = np.linspace(extent[2], extent[3], self.rows)
        self.dx       = self.x_coords[1] - self.x_coords[0]
        self.dt       = self.params.get('dt')
        self.log      = {
            'intergation_lc': None,     #  (x, y) 
            'intergation_id': None,     #  (x, y)
            'I_interg': None,           #  (Space, Time)
            'dynamic_modes': [],        #  (Segment, Mode, Space, Time)
            'frequency': [],            #  (Segment, Mode)
            'contribution': [],         #  (Segment, Mode)
            'wavenumber': [],           #  (Segment, Mode, Space, 2)
            'depth': [],                #  (Segment, Mode, Space)
            'epsilon': [],              #  (Segment, Mode, Space)
            'refraction': []            #  (Segment, Mode, Space)
        }
        
    @time_it    
    def _get_cached_lanczos_resem(self, Vid):
        if self.params['r_win'] != 1:
            return md.LanczosResem(Vid, self.params['r_win'])
        else:
            return Vid
        
    @time_it            
    def preprocessing(self):
        self.I = md.FreeFormMasker().run(self.I, self.params)
        self.generate_sparse_mesh()
          
    @time_it    
    def generate_sparse_mesh(self):
        step = int(self.params['jump'] / self.dx)
        x_coords_int, y_coords_int = np.meshgrid(self.x_coords[::step], self.y_coords[::step])
        x_indices, y_indices = np.meshgrid(np.arange(0, self.cols, step), np.arange(0, self.rows, step))
        intrgpoints = self.I[::step, ::step, 0]
        valid_pixels = ~intrgpoints.mask

        self.interg_pts = np.vstack((x_coords_int[valid_pixels].ravel(), y_coords_int[valid_pixels].ravel())).T
        self.indices = np.vstack((x_indices[valid_pixels].ravel(), y_indices[valid_pixels].ravel())).T 
        
        self.log['intergation_lc'] = self.interg_pts
        self.log['intergation_id'] = self.indices
        self.log['I_interg'] = self.I[self.indices[:,1], self.indices[:,0], :]

    def initial_window_config(self, T):
        if self.params['windowing'] == 'Adaptive':
            winsiz = self.params['win_coef'] * 1.56 * (T ** 2) / self.dx
        elif isinstance(self.params['windowing'], (int, float)):
            winsiz = self.params['windowing'] / self.dx
        else:
            raise ValueError("Unsupported windowing type specified")
        return np.array((winsiz, winsiz), dtype='float32')

    @time_it        
    def mode_decomposition(self, I_seg, seg_idx):        
        I_seg  =  md.ImageEnhancement(I_seg, self.params['dt'])
        DMs, T, cont = md.DMD(I_seg, self.params, seg_idx).run()
        
        self.log['dynamic_modes'].append(DMs)
        self.log['frequency'].append(1/T)
        self.log['contribution'].append(cont)
                
        return DMs, T, cont, len(T)
    
    def nonlinearity_correction(self, h_lin, theta, k, T):
        H_ref, h_ref, theta_ref = self.params['H_ref'], self.params['h_ref'], self.params['theta_ref']
        epsilon_storage = [None]
        refraction_storage = [None]
        k_deep = (2*np.pi/T)**2/9.81
        activation = lambda h, beta: 1/(1+np.exp(beta*(k_deep*h - 0.5)))
        def updating(h):
            dir_diff = min(abs(theta_ref-theta), np.radians(30))
            refraction = 1/np.cos(dir_diff)
            H = min(H_ref * (h_ref/h)**0.25 * refraction**0.5, h*0.78)
            epsilon = 0.5*H/h
            epsilon *= activation(h, beta = 10)
            epsilon_storage[0] = epsilon
            refraction_storage[0] = refraction
            
            return h - h_lin / (1 + epsilon)

        h_new = fsolve(updating, h_lin)[0]

        return h_new, epsilon_storage[0][0], refraction_storage[0]

    @time_it            
    def depth_estimator(self, seg_idx, nloc, T, DMs, cont):
        h_mode = np.zeros(nloc)
        winsiz_init = self.initial_window_config(T)
        
        for loc_idx in range(nloc):
            h_stack = []
            for j, T_each in enumerate(T):
                k,  h, theta = np.nan, np.nan, np.nan
                subwindow = winsiz_init[:, j] if self.params['windowing'] == 'Adaptive' else winsiz_init[:, None]
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
                        
                        wavelength = np.clip(np.array([np.abs(2*np.pi/np.maximum(k, 1e-10)/np.cos(theta)), 
                                                       np.abs(2*np.pi/np.maximum(k, 1e-10)/np.sin(theta))]), 
                                             30, 120)     
                        subwindow = self.params['win_coef'] * wavelength / self.dx                    
                    else: 
                        break
    
                kx, ky = k*np.cos(-theta), k*np.sin(-theta)        
                argument = (2*np.pi/T[j])**2/k/9.81
                h = 1/k*np.arctanh(argument) if not np.isnan(argument) and argument < 1 else np.nan
                                                            
                if self.params['NL_flag'] == 'on' and not np.isnan(h):
                    h, epsilon, refraction = self.nonlinearity_correction(h, theta, k, T_each)
                    self.log['epsilon'][-1][j, loc_idx] = epsilon
                    self.log['refraction'][-1][j, loc_idx] = refraction

                self.log['wavenumber'][-1][j, loc_idx, :] = [kx, ky]
                self.log['depth'][-1][j, loc_idx] = h
                
                h_stack.append(h)
                
            h_bar = np.nansum(h_stack * cont) / np.sum(cont[~np.isnan(h_stack)]) if np.sum(cont[~np.isnan(h_stack)]) != 0 else np.nan
            h_mode[loc_idx] = h_bar

        h_mode = md.Excludingoutlier(self.params['Sth'], self.interg_pts, h_mode, self.params['jump'])
        
        return h_mode
            
    @time_it                   
    def postprocessing(self, result):
        valid = ~np.isnan(result).any(axis=1)
        path = os.path.join(self.params['savedir'], '__cache__', 'log.pkl')
        with open(path, "wb") as file:
            pickle.dump(self.log, file)
        return result[valid, :]

    def run(self):
        start_time = time.time()
        self.preprocessing()
        nperseg    = int(self.params['VidSeg']/self.params['dt'])
        nseg, nloc = int(self.frames/nperseg), len(self.indices)

        nseg = 1
            
        h_seg = []
        for seg_idx in range(nseg): # Video Segmentation      
            print(f'[ DfW in progress: {seg_idx + 1}/{nseg} ]')    
            I_seg = self.I[:, :, seg_idx*nperseg : (seg_idx + 1)*nperseg]            
            DMs, T, cont, nmode = self.mode_decomposition(I_seg, seg_idx)
            # print('DMD done: ', T, cont)

            self.log['wavenumber'].append(np.full((nmode, nloc, 2), np.nan, dtype = np.float32))
            self.log['depth'].append(np.full((nmode, nloc), np.nan, dtype = np.float32))
            self.log['epsilon'].append(np.full((nmode, nloc), np.nan, dtype = np.float32))        
            self.log['refraction'].append(np.full((nmode, nloc), np.nan, dtype = np.float32))        

            h_mode = self.depth_estimator(seg_idx, nloc, T, DMs, cont)

            h_seg.append(h_mode)
            
        h_mean = np.nanmean(np.array(h_seg), axis=0)
        h_std = np.nanstd(np.array(h_seg), axis=0)
        
        result = np.column_stack((self.interg_pts, h_mean, h_std))
        result = self.postprocessing(result) 

        print(f' Total elapsed time: [{(time.time()-start_time)/60:.2f} min]')
        
        return np.column_stack((self.interg_pts, h_mean, h_std))
    
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
        result_stack = result_stack[~np.isnan(result_stack).any(axis=1)]
        return result_stack
           
    def Exporting(self, result_stack):
        header = r'X(m), Y(m), GT(m), h(m), std(m)'
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
                
    def run(self, result, groundtruth):
        start_time = time.time()
        print('- Exporting the DfW results...', end="")
        
        gt = self.Groundtruth_interpolation(result[:,:2], groundtruth)
        result_stack = self.Stacking(result, gt)
        
        self.Exporting(result_stack)
        print(f'\r- Exporting the DfW results [{time.time()-start_time:.2f}s]')
        
        return result_stack
    
#%%
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["figure.dpi"]  = 300

class visualization():        
    def __init__(self, Vid, extent, params):
        # Basemap generation
        self.basemap = Vid[:,:,0].astype(np.float32)
        self.basemap[self.basemap == 0] = np.nan
        
        # Plotting parameters
        self.hran = (1, 6)
        self.cmap = plt.cm.get_cmap('Spectral', 15)
        self.savedir = params['savedir']
        self.current_flag = params.get('Current')
        self.NL_flag = params['NL_flag']
        self.figsize = (4,4)
        self.er = (-1.5, 1.5)
        self.offset = [276000, 3951000]
        self.extent = [extent[0] - self.offset[0], 
                       extent[1] - self.offset[0], 
                       extent[2] - self.offset[1], 
                       extent[3] - self.offset[1]]  
        # self.er = (-1.5, 1.51)

    def getDfWresult(self):
        load_path = os.path.join(self.savedir, r'results.csv')
        return np.loadtxt(load_path, delimiter=',', skiprows=1)
    
    def plotting_parameter(self, ran, cmap):
        level_filled = np.linspace(ran[0], ran[1] + 1e-3, 30)
        level_lines = np.arange(ran[0], ran[1] + 1e-3, 0.5)
        ticks =  np.arange(level_filled.min(),level_filled.max(), 1)
        cmap = plt.cm.get_cmap('gist_earth_r', 30)
        return level_filled, level_lines, ticks, cmap

    def MappingEstimate(self, result_stack):
        print("Plotting DfW estimates...")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin='lower', extent = self.extent, cmap='bone')
        
        x, y, z = result_stack[:,0], result_stack[:,1], result_stack[:,3]
        x, y, z = x[~np.isnan(z)], y[~np.isnan(z)], z[~np.isnan(z)]
                        
        lv_f, lv_l, ticks, cmap = self.plotting_parameter(self.hran, cmap = 'Spectral')       
        
        ct = ax.tricontourf(x - self.offset[0], y - self.offset[1], 
                            z, levels = lv_f, cmap = cmap)
        
        cbar = plt.colorbar(ct, ax = ax, label='Water depth (m)', shrink = 0.7)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        
        ctl = ax.tricontour(x - self.offset[0], y - self.offset[1],
                            z, levels = lv_l, colors = 'black', linewidths = 0.3)       
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
                
        lv_f, lv_l, ticks, cmap = self.plotting_parameter(self.hran, cmap = 'Spectral')   
        
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
               
    def MappingStd(self, result_stack):
        print("Plotting Std...")
        Std = result_stack[:,4]

        x, y, z = result_stack[:,0], result_stack[:,1], Std
        x, y, z = x[~np.isnan(z)], y[~np.isnan(z)], z[~np.isnan(z)]
        
        ran = [0, 1.01]
        z = np.clip(z, ran[0], ran[1])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin='lower', extent = self.extent, cmap='bone')
        
        ct = ax.tricontourf(x - self.offset[0], y - self.offset[1], z, 
                            levels = np.linspace(ran[0],ran[1], 15), 
                            cmap = 'Spectral_r',
                            alpha = 1.0)
        cbar = plt.colorbar(ct, ax = ax, label='Standard deviation (m)', shrink = 0.7)
        
        major_ticks = np.round(np.arange(ran[0], ran[1], 0.2), 1)
        cbar.set_ticks(major_ticks)
        cbar.set_ticklabels(major_ticks)
        ax.tricontour(x - self.offset[0], y - self.offset[1], z,
                      levels = np.linspace(ran[0],ran[1], 11), 
                      colors = 'black',
                      linewidths = 0.2)
        
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.tight_layout()    
        full_path = os.path.join(self.savedir, r'Std')
        plt.savefig(full_path)
        
        
    def MappingNonlinearity(self):
        print("Plotting Nonlinearity...")
        
        path = os.path.join(self.savedir, '__cache__', 'log.pkl')
        with open(path, "rb") as file:
            log = pickle.load(file)
            
        dirpath = os.path.join(self.savedir, 'Nonlinearity_parameters')
            
        intergation_lc = log['intergation_lc']
        epsilon = log['epsilon']

        epsilon_stack = []
        for seg, inepsilon_pseg in enumerate(epsilon):
            for mode in range(inepsilon_pseg.shape[0]): 
                epsilon_stack.append(inepsilon_pseg[mode, :])

        " Average "
        epsilon_stack = np.array(epsilon_stack)
       
        x, y, z = intergation_lc[:,0], intergation_lc[:,1], np.nanmean(epsilon_stack, 0)
        
        x, y, z = x[~np.isnan(z)], y[~np.isnan(z)], z[~np.isnan(z)]
        ran = [0, 0.3]
        z = np.clip(z, ran[0], ran[1])
        
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin='lower', extent = self.extent, cmap='bone')
        
        ct = ax.tricontourf(x - self.offset[0], y - self.offset[1], z, 
                            levels = np.linspace(ran[0],ran[1], 15), 
                            cmap = 'pink_r',
                            alpha = 1.0)
        cbar = plt.colorbar(ct, ax = ax, label='Nonlinearity parameter (m)', shrink = 0.7)
        
        major_ticks = np.round(np.linspace(ran[0], ran[1], 5), 1)
        # major_ticks = np.linspace(ran[0], ran[1], 5)
        cbar.set_ticks(major_ticks)
        cbar.set_ticklabels(major_ticks)
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
                
        full_path = os.path.join(dirpath, 'epsilon_stack.csv')
        np.savetxt(full_path, np.array(epsilon_stack).T, fmt='%.6f', delimiter=',')
        
        full_path = os.path.join(dirpath, 'epsilon_stack_average.csv')
        epsilon_stack_average = np.hstack((x, y, z))
        np.savetxt(full_path, epsilon_stack_average, fmt='%.6f', delimiter=',')
        
    def run(self):
        result = self.getDfWresult()
        self.MappingEstimate(result)
        self.MappingGroundTruth(result)
        self.MappingDiff(result)
        self.MappingStd(result)
        if self.NL_flag == 'on':
            self.MappingNonlinearity()
            
        plt.close('all')
        
