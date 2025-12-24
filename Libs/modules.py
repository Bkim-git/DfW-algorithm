""" Required Libarires """
import numpy as np
import cv2
import os
from pathlib import Path
from skimage.transform import radon
import matplotlib.pyplot as plt

# import numpy.ma as ma
from scipy.signal import welch
from pydmd import OptDMD
from scipy.ndimage import zoom
from scipy.signal import butter, filtfilt
from multiprocessing import shared_memory
from contextlib import redirect_stderr
from scipy.optimize import fsolve

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydmd")

"""
01. Preprocessings
"""
def LanczosResem(I, gt, dx, dy, target):
    r_x = dx / target
    r_y = dy / target

    Nx_new = int(I.shape[1] * r_x)
    Ny_new = int(I.shape[0] * r_y)
    new_size = (Nx_new, Ny_new)
    
    if gt is not None:
        gt = cv2.resize(gt, new_size, interpolation=cv2.INTER_LANCZOS4)

    Nt = I.shape[2]
    I_rs = np.empty((Ny_new, Nx_new, Nt), dtype=np.float32)

    for i in range(Nt):
        frame = I[:, :, i]
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError(f"Invalid slice at index {i} with shape {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError(f"Slice {i} contains NaN or Inf")
        I_rs[:, :, i] = cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)

    return I_rs, gt

def subwindowing(frame, interg_point, winsiz):
    x_idx, y_idx = interg_point
    x_min  =  max(int(x_idx - winsiz[0]//2), 0)
    x_max  =  min(int(x_idx + winsiz[0]//2), frame.shape[1])
    y_min  =  max(int(y_idx - winsiz[1]//2), 0)
    y_max  =  min(int(y_idx + winsiz[1]//2), frame.shape[0])
    windowsize = (x_max - x_min)*(y_max - y_min)
    if windowsize > winsiz[0]*winsiz[1]*(2/3):
        return frame[y_min:y_max, x_min:x_max]
    else:
        return None

def WhitecappingRemoval(I):
    I = I.astype(np.float32)
    rows, cols, frames = I.shape
    I_wcr = np.zeros_like(I)
    for f in range(2, frames):
        avg_prev = 0.5 * (I[:, :, f-1] + I[:, :, f-2])
        I_wcr[:, :, f] = I[:, :, f] - avg_prev
    return I_wcr  

class FreeFormMasker:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.img, self.img_display, self.current_mask = None, None, None
        self.points, self.masks_history = [], []

    def initialize_window(self, window_name):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_events)

    def mouse_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.points = [(x, y)]
            self.current_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.img_display, (self.points[-1][0], self.points[-1][1]), (x, y), (0, 0, 0), 1)
                self.points.append((x, y))
                self.redraw_image()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.points.append((self.ix, self.iy))  # Close the shape
            self.fill_current_selection(self.points)

    def fill_current_selection(self, points):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)
        mask_floodfill = mask.astype(bool)
        mask_floodfill &= ~np.isnan(self.img)
        self.masks_history.append(mask_floodfill)
        self.redraw_image()

    def select_free_form(self, img_input):
        self.img = img_input.copy()
        self.img_display = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.masks_history = []

        window_name = "Image Selection Tool"
        self.initialize_window(window_name)

        while True:
            cv2.imshow(window_name, self.img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 8:  # Backspace
                if self.masks_history:
                    self.masks_history.pop()
                    self.redraw_image()
                    
        cv2.destroyAllWindows()
        return np.any(self.masks_history, axis=0) if self.masks_history else np.zeros_like(self.img, dtype=bool)

    def redraw_image(self):
        self.img_display = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for mask in self.masks_history:
            self.img_display[mask] = (0, 0, 0)
        if self.drawing:
            for i in range(len(self.points) - 1):
                cv2.line(self.img_display, self.points[i], self.points[i + 1], (0, 0, 0), 5)

    def adjust_mask_to_image(self, mask, I):
        if mask.shape != I.shape[:2]:
            ratio_h = I.shape[0] / mask.shape[0]
            ratio_w = I.shape[1] / mask.shape[1]
            mask_resized = zoom(mask, (ratio_h, ratio_w), order=0)
        else:
            mask_resized = mask
        return mask_resized

    def run(self, I, params):
        rows, cols, frames = I.shape
        firstframe = I[:, :, 0]
        mask4nan = (firstframe == 0) | np.isnan(firstframe)
        mask_path = Path("data") / params["site"] / params["case"] / "mask.npy"
        
        if os.path.isfile(mask_path):
            mask = np.load(mask_path)
        else:
            if params['flag_masking']:
                MeanImg = np.mean(I[:,:,::int(frames/100)], axis=2)
                MeanImg = MeanImg.astype(np.uint8)
                # masker = FreeFormMasker()
                mask4dry = self.select_free_form(MeanImg)
            else:
                mask4dry = np.zeros_like(firstframe, dtype=bool)
            mask = (mask4nan | mask4dry)
            np.save(mask_path, mask)
        
        mask = self.adjust_mask_to_image(mask, I)
        mask = np.broadcast_to(mask[:, :, None], I.shape)
        return np.ma.masked_where(mask, I)

def ImageEnhancement(I, dt):
    if not np.ma.isMaskedArray(I):
        I = np.ma.masked_array(I, mask=False)
    I = I.astype(np.float32, copy=False)
    I -= np.ma.mean(I, axis=2, keepdims=True)
    I /= np.ma.std(I)
    return I

"""
02. Wave Analysis
"""
class DMD():
    def __init__(self, I, params, seg_idx, dt):
        self.I = I
        self.rows, self.cols, self.frames = I.shape
        self.dt = dt
        self.seg_idx = seg_idx + 1
        self.params = params
        self.r = params.get('modenum')
        self.flag_bandpass = params.get('flag_bandpass')
        
        if np.ma.isMaskedArray(I):
            self.I_valid = I.compressed().reshape(-1, self.frames)
            self.mask = np.ma.getmaskarray(I[:,:,0])
        else:
            self.I_valid = I.reshape(-1, self.frames)
            self.mask = np.zeros((self.rows, self.cols), dtype=bool)   
            
        self.savedir = Path(params["savedir"]) / "DMD_RESULTS"
        os.makedirs(self.savedir, exist_ok=True)
                    
        self.T, self.cont = None, None
        
    def peak_frequency(self, step = 20, nperseg=None, noverlap=None):
        I_sparse = self.I[::step, ::step, :]
        I_sparse = I_sparse.reshape(-1, I_sparse.shape[2])
        freq, psd = welch(I_sparse, fs=1 / self.dt, 
                          nperseg=I_sparse.shape[1], 
                          axis=1)
        return freq[np.argmax(np.mean(psd, axis=0))]
    
    def reshaping(self, Phi):
        Phi_rshp = np.full((self.rows * self.cols, Phi.shape[1]),
                           np.nan, dtype=complex)
        Phi_rshp[~self.mask.ravel(), :] = Phi
        Phi_rshp = Phi_rshp.reshape(self.rows, self.cols, -1)
        Phi_rshp = np.transpose(Phi_rshp, (2, 0, 1))
        return np.ma.masked_invalid(Phi_rshp)
    
    def PlottingSpectrum(self,freq_valid, fp):
        f, psd = welch(self.I_valid, fs=1/self.dt, nperseg=self.frames, axis=-1)
        Q25, Q75, Median = np.quantile(psd, 0.25, axis=0), np.quantile(psd, 0.75, axis=0), np.median(psd, axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.fill_between(f, Q25 / np.max(Median), Q75 / np.max(Median), color='lightgray', alpha=0.5)
        ax.plot(f, Median / np.max(Median), color='black', linestyle='solid')
        ax.bar(1 / self.T, self.cont, width=0.01, color='tab:red')

        ax.axvline(x=freq_valid[0], linestyle='--', color='k', linewidth=0.5)
        ax.axvline(x=freq_valid[1], linestyle='--', color='k', linewidth=0.5)
        ax.axvline(x=fp, linestyle='--', color='k', linewidth=1.0)
        
        ax.text(0.8, 0.95, f'nmode = {len(self.T)}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='left')
    
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('P / Pmax [-]')
        plt.savefig(Path(self.savedir) / f"Spectrum_Seg{self.seg_idx}", dpi=300)
        plt.close('all')

    def PlottingDynamicModes(self):
        Phi = np.array(self.Phi.real, copy=True)
        T, cont = self.T, self.cont
        nmode = Phi.shape[0]
    
        ncol = int(np.ceil(np.sqrt(nmode)))
        nrow = int(np.ceil(nmode / ncol))
    
        vmax = np.percentile(np.abs(Phi[np.isfinite(Phi)]), 99)
    
        fig, axes = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow))
        axes = np.atleast_2d(axes)
    
        for j in range(nmode):
            ax = axes[j // ncol, j % ncol]
            ax.imshow(Phi[j, :, :].real, cmap='RdYlBu',
                      vmin=-vmax, vmax=vmax, origin='lower')
            ax.set_title(f"Mode {j+1}\nTp={T[j]:.2f}s, c={cont[j]*100:.1f}%", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    
        for k in range(nmode, nrow*ncol):
            axes[k // ncol, k % ncol].axis('off')
    
        plt.tight_layout()
        plt.savefig(Path(self.savedir) / f"DynamicModes_Seg{self.seg_idx}.png", dpi=300)
        plt.close('all')

    def _bandpass_filter(self, freq_valid):
        fs = 1.0 / self.dt
        f1, f2 = np.sort(freq_valid)
        nyq = 0.5 * fs
        low = f1 / nyq
        high = f2 / nyq
        b, a = butter(5, [low, high], btype='band')
        self.I_valid = filtfilt(b, a, self.I_valid, axis=1, method="pad")

    def dmd(self, fp):
        freq_valid = np.array([max(0.6*fp, 1/32), min(1.5*fp, 1/2)], dtype=float)
        period_valid = 1 / freq_valid
        
        if self.flag_bandpass:
            self._bandpass_filter(freq_valid)
            
        dmd = OptDMD(svd_rank = 3*self.r, opt=True)
        dmd.fit(self.I_valid)
 
        omega = np.log(dmd.eigs) / self.dt
        T     =  np.divide(2 * np.pi, np.where(omega.imag != 0, omega.imag, np.nan))
        phi = dmd.modes
        b = np.linalg.lstsq(phi, self.I_valid[:, 0], rcond=None)[0]
        cont = np.linalg.norm(phi * b[np.newaxis, :], axis=0)**2
        
        mask = (T >= period_valid[1]) & (T <= period_valid[0])
        valid_idx = np.argsort(-cont[mask])[:self.r]
        valid_idx = np.where(mask)[0][valid_idx]

        self.Phi  =  self.reshaping(phi[:,valid_idx])
        self.T = T[valid_idx]
        self.cont = cont[valid_idx]/np.sum(cont[valid_idx])
        
        self.PlottingSpectrum(freq_valid, fp)
        self.PlottingDynamicModes()
        
    def run(self):
        fp = self.params['fp'] if 'fp' in self.params else self.peak_frequency()
        self.dmd(fp)
        return self.Phi, self.T, self.cont


def k_solver(Phi, dx):
    Phi = np.nan_to_num(
        Phi.real.astype(np.float32, copy=False),
        nan=0.0
    )
    Phi = cv2.GaussianBlur(Phi, (3, 3), 0)

    ntheta = min(max(Phi.shape), 181)
    theta = np.linspace(0.0, 180.0, ntheta, dtype=np.float32)
    sino = radon(Phi, theta=theta, circle=False)

    idx = np.nanargmax(np.sum(sino * sino, axis=0))
    theta_peak = -np.deg2rad(theta[idx])

    proj = sino[:, idx]
    proj /= (np.abs(np.cos(theta_peak)) + np.abs(np.sin(theta_peak))) * max(Phi.shape)

    if np.isnan(proj).any():
        x = np.arange(proj.size)
        m = ~np.isnan(proj)
        proj = np.interp(x, x[m], proj[m])

    n = proj.size
    n_pad = 1 << (n - 1).bit_length()
    fft = np.fft.rfft(proj, n_pad) / n
    freq = np.fft.rfftfreq(n_pad, dx)
    mag = np.abs(fft)

    mag[(freq <= 1/(n*dx)) | (freq >= 0.5)] = 0.0
    pk = np.argmax(mag)

    if 0 < pk < mag.size - 1:
        c = np.polyfit(freq[pk-1:pk+2], mag[pk-1:pk+2], 2)
        k = -c[1] / (2*c[0]) if c[0] != 0 else freq[pk]
    else:
        k = freq[pk]

    return (2*np.pi*k if k > 0 else np.nan), theta_peak


def depth_single_point(loc_idx,
                       indices,
                       shm_data_name, shm_mask_name,
                       shp, dtype_name,
                       T, cont, dx, win_coef):

    shm_data = shared_memory.SharedMemory(name=shm_data_name)
    data = np.ndarray(shp, dtype=np.dtype(dtype_name), buffer=shm_data.buf)
    shm_mask = shared_memory.SharedMemory(name=shm_mask_name)
    mask = np.ndarray(shp, dtype=bool, buffer=shm_mask.buf)

    DMs = np.ma.masked_array(data, mask=mask)

    nmode = len(T)
    hstack = np.full(nmode, np.nan)
    kstack = np.full((nmode, 2), np.nan)

    winsiz_init = np.minimum(win_coef*1.56*(T**2)/dx, 200/dx)
    
    x, y = indices[loc_idx]

    for j, T_each in enumerate(T):
        k, theta = np.nan, np.nan
        subwindow = np.array([winsiz_init[j], winsiz_init[j]])
        tol, count, k_old = 1, 0, None
        
        DM_slice = DMs[j]
        while tol > 1e-2 and count < 10:
            count += 1
            DM_win = subwindowing(DM_slice, (x, y), subwindow)

            if isinstance(DM_win, np.ndarray) and np.sum(DM_win.mask)/DM_win.size < 0.5:
                k, theta = k_solver(DM_win, dx)

                if np.isnan(k): 
                    break

                if k_old is not None:
                    tol = abs((k - k_old) / (k_old + 1e-6))
                k_old = k

                wavelength = np.clip(np.array([
                    2*np.pi / (np.maximum(k, 1e-10) * np.maximum(abs(np.cos(theta)), 1e-10)),
                    2*np.pi / (np.maximum(k, 1e-10) * np.maximum(abs(np.sin(theta)), 1e-10))
                ]), 20, 200)

                subwindow = (win_coef * wavelength / dx).astype(int)
            else:
                break
        
        theta = np.mod(theta, 2*np.pi)
        kstack[j] = np.array([k*np.cos(theta), k*np.sin(theta)], dtype=float)

        argument = (2*np.pi/T[j])**2 / (k+1e-8) / 9.81
        hstack[j] = 1/k*np.arctanh(argument) if not np.isnan(argument) and argument < 1 else np.nan

    return hstack, kstack

def nonlinearity_correction(h_ini, T, H0, h0, alpha = 0.5, beta = 10):
    k_deep = (2*np.pi/T)**2 / 9.81  
    activation = lambda hk0: 1 / (1 + np.exp(beta*(hk0 - 0.5)))

    h_updated = np.full_like(h_ini, np.nan)
    for j, h_ini_ in enumerate(h_ini):
        if not np.isfinite(h_ini[j]) or h_ini[j] <= 1e-6:
            continue    
        
        def updating(h):
            H = min(H0*(h0/h)**0.25, 0.78*h)
            epsilon = alpha*H/h * activation(h*k_deep[j])
            return h - h_ini_ / (1 + epsilon)

        with open(os.devnull, 'w') as f, redirect_stderr(f):
            h_updated[j] = fsolve(updating, h_ini_)[0]
            
    return h_updated
#%%
"""
04. Postprocessings
"""
from scipy.spatial import KDTree
def Excludingoutlier(Sth, distance, interg_pts, h):
    h = np.asarray(h)
    coordinates = interg_pts
    tree = KDTree(coordinates)
    outlier_count = 0
    for i in range(len(h)):
        indices = tree.query_ball_point(coordinates[i], distance*1.05)
        indices.remove(i)
        neighbors = h[indices]
        if len(neighbors) > 0:
            slope = np.arctan(np.abs(h[i] - np.nanmean(neighbors))/distance)
            if slope > Sth:
                h[i] = np.nan
                outlier_count += 1
       
    percentage_replaced = (outlier_count / len(h)) * 100
    print(f"- substep: _outlier_exclusion [{outlier_count}/{len(h)}({percentage_replaced:.2f}%)]")
    return h





