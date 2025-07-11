<<<<<<< HEAD
""" Required Libarires """

import numpy as np
import cv2
import os
from scipy.linalg import svd
from skimage.transform import radon
import matplotlib.pyplot as plt

# import numpy.ma as ma
from datetime import datetime
from scipy.signal import welch, filtfilt, firwin, butter
from pydmd import OptDMD
      
"""
01. Preprocessings
"""
def LanczosResem(I, dx, dx_target):
    r = dx / dx_target
    ns = (int(I.shape[1]*r), int(I.shape[0]*r))
    I_rs = np.empty((ns[1], ns[0], I.shape[2]), dtype=np.float32)
    for i in range(I.shape[2]):
        frame = I[:, :, i]
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError(f"Invalid slice at index {i} with shape {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError(f"Slice {i} contains NaN or Inf")
        I_rs[:, :, i] = cv2.resize(frame, ns, interpolation=cv2.INTER_LANCZOS4)
        
    return I_rs

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
        return False

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

    def run(self, I, params):
        rows, cols, frames = I.shape
        firstframe = I[:, :, 0]
        mask4nan = (firstframe == 0) | np.isnan(firstframe)
        mask_path  = os.path.join(params['working_dir'], 
                                  'data', params['Site'],
                                  params['Case'], r'mask.npy')
        
        if os.path.isfile(mask_path):
            print(f'       Predefined mask exist: {mask_path}')
            mask = np.load(mask_path)
        else:
            if params['flag_Masking']:
                MeanImg = np.mean(I[:,:,::int(frames/100)], axis=2)
                MeanImg = MeanImg.astype(np.uint8)
                # masker = FreeFormMasker()
                mask4dry = self.select_free_form(MeanImg)
            else:
                mask4dry = np.zeros_like(firstframe, dtype=bool)
            mask = (mask4nan | mask4dry)
            np.save(mask_path, mask)
        
        try:
            mask = np.broadcast_to(mask[:, :, None], I.shape)
            return np.ma.masked_where(mask, I)
        except:
            raise ValueError(f"          Predefined mask shape {mask.shape} does not match given data {I.shape[:2]}")

def bandpass_filter(data, fs, lowcut=None, highcut=None, order=3):
    nyquist = 0.5 * fs
    if lowcut and highcut:
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    elif highcut:
        b, a = butter(order, highcut / nyquist, btype='low')
    elif lowcut:
        b, a = butter(order, lowcut / nyquist, btype='high')
    else:
        raise ValueError("Specify at least one of lowcut or highcut.")
    return filtfilt(b, a, data, axis = 2)

def fir_bandpass_filter(data, fs, lowcut=None, highcut=None, numtaps=101, window='hamming'):
    nyquist = 0.5 * fs

    if lowcut and highcut:
        # Bandpass filter
        cutoff = [lowcut / nyquist, highcut / nyquist]
        b = firwin(numtaps, cutoff, window=window, pass_zero=False)
    elif highcut:
        # Lowpass filter
        cutoff = highcut / nyquist
        b = firwin(numtaps, cutoff, window=window, pass_zero=True)
    elif lowcut:
        # Highpass filter
        cutoff = lowcut / nyquist
        b = firwin(numtaps, cutoff, window=window, pass_zero=False)
    else:
        raise ValueError("Specify at least one of lowcut or highcut.")

    return filtfilt(b, [1.0], data, axis=2, method='gust')

def apply_clahe(frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame)

def ImageEnhancement(I, dt):
    data = I.data if np.ma.isMaskedArray(I) else I
    data = np.nan_to_num(data, nan=0).astype(np.uint8, copy=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    data = np.stack([clahe.apply(frame) for frame in data.transpose(2, 0, 1)], axis=2)
    data = bandpass_filter(data, 1/dt, lowcut= 1/16, highcut = 1/3)
    I = np.ma.masked_array(data.astype(np.float32), mask=I.mask)
    I -= np.ma.mean(I, axis=2, keepdims=True)
    I = np.clip(I, np.quantile(I.compressed(), 0.01), np.quantile(I.compressed(), 0.99))
    I = (I - np.ma.min(I)) / (np.ma.max(I) - np.ma.min(I))
    return I

"""
02. Wave Analysis
"""
class DMD():
    def __init__(self, I, params, seg_idx):
        self.I = I
        self.rows, self.cols, self.frames = I.shape
        self.dt = params['dt']
        self.seg_idx = seg_idx + 1
        self.params = params
        self.r = params.get('modenum')
        
        if np.ma.isMaskedArray(I):
            self.I_valid = I.compressed().reshape(-1, self.frames)
            self.mask = np.ma.getmaskarray(I[:,:,0])
        else:
            self.I_valid = I.reshape(-1, self.frames)
            self.mask = np.zeros((self.rows, self.cols), dtype=bool)   
            
        self.savedir = os.path.join(params.get('savedir'), 'Spectra')
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
                    
        self.T, self.cont = None, None
        
    def peak_frequency(self, step = 20, nperseg=None, noverlap=None):
        I_sparse = self.I[::step, ::step, :]
        I_sparse = I_sparse.reshape(-1, I_sparse.shape[2])
        freq, psd = welch(I_sparse, fs=1 / self.dt, 
                          nperseg=I_sparse.shape[1], 
                          axis=1)
        return freq[np.argmax(np.mean(psd, axis=0))]
      
    def rank_truncation(self):
        _, S, _ = svd(self.I_valid, full_matrices=False)        
        energy = np.cumsum(S**2) / np.sum(S**2)
        return min(np.searchsorted(energy, 0.95), self.r)
    
    def reshaping(self, Phi):
        Phi_rshp = np.full((self.rows * self.cols, Phi.shape[1]), np.nan, dtype=complex)
        Phi_rshp[~self.mask.ravel(), :] = Phi
        return np.ma.masked_invalid(Phi_rshp.reshape(self.rows, self.cols, -1))
    
    def PlottingSpectrum(self, T, cont, freq_valid, fp):
        f, psd = welch(self.I_valid, fs=2, nperseg=self.frames, axis=-1)
        Q25, Q75, Median = np.quantile(psd, 0.25, axis=0), np.quantile(psd, 0.75, axis=0), np.median(psd, axis=0)
        
        signal_band = (f >= freq_valid[1]) & (f <= freq_valid[0])  # Hz
        
        fig, ax = plt.subplots(figsize = (6,4))
        ax.fill_between(f, Q25 / np.max(Median), Q75 / np.max(Median), color='lightgray', alpha=0.5)        
        plt.plot(f, Median/np.max(Median), color='black', linestyle = 'solid')
        plt.bar(1/T, cont, width=0.01, color = 'tab:red')
        ax.axvline(x=freq_valid[0], linestyle='--', color='k', linewidth = 0.5)
        ax.axvline(x=freq_valid[1], linestyle='--', color='k', linewidth = 0.5)
        ax.axvline(x=fp, linestyle='--', color='k', linewidth = 1.0)
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('P / Pmax [-]')
        plt.savefig(os.path.join(self.savedir, f'Specrum_Seg{self.seg_idx}'))
        plt.close('all')
                
    def dmd(self, r, fp, vis = None):
        freq_valid = np.array([max(fp * 1.5, 1/3), min(fp * 0.6, 1 / 10)])
        period_valid = 1 / freq_valid
        
        dmd = OptDMD(svd_rank = 2*r+2, opt=True)
        dmd.fit(self.I_valid)
 
        omega = np.log(dmd.eigs) / self.dt
        T     =  np.divide(2 * np.pi, np.where(omega.imag != 0, omega.imag, np.nan))  
        # freqs = np.abs(np.imag(omega)) / (2 * np.pi)
        phi = dmd.modes.real
        b = np.linalg.lstsq(phi, self.I_valid[:, 0], rcond=None)[0]
        cont = np.linalg.norm(phi * b[np.newaxis, :], axis=0)**2
                
        used = []
        valid_idx = []
        
        for j in np.argsort(-cont):
            T_j = T[j]
            if period_valid[0] <= T_j <= period_valid[1] and all(abs(T_j - u) > 1e-3 for u in used):
                used.append(T_j)
                valid_idx.append(j)
                if len(valid_idx) == r:
                    break

        self.Phi  =  self.reshaping(phi[:,valid_idx])
        self.T = T[valid_idx]
        self.cont = cont[valid_idx]/np.sum(cont[valid_idx])
        self.PlottingSpectrum(self.T, self.cont, freq_valid, fp)
                    
    def run(self):
        r  =  self.rank_truncation()
        fp =  self.peak_frequency()
        self.dmd(r, fp)
        return self.Phi, self.T, self.cont

def RadonTransform(Phi, dx, plotting = 'off'):
    rows, cols = Phi.shape
    # Phi = cv2.GaussianBlur(Phi, (5, 5), 0)
    # target_dx = 0.5
       
    if np.ma.is_masked(Phi):
        Phi = np.ma.filled(Phi, fill_value=0)

    # # Phi = cv2.GaussianBlur(Phi, (5, 5), 0)    
    # if dx/target_dx < 1.0:
    #     Phi       =  rescale(Phi, scale = dx/target_dx, mode='reflect')
    
    # height, width = target_dx * Phi.shape[0], target_dx * Phi.shape[1]
    # diagonal   = np.sqrt(height**2 + width**2)
    theta      =  np.linspace(0.0, 180.0, np.max(Phi.shape))
    sinogram   =  radon(Phi, theta = theta, circle=False)
    IPA        =  np.nansum(np.square(sinogram), axis=0) # Intensity per angle
    idx_max    =  np.argmax(IPA)
    theta_max  =  np.deg2rad(-theta[idx_max])
    
    # dx_proj    =  diagonal / sinogram.shape[0]
    dx_proj    =  dx
    Proj       =  sinogram[:,idx_max]
    
    n_pixels = np.abs(np.cos(theta_max)) + np.abs(np.sin(theta_max))
    n_pixels *= max(Phi.shape)
    Proj /= n_pixels
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if plotting == 'on':
        if current_time != datetime.now().strftime('%Y%m%d_%H%M%S'):
            # Create the 'sinogram' directory if it doesn't exist
            folder_name = r'B:\05_Codes\DfW_v1.0\data\BS02_C1\res\sinogram'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        
            # Create filename with current timestamp
            
            file_name = f'Sinogram_{current_time}.png'
            file_path = os.path.join(folder_name, file_name)
        
            # Visualization
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            
            # Input image (Phi)
            wave_dir_x = np.linspace(0, Phi.shape[1]*dx, 100)
            wave_dir_y = np.tan(-theta_max)*(wave_dir_x -Phi.shape[1]*dx/2) + Phi.shape[0]*dx/2
            axs[0].imshow(Phi, extent=(0, Phi.shape[1]*dx, 0, Phi.shape[0]*dx), origin = 'lower', cmap='gray')
            axs[0].plot(wave_dir_x, wave_dir_y, 'k--')
            axs[0].set_title('Input Image (Phi)')
            axs[0].set_xlabel('x (m)')
            axs[0].set_ylabel('y (m)')
            # axs[0].axis('off')  # Remove axes for better visualization
        
            # Sinogram
            axs[1].imshow(sinogram, extent=(0, 180, 0, sinogram.shape[0]*dx_proj), aspect='auto', cmap='jet')
            axs[1].axvline(np.rad2deg(theta_max), color = 'k')
            axs[1].set_title('Sinogram')
            axs[1].set_xlabel('Angle (degrees)')
            axs[1].set_ylabel('Projection distance (m)')
            
            # Summation along angle axis
            axs[2].plot(theta, IPA)
            axs[2].set_title('Summation along angles')
            axs[2].set_xlabel('Angle (degrees)')
            axs[2].set_ylabel('Summation')
        
            # Projection at maximum angle
            distance = np.arange(len(Proj)) * dx_proj  # Convert indices to meters
            axs[3].plot(distance, Proj)
            axs[3].set_title(f'Projection at max angle ({np.rad2deg(theta_max):.2f}°)')
            axs[3].set_xlabel('Projection distance')
            axs[3].set_ylabel('Normalized Intensity')
        
            plt.tight_layout()
            plt.savefig(file_path)  # Save the plot as PNG
            plt.close()
    
    return Proj, theta_max, dx_proj

def wavenumest(proj, dx_proj):
    n = len(proj)
    n_padded = 2 ** int(np.ceil(np.log2(n)))  # Use next power of 2 for efficient FFT
    fft = np.fft.fft(proj, n_padded)[:n_padded//2+1] / n
    freq = np.fft.fftfreq(n_padded, dx_proj)[:n_padded//2+1]
    magnitude = np.abs(fft)
    
    peak_idx = np.argmax(magnitude)
    
    if 1 <= peak_idx < len(magnitude) - 1:  # Ensure within bounds
        x_vals = freq[peak_idx - 1:peak_idx + 2]
        y_vals = magnitude[peak_idx - 1:peak_idx + 2]
        
        # Fit a quadratic: y = ax^2 + bx + c
        coeffs = np.polyfit(x_vals, y_vals, 2)
        a, b, _ = coeffs
        
        if a != 0:
            refined_peak_freq = -b / (2 * a)
        else:
            refined_peak_freq = freq[peak_idx]
    else:
        refined_peak_freq = freq[peak_idx]
    return 2 * np.pi * refined_peak_freq if refined_peak_freq > 0 else np.nan

#%%
"""
03. Postprocessings
"""
from scipy.spatial import KDTree
def Excludingoutlier(Sth, interg_pts, h, distance):
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
    print(f"         {outlier_count}/{len(h)}({percentage_replaced:.2f}%) of depth values were excluded")
    return h
=======
""" Required Libarires """

import numpy as np
import cv2
import os
from scipy.linalg import svd
from skimage.transform import radon
import matplotlib.pyplot as plt

# import numpy.ma as ma
from datetime import datetime
from scipy.signal import welch, filtfilt, firwin, butter
from pydmd import OptDMD
      
"""
01. Preprocessings
"""
def LanczosResem(I, dx, dx_target):
    r = dx / dx_target
    ns = (int(I.shape[1]*r), int(I.shape[0]*r))
    I_rs = np.empty((ns[1], ns[0], I.shape[2]), dtype=np.float32)
    for i in range(I.shape[2]):
        frame = I[:, :, i]
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError(f"Invalid slice at index {i} with shape {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError(f"Slice {i} contains NaN or Inf")
        I_rs[:, :, i] = cv2.resize(frame, ns, interpolation=cv2.INTER_LANCZOS4)
        
    return I_rs

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
        return False

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

    def run(self, I, params):
        rows, cols, frames = I.shape
        firstframe = I[:, :, 0]
        mask4nan = (firstframe == 0) | np.isnan(firstframe)
        mask_path  = os.path.join(params['working_dir'], 
                                  'data', params['Site'],
                                  params['Case'], r'mask.npy')
        
        if os.path.isfile(mask_path):
            print(f'       Predefined mask exist: {mask_path}')
            mask = np.load(mask_path)
        else:
            if params['flag_Masking']:
                MeanImg = np.mean(I[:,:,::int(frames/100)], axis=2)
                MeanImg = MeanImg.astype(np.uint8)
                # masker = FreeFormMasker()
                mask4dry = self.select_free_form(MeanImg)
            else:
                mask4dry = np.zeros_like(firstframe, dtype=bool)
            mask = (mask4nan | mask4dry)
            np.save(mask_path, mask)
        
        try:
            mask = np.broadcast_to(mask[:, :, None], I.shape)
            return np.ma.masked_where(mask, I)
        except:
            raise ValueError(f"          Predefined mask shape {mask.shape} does not match given data {I.shape[:2]}")

def bandpass_filter(data, fs, lowcut=None, highcut=None, order=3):
    nyquist = 0.5 * fs
    if lowcut and highcut:
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    elif highcut:
        b, a = butter(order, highcut / nyquist, btype='low')
    elif lowcut:
        b, a = butter(order, lowcut / nyquist, btype='high')
    else:
        raise ValueError("Specify at least one of lowcut or highcut.")
    return filtfilt(b, a, data, axis = 2)

def fir_bandpass_filter(data, fs, lowcut=None, highcut=None, numtaps=101, window='hamming'):
    nyquist = 0.5 * fs

    if lowcut and highcut:
        # Bandpass filter
        cutoff = [lowcut / nyquist, highcut / nyquist]
        b = firwin(numtaps, cutoff, window=window, pass_zero=False)
    elif highcut:
        # Lowpass filter
        cutoff = highcut / nyquist
        b = firwin(numtaps, cutoff, window=window, pass_zero=True)
    elif lowcut:
        # Highpass filter
        cutoff = lowcut / nyquist
        b = firwin(numtaps, cutoff, window=window, pass_zero=False)
    else:
        raise ValueError("Specify at least one of lowcut or highcut.")

    return filtfilt(b, [1.0], data, axis=2, method='gust')

def apply_clahe(frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame)

def ImageEnhancement(I, dt):
    data = I.data if np.ma.isMaskedArray(I) else I
    data = np.nan_to_num(data, nan=0).astype(np.uint8, copy=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    data = np.stack([clahe.apply(frame) for frame in data.transpose(2, 0, 1)], axis=2)
    data = bandpass_filter(data, 1/dt, lowcut= 1/16, highcut = 1/3)
    I = np.ma.masked_array(data.astype(np.float32), mask=I.mask)
    I -= np.ma.mean(I, axis=2, keepdims=True)
    I = np.clip(I, np.quantile(I.compressed(), 0.01), np.quantile(I.compressed(), 0.99))
    I = (I - np.ma.min(I)) / (np.ma.max(I) - np.ma.min(I))
    return I

"""
02. Wave Analysis
"""
class DMD():
    def __init__(self, I, params, seg_idx):
        self.I = I
        self.rows, self.cols, self.frames = I.shape
        self.dt = params['dt']
        self.seg_idx = seg_idx + 1
        self.params = params
        self.r = params.get('modenum')
        
        if np.ma.isMaskedArray(I):
            self.I_valid = I.compressed().reshape(-1, self.frames)
            self.mask = np.ma.getmaskarray(I[:,:,0])
        else:
            self.I_valid = I.reshape(-1, self.frames)
            self.mask = np.zeros((self.rows, self.cols), dtype=bool)   
            
        self.savedir = os.path.join(params.get('savedir'), 'Spectra')
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
                    
        self.T, self.cont = None, None
        
    def peak_frequency(self, step = 20, nperseg=None, noverlap=None):
        I_sparse = self.I[::step, ::step, :]
        I_sparse = I_sparse.reshape(-1, I_sparse.shape[2])
        freq, psd = welch(I_sparse, fs=1 / self.dt, 
                          nperseg=I_sparse.shape[1], 
                          axis=1)
        return freq[np.argmax(np.mean(psd, axis=0))]
      
    def rank_truncation(self):
        _, S, _ = svd(self.I_valid, full_matrices=False)        
        energy = np.cumsum(S**2) / np.sum(S**2)
        return min(np.searchsorted(energy, 0.95), self.r)
    
    def reshaping(self, Phi):
        Phi_rshp = np.full((self.rows * self.cols, Phi.shape[1]), np.nan, dtype=complex)
        Phi_rshp[~self.mask.ravel(), :] = Phi
        return np.ma.masked_invalid(Phi_rshp.reshape(self.rows, self.cols, -1))
    
    def PlottingSpectrum(self, T, cont, freq_valid, fp):
        f, psd = welch(self.I_valid, fs=2, nperseg=self.frames, axis=-1)
        Q25, Q75, Median = np.quantile(psd, 0.25, axis=0), np.quantile(psd, 0.75, axis=0), np.median(psd, axis=0)
        
        signal_band = (f >= freq_valid[1]) & (f <= freq_valid[0])  # Hz
        
        fig, ax = plt.subplots(figsize = (6,4))
        ax.fill_between(f, Q25 / np.max(Median), Q75 / np.max(Median), color='lightgray', alpha=0.5)        
        plt.plot(f, Median/np.max(Median), color='black', linestyle = 'solid')
        plt.bar(1/T, cont, width=0.01, color = 'tab:red')
        ax.axvline(x=freq_valid[0], linestyle='--', color='k', linewidth = 0.5)
        ax.axvline(x=freq_valid[1], linestyle='--', color='k', linewidth = 0.5)
        ax.axvline(x=fp, linestyle='--', color='k', linewidth = 1.0)
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('P / Pmax [-]')
        plt.savefig(os.path.join(self.savedir, f'Specrum_Seg{self.seg_idx}'))
        plt.close('all')
                
    def dmd(self, r, fp, vis = None):
        freq_valid = np.array([max(fp * 1.5, 1/3), min(fp * 0.6, 1 / 10)])
        period_valid = 1 / freq_valid
        
        dmd = OptDMD(svd_rank = 2*r+2, opt=True)
        dmd.fit(self.I_valid)
 
        omega = np.log(dmd.eigs) / self.dt
        T     =  np.divide(2 * np.pi, np.where(omega.imag != 0, omega.imag, np.nan))  
        # freqs = np.abs(np.imag(omega)) / (2 * np.pi)
        phi = dmd.modes.real
        b = np.linalg.lstsq(phi, self.I_valid[:, 0], rcond=None)[0]
        cont = np.linalg.norm(phi * b[np.newaxis, :], axis=0)**2
                
        used = []
        valid_idx = []
        
        for j in np.argsort(-cont):
            T_j = T[j]
            if period_valid[0] <= T_j <= period_valid[1] and all(abs(T_j - u) > 1e-3 for u in used):
                used.append(T_j)
                valid_idx.append(j)
                if len(valid_idx) == r:
                    break

        self.Phi  =  self.reshaping(phi[:,valid_idx])
        self.T = T[valid_idx]
        self.cont = cont[valid_idx]/np.sum(cont[valid_idx])
        self.PlottingSpectrum(self.T, self.cont, freq_valid, fp)
                    
    def run(self):
        r  =  self.rank_truncation()
        fp =  self.peak_frequency()
        self.dmd(r, fp)
        return self.Phi, self.T, self.cont

def RadonTransform(Phi, dx, plotting = 'off'):
    rows, cols = Phi.shape
    # Phi = cv2.GaussianBlur(Phi, (5, 5), 0)
    # target_dx = 0.5
       
    if np.ma.is_masked(Phi):
        Phi = np.ma.filled(Phi, fill_value=0)

    # # Phi = cv2.GaussianBlur(Phi, (5, 5), 0)    
    # if dx/target_dx < 1.0:
    #     Phi       =  rescale(Phi, scale = dx/target_dx, mode='reflect')
    
    # height, width = target_dx * Phi.shape[0], target_dx * Phi.shape[1]
    # diagonal   = np.sqrt(height**2 + width**2)
    theta      =  np.linspace(0.0, 180.0, np.max(Phi.shape))
    sinogram   =  radon(Phi, theta = theta, circle=False)
    IPA        =  np.nansum(np.square(sinogram), axis=0) # Intensity per angle
    idx_max    =  np.argmax(IPA)
    theta_max  =  np.deg2rad(-theta[idx_max])
    
    # dx_proj    =  diagonal / sinogram.shape[0]
    dx_proj    =  dx
    Proj       =  sinogram[:,idx_max]
    
    n_pixels = np.abs(np.cos(theta_max)) + np.abs(np.sin(theta_max))
    n_pixels *= max(Phi.shape)
    Proj /= n_pixels
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if plotting == 'on':
        if current_time != datetime.now().strftime('%Y%m%d_%H%M%S'):
            # Create the 'sinogram' directory if it doesn't exist
            folder_name = r'B:\05_Codes\DfW_v1.0\data\BS02_C1\res\sinogram'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        
            # Create filename with current timestamp
            
            file_name = f'Sinogram_{current_time}.png'
            file_path = os.path.join(folder_name, file_name)
        
            # Visualization
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            
            # Input image (Phi)
            wave_dir_x = np.linspace(0, Phi.shape[1]*dx, 100)
            wave_dir_y = np.tan(-theta_max)*(wave_dir_x -Phi.shape[1]*dx/2) + Phi.shape[0]*dx/2
            axs[0].imshow(Phi, extent=(0, Phi.shape[1]*dx, 0, Phi.shape[0]*dx), origin = 'lower', cmap='gray')
            axs[0].plot(wave_dir_x, wave_dir_y, 'k--')
            axs[0].set_title('Input Image (Phi)')
            axs[0].set_xlabel('x (m)')
            axs[0].set_ylabel('y (m)')
            # axs[0].axis('off')  # Remove axes for better visualization
        
            # Sinogram
            axs[1].imshow(sinogram, extent=(0, 180, 0, sinogram.shape[0]*dx_proj), aspect='auto', cmap='jet')
            axs[1].axvline(np.rad2deg(theta_max), color = 'k')
            axs[1].set_title('Sinogram')
            axs[1].set_xlabel('Angle (degrees)')
            axs[1].set_ylabel('Projection distance (m)')
            
            # Summation along angle axis
            axs[2].plot(theta, IPA)
            axs[2].set_title('Summation along angles')
            axs[2].set_xlabel('Angle (degrees)')
            axs[2].set_ylabel('Summation')
        
            # Projection at maximum angle
            distance = np.arange(len(Proj)) * dx_proj  # Convert indices to meters
            axs[3].plot(distance, Proj)
            axs[3].set_title(f'Projection at max angle ({np.rad2deg(theta_max):.2f}°)')
            axs[3].set_xlabel('Projection distance')
            axs[3].set_ylabel('Normalized Intensity')
        
            plt.tight_layout()
            plt.savefig(file_path)  # Save the plot as PNG
            plt.close()
    
    return Proj, theta_max, dx_proj

def wavenumest(proj, dx_proj):
    n = len(proj)
    n_padded = 2 ** int(np.ceil(np.log2(n)))  # Use next power of 2 for efficient FFT
    fft = np.fft.fft(proj, n_padded)[:n_padded//2+1] / n
    freq = np.fft.fftfreq(n_padded, dx_proj)[:n_padded//2+1]
    magnitude = np.abs(fft)
    
    peak_idx = np.argmax(magnitude)
    
    if 1 <= peak_idx < len(magnitude) - 1:  # Ensure within bounds
        x_vals = freq[peak_idx - 1:peak_idx + 2]
        y_vals = magnitude[peak_idx - 1:peak_idx + 2]
        
        # Fit a quadratic: y = ax^2 + bx + c
        coeffs = np.polyfit(x_vals, y_vals, 2)
        a, b, _ = coeffs
        
        if a != 0:
            refined_peak_freq = -b / (2 * a)
        else:
            refined_peak_freq = freq[peak_idx]
    else:
        refined_peak_freq = freq[peak_idx]
    return 2 * np.pi * refined_peak_freq if refined_peak_freq > 0 else np.nan

#%%
"""
03. Postprocessings
"""
from scipy.spatial import KDTree
def Excludingoutlier(Sth, interg_pts, h, distance):
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
    print(f"         {outlier_count}/{len(h)}({percentage_replaced:.2f}%) of depth values were excluded")
    return h
>>>>>>> 9b2a775 (Initial commit with full project and Git LFS tracking)
