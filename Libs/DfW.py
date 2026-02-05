import os
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

import modules as md

class TqdmParallel(Parallel):
    def __init__(self, tqdm_bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_bar = tqdm_bar

    def print_progress(self):
        if self._original_iterator is not None:
            self.tqdm_bar.n = self.n_completed_tasks
            self.tqdm_bar.refresh()

def time_it(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        print(f"- substep: {func.__name__} [{time.time() - t0:.1f} s]")
        return out
    return wrapper

class Operator:
    _thread_env_set = False

    def __init__(self, Vid, extent, parameters):
        Vid = np.transpose(Vid, (1, 2, 0))

        self.params = parameters
        self.dx = self.params["dx"]
        self.dt = self.params["dt"]
        self.savedir = self.params["savedir"]

        self.I = self._lanczos_resem(Vid, extent)

        self.rows, self.cols, self.frames = self.I.shape
        self.x_coords = np.linspace(extent[0], extent[1], self.cols)
        self.y_coords = np.linspace(extent[3], extent[2], self.rows)

    @classmethod
    def run_once(cls, Vid, extent, parameters, gt=None, Res_integ=None):
        obj = cls(Vid, extent, parameters, gt=gt)
        obj.run(Res_integ=Res_integ)
        return obj

    @staticmethod
    def _set_single_thread_env():
        try:
            import mkl
            mkl.set_num_threads(1)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    @classmethod
    def _ensure_thread_env(cls):
        if cls._thread_env_set:
            return
        cls._set_single_thread_env()
        cls._thread_env_set = True

    @staticmethod
    def _to_shared(arr: np.ndarray) -> shared_memory.SharedMemory:
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, arr.dtype, buffer=shm.buf)[:] = arr
        return shm

    @staticmethod
    def _safe_unlink(shm: shared_memory.SharedMemory | None):
        if shm is None:
            return
        shm.close()
        shm.unlink()

    @staticmethod
    @contextmanager
    def _shared_pair(data: np.ndarray, mask: np.ndarray):
        shm_data = Operator._to_shared(data)
        shm_mask = Operator._to_shared(mask)
        try:
            yield shm_data, shm_mask
        finally:
            Operator._safe_unlink(shm_data)
            Operator._safe_unlink(shm_mask)

    @time_it
    def _lanczos_resem(self, Vid, extent):
        dx = (extent[1] - extent[0]) / Vid.shape[1]
        dy = (extent[3] - extent[2]) / Vid.shape[0]
        if abs(self.dx - dx) < 1e-3:
            return Vid
        else:
            self.dx = max(self.dx, min(dx, dy))
            return md.LanczosResem(Vid, dx, dy, self.dx)

    @time_it
    def _preprocessing(self):
        if self.params["flag_masking"]:
            self.I = md.FreeFormMasker().run(self.I, self.params)
        self._generate_sparse_mesh()

    def _generate_sparse_mesh(self):
        step = int(self.params["step"] / self.dx)

        x_coords_int, y_coords_int = np.meshgrid(
            self.x_coords[::step], self.y_coords[::step]
        )
        x_indices, y_indices = np.meshgrid(
            np.arange(0, self.cols, step), np.arange(0, self.rows, step)
        )

        intrgpoints = self.I[::step, ::step, 0]
        valid = (
            ~intrgpoints.mask
            if np.ma.isMaskedArray(intrgpoints)
            else np.ones_like(intrgpoints, dtype=bool)
        )

        self.interg_pts = np.column_stack((x_coords_int[valid], y_coords_int[valid]))
        self.indices = np.column_stack((x_indices[valid], y_indices[valid]))

    @time_it
    def _mode_decomposition(self, I_seg, seg_idx):
        I_seg = md.ImageEnhancement(I_seg, self.dt)
        DMs, T, cont = md.DMD(I_seg, self.params, seg_idx, self.dt).run()
        return DMs, T, cont, len(T)
    
    @time_it
    def _nonlinearity_correction(self, h_lin, T):
        H0, h0 = self.params["H0"], self.params["h0"]
        h_nonlin = []
        for loc, h_ini in enumerate(h_lin):
            h_nonlin.append(md.nonlinearity_correction(h_ini, T, H0, h0))
        return h_nonlin
        
    def _depth_estimator(self, seg_idx, nloc, T, DMs, cont):
        self._ensure_thread_env()

        DMs_data = np.ascontiguousarray(DMs.data)
        DMs_mask = np.ascontiguousarray(DMs.mask)

        indices = self.indices
        dx = self.dx
        win_coef = self.params["win_coef"]
        n_jobs = self.params["n_jobs"]

        with self._shared_pair(DMs_data, DMs_mask) as (shm_data, shm_mask):
            with tqdm(range(nloc), 
                      desc=r"- Interrogation",
                      miniters=max(1, nloc // 100), 
                      mininterval=1.0, 
                      leave=True) as it:
                results = TqdmParallel(it, n_jobs=n_jobs, backend="loky")(
                    delayed(md.depth_single_point)(
                        loc_idx,
                        indices,
                        shm_data.name,
                        shm_mask.name,
                        DMs_data.shape,
                        DMs_data.dtype.name,
                        T,
                        cont,
                        dx,
                        win_coef,
                    )
                    for loc_idx in it
                )

        hstack = np.array([res[0] for res in results])
        kstack = np.array([res[1] for res in results])
        
        if self.params["flag_nonlinearity"]:
            hstack = self._nonlinearity_correction(hstack, T)

        w = cont[None, :] * np.isfinite(hstack)
        num = np.nansum(hstack * w, axis=1)
        den = np.nansum(w, axis=1)
        h_est = np.divide(
            np.nansum(hstack * w, axis=1), np.nansum(w, axis=1),
            out=np.full_like(num, np.nan, dtype=float),
            where = den > 0
        )

        h_est = md.Excludingoutlier(self.params["Sth"], self.params["step"],
                                    indices, h_est)

        return h_est, kstack

    def run(self, Res_integ=None):
        self._preprocessing()

        nperseg = int(self.params["VidSeg"] / self.dt)
        nseg = int(self.frames / nperseg)
        nloc = len(self.indices)

        output = {
            "points_px": self.indices,
            "points_locs": self.interg_pts,
            "depth": [],
            "wavenumber": [],
            "frequency": [],
            "contribution": [],
            "DM": [],
        }

        for seg_idx in range(nseg):
            t0 = seg_idx * nperseg * self.dt
            t1 = (seg_idx + 1) * nperseg * self.dt
            print(
                f"  [Video segment {seg_idx + 1}/{nseg}] "
                f"{int(t0 // 60):02d}:{int(t0 % 60):02d} - "  
                f"{int(t1 // 60):02d}:{int(t1 % 60):02d}"
            )
            I_seg = self.I[:, :, seg_idx * nperseg:(seg_idx + 1) * nperseg]

            DMs, T, cont, _ = self._mode_decomposition(I_seg, seg_idx)
            hest, kstack_perseg = self._depth_estimator(seg_idx, nloc, T, DMs, cont)

            output["DM"].append(DMs)
            output["depth"].append(hest)
            output["wavenumber"].append(kstack_perseg)
            output["frequency"].append(1 / T)
            output["contribution"].append(cont)

        outpath = self.savedir / "results.npy"
        np.save(outpath, output, allow_pickle=True)
        self.__dict__.clear()

#%%
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
import matplotlib.tri as mtri
plt.rcParams.update({"font.family": "Arial", "figure.dpi": 500})

class Visualization:
    def __init__(self, Vid, extent, params):
        self.Vid = Vid
        self.savedir = Path(params["savedir"])
        self.case = params["case"]

        self.figsize = (4, 4)

        self.offset = np.array(
            [extent[0] - (extent[0] % 100), extent[2] - (extent[2] % 100)],
            dtype=float,
        )
        self.extent = [
            extent[0] - self.offset[0],
            extent[1] - self.offset[0],
            extent[2] - self.offset[1],
            extent[3] - self.offset[1],
        ]

        self.basemap = self._load_basemap()
        self.xy, self.est = self._load_results()

    def _load_basemap(self):
        path = (self.savedir / ".." / ".." / "basemap.png").resolve()
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return np.flipud(img)

        bm = self.Vid[:, :, 0].astype(np.float32)
        bm[bm == 0] = np.nan
        return bm

    def _load_results(self):
        d = np.load(self.savedir / "results.npy", allow_pickle=True).item()
        xy = d["points_locs"].copy()
        xy[:, 0] -= self.offset[0]
        xy[:, 1] -= self.offset[1]
        
        self.mask = np.load(self.savedir / "../../mask.npy")
        return xy, d["depth"]

    def _plot(self, z, filename, ran, interval, cmap="RdYlBu"):
        lv_f = np.linspace(ran[0], ran[1] + 1e-6, int((ran[1]-ran[0]/interval*2)))
        lv_l = np.arange(ran[0], ran[1] + 1e-6, interval)
        ticks = np.arange(lv_f.min(), lv_f.max(), 1)
    
        x, y = self.xy[:, 0], self.xy[:, 1]
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[m], y[m], np.clip(z[m], ran[0], ran[1])
        
        tri = mtri.Triangulation(x, y)
        xtri, ytri = x[tri.triangles], y[tri.triangles]
        
        area = np.abs(
            (xtri[:, 1] - xtri[:, 0]) * (ytri[:, 2] - ytri[:, 0]) -
            (xtri[:, 2] - xtri[:, 0]) * (ytri[:, 1] - ytri[:, 0])
        )
        
        edge_len = np.maximum.reduce([
            np.hypot(xtri[:, 0] - xtri[:, 1], ytri[:, 0] - ytri[:, 1]),
            np.hypot(xtri[:, 1] - xtri[:, 2], ytri[:, 1] - ytri[:, 2]),
            np.hypot(xtri[:, 2] - xtri[:, 0], ytri[:, 2] - ytri[:, 0]),
        ])
        
        mask = (area > np.percentile(area, 99)) | (
            edge_len > 2.5 * np.nanmedian(edge_len))    
        tri.set_mask(mask)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin="lower", extent=self.extent, cmap="bone")
    
        ct = ax.tricontourf(tri, z, levels=lv_f, cmap=cmap)
        cbar = fig.colorbar(ct, ax=ax, label=filename, shrink=0.7)
        cbar.set_ticks(ticks)
    
        ctl = ax.tricontour(tri, z, levels=lv_l, colors="black", linewidths=0.3)
        ax.clabel(ctl, inline=True, fmt="%.1f", fontsize=7)
    
        ax.plot(x, y, ".", color="white", markersize=0.2)
    
        ax.set(
            xlim=(self.extent[0], self.extent[1]),
            ylim=(self.extent[2], self.extent[3]),
            xlabel="x (m)",
            ylabel="y (m)",
        )
    
        ax.ticklabel_format(style="sci", scilimits=(-3, 3), axis="both")
        ax.yaxis.get_offset_text().set_position((0, 1.02))
        ax.xaxis.get_offset_text().set_position((1.02, 0))
    
        fig.tight_layout()
        fig.savefig(self.savedir / filename)
        plt.close(fig)

    def _gt_interpolation(self, gt):
        ny, nx = gt.shape
        xmin, xmax, ymin, ymax = self.extent
    
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
    
        interp = RegularGridInterpolator(
            (y, x),      # (row, col) = (y, x)
            gt,
            bounds_error=False,
            fill_value=np.nan
        )
    
        # self.xy: (N, 2) = [x, y]
        xq = self.xy[:, 0]
        yq = self.xy[:, 1]
    
        pts = np.column_stack((yq, xq))  # (N, 2)
        zq = interp(pts)                 # (N,)
    
        return zq
    
    @staticmethod
    def _error_metrics(err, zq):
        m = np.isfinite(err) & np.isfinite(zq)
        e, z = err[m], zq[m]
        rmse = np.sqrt(np.mean(e**2))
        m_mape = m & (zq != 0.0)
        mape = np.mean(np.abs(err[m_mape] / zq[m_mape])) * 100.0
        pbias = 100.0 * np.sum(e) / np.sum(z)
        return rmse, mape, pbias

    
    def _plot_agreement(self, zq, est_med, filename="Agreement"):
        m = np.isfinite(zq) & np.isfinite(est_med)
        x, y = zq[m], est_med[m]
        if x.size == 0:
            return
    
        rmse, mape, pbias = self._error_metrics(y - x, x)
    
        dens = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
        idx = np.argsort(dens)
        x, y, dens = x[idx], y[idx], dens[idx]
    
        vmin, vmax = min(x.min(), y.min()), max(x.max(), y.max())
    
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.scatter(x, y, c=dens, s=6, cmap="viridis_r", edgecolors="none")
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
    
        ax.set(
            xlabel="Measured depth (m)",
            ylabel="Estimated depth (m)",
            xlim=(vmin, vmax),
            ylim=(vmin, vmax),
            aspect="equal"
        )
    
        ax.text(
            0.05, 0.95,
            f"RMSE  = {rmse:.2f} m\nMAPE  = {mape:.1f} %\nPBIAS = {pbias:.1f} %",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
        )
    
        fig.tight_layout()
        fig.savefig(self.savedir / filename)
        plt.close(fig)
        
    def run(self, gt = None):
        print("[Plotting DfW estimates]")
        # # for i, z in enumerate(self.est, start=1):
        # #     self._plot(z, f"{self.case}_Seg-{i}")

        est_arr = np.array(self.est)
        est_med = np.nanmedian(est_arr, axis=0)
        est_std = np.nanstd(est_arr, axis=0) 
        self._plot(est_med, "EST-mean", ran = [0, 18], interval = 1)
        self._plot(est_std, "EST-std", ran = [-2, 2], interval = 0.5, cmap = "inferno")
        if gt is not None:
            zq = self._gt_interpolation(np.flipud(gt))
            self._plot(zq, "GT", ran = [0, 18], interval = 1)
            err = est_med  - zq
            self._plot(err, "DIFF", ran = [-2, 2], interval = 0.5, cmap = "bwr")
            self._plot_agreement(zq, est_med, filename="Agreement")





