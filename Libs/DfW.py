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
        print(f"- substep: {func.__name__} [{time.time() - t0:.2f} s]")
        return out
    return wrapper

class Operator:
    _thread_env_set = False

    def __init__(self, Vid, extent, parameters, gt=None):
        Vid = np.transpose(Vid, (1, 2, 0))

        self.params = parameters
        self.dx = self.params["dx"]
        self.dt = self.params["dt"]
        self.savedir = self.params["savedir"]

        self.I, self.gt = self._lanczos_resem(Vid, gt, extent)

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
    def _lanczos_resem(self, Vid, gt, extent):
        dx = (extent[1] - extent[0]) / Vid.shape[1]
        dy = (extent[3] - extent[2]) / Vid.shape[0]
        if abs(self.dx - dx) < 1e-3:
            return Vid, gt
        self.dx = max(self.dx, min(dx, dy))
        return md.LanczosResem(Vid, gt, dx, dy, self.dx)

    @time_it
    def _preprocessing(self):
        if self.params["flag_masking"]:
            self.I = md.FreeFormMasker().run(self.I, self.params)
        self._generate_sparse_mesh()
        if self.gt is not None:
            self.gt = self.gt[self.indices[:, 1], self.indices[:, 0]]

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
        
    def _depth_estimator_parallel(self, seg_idx, nloc, T, DMs, cont):
        self._ensure_thread_env()

        DMs_data = np.ascontiguousarray(DMs.data)
        DMs_mask = np.ascontiguousarray(DMs.mask)

        indices = self.indices
        dx = self.dx
        win_coef = self.params["win_coef"]
        n_jobs = self.params["n_jobs"]

        with self._shared_pair(DMs_data, DMs_mask) as (shm_data, shm_mask):
            with tqdm(range(nloc), desc=r"- Interrogation", mininterval=0.2, leave=True) as it:
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
        h_est = np.nansum(hstack * w, axis=1) / np.nansum(w, axis=1)

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
            "groundtruth": self.gt,
            "depth": [],
            "wavenumber": [],
            "frequency": [],
            "contribution": [],
            "DM": [],
        }

        for seg_idx in range(nseg):
            print(f"  [ Segment {seg_idx + 1}/{nseg} ]")
            I_seg = self.I[:, :, seg_idx * nperseg:(seg_idx + 1) * nperseg]

            DMs, T, cont, _ = self._mode_decomposition(I_seg, seg_idx)
            hest, kstack_perseg = self._depth_estimator_parallel(seg_idx, nloc, T, DMs, cont)

            output["DM"].append(DMs)
            output["depth"].append(hest)
            output["wavenumber"].append(kstack_perseg)
            output["frequency"].append(1 / T)
            output["contribution"].append(cont)

        outpath = self.savedir / "results.npy"
        np.save(outpath, output, allow_pickle=True)
        self.__dict__.clear()

#%%
plt.rcParams.update({"font.family": "Arial", "figure.dpi": 500})

class Visualization:
    def __init__(self, Vid, extent, params):
        self.Vid = Vid
        self.savedir = Path(params["savedir"])
        self.case = params["case"]

        self.figsize = (4, 4)
        self.hran = (0, 16)
        self.interval = 1.0

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
        return xy, d["depth"]

    def _levels(self, z):
        zmax = np.nanmax(z)
        if self.hran[1] is not None:
            lv_f = np.linspace(self.hran[0], self.hran[1] + 1e-3, 30)
            lv_l = np.arange(self.hran[0], self.hran[1] + 1e-3, self.interval)
        else:
            lv_f = np.linspace(0, np.ceil(zmax) + 1e-3, 30)
            lv_l = np.arange(0, np.ceil(zmax) + 1e-3, self.interval)

        ticks = np.arange(lv_f.min(), lv_f.max(), 1)
        return lv_f, lv_l, ticks, plt.cm.get_cmap("RdYlBu", 30)

    def _plot(self, z, filename):
        x, y = self.xy[:, 0], self.xy[:, 1]
        m = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z))
        x, y, z = x[m], y[m], z[m]
        if z.size == 0:
            return

        lv_f, lv_l, ticks, cmap = self._levels(z)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.basemap, origin="lower", extent=self.extent, cmap="bone")

        ct = ax.tricontourf(x, y, z, levels=lv_f, cmap=cmap)
        cbar = fig.colorbar(ct, ax=ax, label="Water depth (m)", shrink=0.7)
        cbar.set_ticks(ticks)

        ctl = ax.tricontour(x, y, z, levels=lv_l, colors="black", linewidths=0.3)
        ax.clabel(ctl, inline=True, fmt="%.1f", fontsize=7)

        ax.plot(self.xy[:, 0], self.xy[:, 1], ".", color="white", markersize=0.5)

        ax.set(
            xlim=(self.extent[0], self.extent[1]),
            ylim=(self.extent[2], self.extent[3]),
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        ax.ticklabel_format(style="sci", scilimits=(-3, 3), axis="both")
        ax.yaxis.get_offset_text().set_position((0, 1.02))
        ax.xaxis.get_offset_text().set_position((1.02, 0))

        fig.tight_layout()
        fig.savefig(self.savedir / filename)
        plt.close(fig)

    def run(self):
        print("[Plotting DfW estimates]")
        for i, z in enumerate(self.est, start=1):
            self._plot(z, f"{self.case}_Seg-{i}")

        est_arr = np.array(self.est)
        self._plot(np.nanmedian(est_arr, axis=0), "seg-mean")
        self._plot(np.nanstd(est_arr, axis=0), "seg-std")
        plt.close("all")





