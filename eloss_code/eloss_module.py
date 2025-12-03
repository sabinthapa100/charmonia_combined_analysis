# -*- coding: utf-8 -*-
from __future__ import annotations
import re, math, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np, pandas as pd

# ----------------------- logging -----------------------
def make_logger(name: str = "eloss", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    else:
        logger.setLevel(level)
        for h in logger.handlers:
            h.setLevel(level)
    return logger
log = make_logger()

# ----------------------- IO helpers -----------------------
def _read_table_auto(path: Path, logger: logging.Logger = log) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    tries = [
        dict(sep="\t", header=None, comment="#"),
        dict(sep="\t", header=0, comment="#"),
        dict(sep=",", header=None, comment="#"),
        dict(sep=",", header=0, comment="#"),
        dict(delim_whitespace=True, header=None, comment="#"),
        dict(delim_whitespace=True, header=0, comment="#"),
    ]
    last = None
    for kw in tries:
        try:
            df = pd.read_csv(path, **kw)
            df = df.dropna(axis=1, how="all")
            if df.shape[1] < 3:
                continue
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if df.iloc[:, :3].isna().any().any():
                continue
            return df
        except Exception as e:
            last = e
    if last:
        logger.error(f"Failed to parse {path}: {last}")
    raise RuntimeError(f"Could not parse table: {path}")

def _unique_sorted(a: Iterable[float]) -> np.ndarray:
    A = np.array(list(a), dtype=float)
    A = np.unique(A[~np.isnan(A)])
    A.sort()
    return A

# ----------------------- grids -----------------------
class BilinearRectGrid:
    def __init__(self, x: np.ndarray, y: np.ndarray, Z: np.ndarray):
        if Z.shape != (x.size, y.size):
            raise ValueError("Z shape must be (len(x), len(y))")
        if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
            raise ValueError("Grid coords must be strictly increasing.")
        self.x, self.y, self.Z = x, y, Z

    def _idx(self, arr, v):
        i = np.searchsorted(arr, v) - 1
        return 0 if i < 0 else (arr.size - 2 if i >= arr.size - 1 else int(i))

    def __call__(self, xq: float, yq: float) -> float:
        i = self._idx(self.x, float(xq)); j = self._idx(self.y, float(yq))
        x0,x1 = self.x[i], self.x[i+1]; y0,y1 = self.y[j], self.y[j+1]
        dx = x1 - x0 if x1 != x0 else 1e-12
        dy = y1 - y0 if y1 != y0 else 1e-12
        tx = (min(max(xq, x0), x1) - x0) / dx
        ty = (min(max(yq, y0), y1) - y0) / dy
        Z = self.Z
        return float(Z[i,j]*(1-tx)*(1-ty) + Z[i+1,j]*tx*(1-ty) + Z[i,j+1]*(1-tx)*ty + Z[i+1,j+1]*tx*ty)

@dataclass
class Grid2D:
    y_vals: np.ndarray
    pt_vals: np.ndarray
    Z: np.ndarray
    def bilinear(self) -> BilinearRectGrid:
        return BilinearRectGrid(self.y_vals, self.pt_vals, self.Z)

def _build_rect_grid(df: pd.DataFrame, value_col_index: int) -> Grid2D:
    y = df.iloc[:,0].to_numpy(float)
    pt = df.iloc[:,1].to_numpy(float)
    val = df.iloc[:,value_col_index].to_numpy(float)
    ys = _unique_sorted(y); pts = _unique_sorted(pt)
    Z = np.full((ys.size, pts.size), np.nan, float)
    ymap = {v:i for i,v in enumerate(ys)}; pmap = {v:j for j,v in enumerate(pts)}
    for yi,pi,vi in zip(y,pt,val):
        if not (np.isnan(yi) or np.isnan(pi) or np.isnan(vi)):
            Z[ymap[yi], pmap[pi]] = vi
    if np.isnan(Z).any():
        miss = np.isnan(Z).mean()*100.0
        raise ValueError(f"Input does not form a complete rect grid (≈{miss:.1f}% missing).")
    return Grid2D(ys, pts, Z)

# ----------------------- data containers -----------------------
@dataclass
class CentBin:
    tag: str
    pA_log: Grid2D
    pp_log: Grid2D
    rpa: Grid2D
    rpa_err: Optional[Grid2D] = None   # optional per-point sigma (single run)
    @property
    def y_range(self): return float(self.rpa.y_vals.min()), float(self.rpa.y_vals.max())
    @property
    def pt_range(self): return float(self.rpa.pt_vals.min()), float(self.rpa.pt_vals.max())
    def pA(self, y, pt): return math.exp(self.pA_log.bilinear()(y, pt))
    def pp(self, y, pt): return math.exp(self.pp_log.bilinear()(y, pt))
    def RpA(self, y, pt): return float(self.rpa.bilinear()(y, pt))
    def RpAerr(self, y, pt): return 0.0 if self.rpa_err is None else float(self.rpa_err.bilinear()(y, pt))

# ==============================================================
# =============== Single eLoss run (no band) ===================
# ==============================================================
class ELossRun:
    """Reads one eLoss run (one parameter set)."""
    def __init__(self, base_dir: Path, particle: Optional[str] = None,
                 logger: logging.Logger = log, quad_n_y: int = 32, quad_n_pt: int = 64):
        self.base_dir = Path(base_dir)
        self.logger = logger
        self.particle = particle
        self.energy = None
        self.cent_bins: Dict[str, CentBin] = {}
        self.quad_n_y = int(quad_n_y)
        self.quad_n_pt = int(quad_n_pt)
        self._discover_and_load()

    # recursive discovery (works for top-level coarse + deep fine bins)
    def _discover_and_load(self):
        patterns = ["**/cent_*_*_*.*", "**/cent_*_*.*"]
        files: List[Path] = []
        for pat in patterns:
            files.extend(self.base_dir.rglob(pat))
        files = [f for f in files if f.is_file()]
        if not files:
            raise RuntimeError(f"No files matching 'cent_*' under {self.base_dir}")

        # infer particle if absent (from filenames)
        if self.particle is None:
            for f in files:
                m = re.search(r"cent_[0-9]+-[0-9]+_([A-Za-z0-9]+)_", f.name)
                if m:
                    self.particle = m.group(1); break
            if self.particle is None:
                self.particle = "Unknown"

        # infer energy (folder part with 'TeV')
        for part in self.base_dir.parts:
            if "TeV" in part:
                self.energy = part

        # group paths by centrality tag
        groups: Dict[str, Dict[str, Path]] = {}
        for f in files:
            m = re.search(r"cent_([0-9]+-[0-9]+)", f.name)
            if not m: continue
            tag = m.group(1)
            b = groups.setdefault(tag, {})
            name = f.name
            if "pA-cross-section" in name:   b["pA"] = f
            elif "pp-cross-section" in name: b["pp"] = f
            elif "_RpA" in name:             b["RpA"] = f

        def _cent_key(t: str): L,R = t.split("-"); return (int(L), int(R))
        for tag, bundle in sorted(groups.items(), key=lambda kv: _cent_key(kv[0])):
            miss = [k for k in ("pA", "pp", "RpA") if k not in bundle]
            if miss:
                self.logger.warning(f"[{tag}] skip missing {miss}")
                continue
            try:
                self.cent_bins[tag] = self._load_cent_bin(tag, bundle["pA"], bundle["pp"], bundle["RpA"])
                self.logger.info(f"Loaded {tag}")
            except Exception as e:
                self.logger.error(f"Failed to load {tag}: {e}")

        if not self.cent_bins:
            raise RuntimeError("Found no complete centrality triplets.")

    @property
    def available_cent_tags(self) -> List[str]:
        return sorted(self.cent_bins.keys(), key=lambda t: (int(t.split('-')[0]), int(t.split('-')[1])))

    @property
    def centrality_edges(self) -> List[float]:
        edges = set()
        for tag in self.cent_bins.keys():
            L, R = tag.split("-")
            edges.add(float(L)); edges.add(float(R))
        return sorted(edges)

    def _load_cent_bin(self, tag, pA_path, pp_path, rpa_path) -> CentBin:
        df_pA = _read_table_auto(Path(pA_path), logger=self.logger)
        df_pp = _read_table_auto(Path(pp_path), logger=self.logger)
        df_rp = _read_table_auto(Path(rpa_path), logger=self.logger)
        # logs for cross-sections, linear for RpA
        pA_log = _build_rect_grid(df_pA.iloc[:, :4], 2); pA_log.Z = np.log(np.clip(pA_log.Z, 1e-300, np.inf))
        pp_log = _build_rect_grid(df_pp.iloc[:, :4], 2); pp_log.Z = np.log(np.clip(pp_log.Z, 1e-300, np.inf))
        rpa    = _build_rect_grid(df_rp.iloc[:, :4], 2)
        rpa_err = None
        if df_rp.shape[1] >= 4:
            rpa_err = _build_rect_grid(df_rp.iloc[:, :4], 3)
        return CentBin(tag, pA_log, pp_log, rpa, rpa_err)

    # ------- Gauss–Legendre quadrature -------
    @staticmethod
    def _gauss_legendre_nodes(a,b,n):
        x,w = np.polynomial.legendre.leggauss(int(n)); t=0.5*(b-a); m=0.5*(b+a)
        return (m+t*x), (t*w)

    def _weight_pt(self, cent: CentBin, pt: float) -> float:
        # f(pt) ∝ pA(y=0, pt) * pt  (clamped)
        y0 = 0.0
        y_min,y_max = cent.y_range
        yq = float(np.clip(y0, y_min, y_max))
        return max(1e-300, cent.pA(yq, float(pt)) * float(pt))

    # ------- means (single run) -------
    def mean_rpa_over_y(self, cent_tag, y_range, pt_range):
        cent=self.cent_bins[cent_tag]
        y1,y2 = max(min(y_range),cent.y_range[0]), min(max(y_range),cent.y_range[1])
        p1,p2 = max(min(pt_range),cent.pt_range[0]), min(max(pt_range),cent.pt_range[1])
        pts,wpt = self._gauss_legendre_nodes(p1,p2,self.quad_n_pt)
        denom = float(np.sum([self._weight_pt(cent,p)*wp for p,wp in zip(pts,wpt)]))
        def g(y): 
            return float(np.sum([cent.RpA(y,p)*self._weight_pt(cent,p)*wp for p,wp in zip(pts,wpt)])/denom)
        ys,wys = self._gauss_legendre_nodes(y1,y2,self.quad_n_y)
        return float(np.sum([g(y)*wy for y,wy in zip(ys,wys)])/(y2-y1))

    def mean_rpa_over_pt(self, cent_tag, y_range, pt_range):
        cent=self.cent_bins[cent_tag]
        y1,y2 = max(min(y_range),cent.y_range[0]), min(max(y_range),cent.y_range[1])
        p1,p2 = max(min(pt_range),cent.pt_range[0]), min(max(pt_range),cent.pt_range[1])
        ys,wys = self._gauss_legendre_nodes(y1,y2,self.quad_n_y)
        def rpy(pt): return float(np.sum([cent.RpA(y,pt)*wy for y,wy in zip(ys,wys)])/(y2-y1))
        pts,wpt = self._gauss_legendre_nodes(p1,p2,self.quad_n_pt)
        num   = float(np.sum([rpy(p)*self._weight_pt(cent,p)*wp for p,wp in zip(pts,wpt)]))
        denom = float(np.sum([self._weight_pt(cent,p)*wp for p,wp in zip(pts,wpt)]))
        return num/denom

    def mean_rpa_over_y_and_pt(self, cent_tag, y_range, pt_range):
        cent=self.cent_bins[cent_tag]
        y1,y2 = max(min(y_range),cent.y_range[0]), min(max(y_range),cent.y_range[1])
        p1,p2 = max(min(pt_range),cent.pt_range[0]), min(max(pt_range),cent.pt_range[1])
        pts,wpt = self._gauss_legendre_nodes(p1,p2,self.quad_n_pt)
        ys,wys  = self._gauss_legendre_nodes(y1,y2,self.quad_n_y)
        denom = float(np.sum([self._weight_pt(cent,p)*wp for p,wp in zip(pts,wpt)])*(y2-y1))
        acc = 0.0
        for p,wp in zip(pts,wpt):
            for y,wy in zip(ys,wys):
                acc += cent.RpA(y,p)*self._weight_pt(cent,p)*wp*wy
        return acc/denom

    def mean_rpa_err_over_y_and_pt(self, cent_tag, y_range, pt_range):
        cent=self.cent_bins[cent_tag]
        if cent.rpa_err is None:
            return 0.0
        y1,y2 = max(min(y_range),cent.y_range[0]), min(max(y_range),cent.y_range[1])
        p1,p2 = max(min(pt_range),cent.pt_range[0]), min(max(pt_range),cent.pt_range[1])
        pts,wpt = self._gauss_legendre_nodes(p1,p2,self.quad_n_pt)
        ys,wys  = self._gauss_legendre_nodes(y1,y2,self.quad_n_y)
        denom = float(np.sum([self._weight_pt(cent,p)*wp for p,wp in zip(pts,wpt)])*(y2-y1))
        acc = 0.0
        for p,wp in zip(pts,wpt):
            for y,wy in zip(ys,wys):
                e = cent.RpAerr(y,p)
                acc += (e*e)*self._weight_pt(cent,p)*wp*wy
        return math.sqrt(acc/denom)

    # ------- binned curves (single run) -------
    def rpa_vs_centrality(self, y_range=(-5.0,5.0), pt_range=(0.1,20.0)):
        rows=[]
        for tag in self.available_cent_tags:
            L,R = [float(x) for x in tag.split("-")]
            mid, dx = 0.5*(L+R), 0.5*(R-L)
            try:
                val = self.mean_rpa_over_y_and_pt(tag,y_range,pt_range)
                err = self.mean_rpa_err_over_y_and_pt(tag,y_range,pt_range)
                rows.append(dict(cent_tag=tag, cent_mid=mid, xerr=dx,
                                 cent_left=L, cent_right=R, RpA=val, dRpA=err))
            except Exception as e:
                self.logger.error(f"Average failed for {tag}: {e}")
        return pd.DataFrame(rows).sort_values("cent_mid").reset_index(drop=True)

    def rpa_vs_y(self, cent_tag, y_edges, pt_range=(0.1,20.0)):
        y_edges = np.asarray(list(y_edges), float); rows=[]
        for yl,yr in zip(y_edges[:-1], y_edges[1:]):
            try:
                rows.append(dict(y_left=yl, y_right=yr, y_mid=0.5*(yl+yr),
                                 RpA=self.mean_rpa_over_y(cent_tag, (yl,yr), pt_range)))
            except Exception as e:
                self.logger.error(f"rpa_vs_y {cent_tag} [{yl},{yr}] failed: {e}")
        return pd.DataFrame(rows)

    def rpa_vs_pt(self, cent_tag, y_range, pt_edges):
        pt_edges = np.asarray(list(pt_edges), float); rows=[]
        for pl,pr in zip(pt_edges[:-1], pt_edges[1:]):
            try:
                rows.append(dict(pt_left=pl, pt_right=pr, pt_mid=0.5*(pl+pr),
                                 RpA=self.mean_rpa_over_pt(cent_tag, y_range, (pl,pr))))
            except Exception as e:
                self.logger.error(f"rpa_vs_pt {cent_tag} [{pl},{pr}] failed: {e}")
        return pd.DataFrame(rows)

# ==============================================================
# =============== Ensemble (bands from many runs) ==============
# ==============================================================
class ELossEnsemble:
    """Stack of ELossRun with envelope bands (min..max across runs).
       Central = mean across runs.
       Half-width (for combiner) = 0.5*(max - min).
    """
    def __init__(self, run_roots: List[Path], *, particle=None, logger=None,
                 quad_n_y: int = 32, quad_n_pt: int = 64):
        self.logger = logger or make_logger()
        self.runs: List[ELossRun] = [
            ELossRun(Path(rr), particle=particle, logger=self.logger,
                     quad_n_y=quad_n_y, quad_n_pt=quad_n_pt)
            for rr in run_roots
        ]
        if not self.runs:
            raise RuntimeError("ELossEnsemble: no runs discovered.")
        self.particle = self.runs[0].particle
        self.energy   = self.runs[0].energy
        common = set(self.runs[0].cent_bins.keys())
        for r in self.runs[1:]:
            common &= set(r.cent_bins.keys())
        if not common:
            raise RuntimeError("ELossEnsemble: no common centrality tags across runs.")
        self._cent_tags = sorted(common, key=lambda t: (int(t.split('-')[0]), int(t.split('-')[1])))

    # light compatibility surface for plotting helpers
    @property
    def cent_bins(self):  # mapping-like (not used for numbers)
        return {t: self.runs[0].cent_bins[t] for t in self._cent_tags}
    @property
    def available_cent_tags(self) -> List[str]:
        return list(self._cent_tags)
    @property
    def centrality_edges(self) -> List[float]:
        edges = set()
        for tag in self._cent_tags:
            L,R = tag.split("-"); edges.add(float(L)); edges.add(float(R))
        return sorted(edges)

    # ---------- combine numerics ----------
    def _stack_vs_cent(self, y_range, pt_range) -> pd.DataFrame:
        dfs = [r.rpa_vs_centrality(y_range=y_range, pt_range=pt_range) for r in self.runs]
        df0 = dfs[0][["cent_tag","cent_mid","xerr","cent_left","cent_right"]].copy()
        stacked = np.column_stack([df["RpA"].to_numpy(float) for df in dfs])
        df0["RpA"] = stacked.mean(axis=1)
        df0["lo"]  = stacked.min(axis=1)
        df0["hi"]  = stacked.max(axis=1)
        df0["dRpA"] = 0.5*(df0["hi"] - df0["lo"])  # for errorbar fallback
        return df0

    def rpa_vs_centrality(self, y_range=(-5.0,5.0), pt_range=(0.1,20.0)) -> pd.DataFrame:
        return self._stack_vs_cent(y_range, pt_range)

    def rpa_vs_y(self, cent_tag, y_edges, pt_range=(0.1,20.0)) -> pd.DataFrame:
        dfs = [r.rpa_vs_y(cent_tag, y_edges=y_edges, pt_range=pt_range) for r in self.runs]
        df0 = dfs[0][["y_left","y_right","y_mid"]].copy()
        stacked = np.column_stack([df["RpA"].to_numpy(float) for df in dfs])
        df0["RpA"] = stacked.mean(axis=1)
        df0["lo"]  = stacked.min(axis=1)
        df0["hi"]  = stacked.max(axis=1)
        return df0

    def rpa_vs_pt(self, cent_tag, y_range, pt_edges) -> pd.DataFrame:
        dfs = [r.rpa_vs_pt(cent_tag, y_range=y_range, pt_edges=pt_edges) for r in self.runs]
        df0 = dfs[0][["pt_left","pt_right","pt_mid"]].copy()
        stacked = np.column_stack([df["RpA"].to_numpy(float) for df in dfs])
        df0["RpA"] = stacked.mean(axis=1)
        df0["lo"]  = stacked.min(axis=1)
        df0["hi"]  = stacked.max(axis=1)
        return df0

    # ----- API for Combiner (central + symmetric half-width) -----
    def mean_rpa_over_y_and_pt(self, cent_tag, y_range, pt_range) -> float:
        vals = [r.mean_rpa_over_y_and_pt(cent_tag, y_range, pt_range) for r in self.runs]
        return float(np.mean(vals))
    def mean_rpa_err_over_y_and_pt(self, cent_tag, y_range, pt_range) -> float:
        vals = [r.mean_rpa_over_y_and_pt(cent_tag, y_range, pt_range) for r in self.runs]
        if len(vals) < 2: return 0.0
        return 0.5 * (float(np.max(vals)) - float(np.min(vals)))

# ==============================================================
# =============== Loader (single or ensemble) ==================
# ==============================================================
def load_eloss_run(base_dir: str, particle: Optional[str] = None,
                   log_level: int = logging.INFO, quad_n_y: int = 32, quad_n_pt: int = 64,
                   choose: Optional[List[str]] = None) -> ELossRun | ELossEnsemble:
    """
    If BASE contains >=2 'output_*' subruns with cent_* files, return ELossEnsemble
    (uses all runs or those listed in 'choose').
    Otherwise return a single ELossRun (BASE itself).
    """
    logger = make_logger(level=log_level)
    base = Path(base_dir)

    # discover candidate run roots under output_*/*
    run_roots: List[Path] = []
    for out in sorted(base.glob("output_*")):
        if not out.is_dir(): continue
        if choose is not None and (out.name not in set(choose)):
            continue
        found = None
        # prefer a child dir with cent_* (e.g., JPsi)
        if particle and (out/particle).is_dir() and list((out/particle).glob("cent_*_*.*")):
            found = out/particle
        if found is None:
            for d in out.iterdir():
                if d.is_dir() and list(d.glob("cent_*_*.*")):
                    found = d; break
        if found is None and list(out.glob("cent_*_*.*")):
            found = out
        if found is not None:
            run_roots.append(found)

    # ensemble case
    if len(run_roots) >= 2:
        return ELossEnsemble(run_roots, particle=particle, logger=logger,
                             quad_n_y=quad_n_y, quad_n_pt=quad_n_pt)

    # single
    return ELossRun(base, particle=particle, logger=logger,
                    quad_n_y=quad_n_y, quad_n_pt=quad_n_pt)

# ==============================================================
# =============== Plotting helpers (band-aware) ================
# ==============================================================
def _init_ax(ax=None, xlabel=None, ylabel=None, title=None):
    import matplotlib.pyplot as plt
    if ax is None: fig, ax = plt.subplots(figsize=(6,4))
    else: fig = ax.figure
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    return fig, ax

def plot_rpa_vs_centrality(run: ELossRun | ELossEnsemble, y_range=(-5,5), pt_range=(0.1,20.0),
                           show_yerr=True, ax=None):
    import matplotlib.pyplot as plt
    df = run.rpa_vs_centrality(y_range=y_range, pt_range=pt_range)
    fig, ax = _init_ax(ax, "Centrality [%]", "RpA",
                       f"RpA vs Centrality ({getattr(run,'particle','')} @ {getattr(run,'energy','')})")
    if {"lo","hi"}.issubset(df.columns):
        ax.fill_between(df["cent_right"], df["lo"], df["hi"], step="post", alpha=0.25)
        ax.step(df["cent_right"], df["RpA"], where="post")
    else:
        yerr = df["dRpA"].values if (show_yerr and "dRpA" in df) else None
        xerr = df["xerr"].values if "xerr" in df else None
        ax.errorbar(df["cent_mid"], df["RpA"], xerr=xerr, yerr=yerr, fmt="o", capsize=3)
    ax.set_xlim(-2, 102)
    return fig, ax, df

def plot_rpa_vs_y_grid(run: ELossRun | ELossEnsemble, cent_tags: Optional[List[str]] = None,
                       y_edges=None, pt_range=(0.1,20.0), ncols=2, figsize=(10,8)):
    import matplotlib.pyplot as plt, numpy as np
    if cent_tags is None:
        cent_tags = run.available_cent_tags
    if y_edges is None:
        y_edges = np.linspace(-5, 5, 21)
    n = len(cent_tags); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False); axes = axes.ravel()
    for i, tag in enumerate(cent_tags):
        df = run.rpa_vs_y(tag, y_edges=y_edges, pt_range=pt_range)
        _, ax = _init_ax(axes[i], "y", "RpA", f"{tag}%")
        if {"lo","hi"}.issubset(df.columns):
            ax.fill_between(df["y_right"], df["lo"], df["hi"], step="post", alpha=0.25)
        ax.step(df["y_right"], df["RpA"], where="post")
    for j in range(i+1, axes.size): axes[j].set_visible(False)
    fig.tight_layout()
    return fig, axes[:n]

def plot_rpa_vs_pt_grid(run: ELossRun | ELossEnsemble, y_range, cent_tags: Optional[List[str]] = None,
                        pt_edges=None, ncols=2, figsize=(10,8)):
    import matplotlib.pyplot as plt, numpy as np
    if cent_tags is None:
        cent_tags = run.available_cent_tags
    if pt_edges is None:
        pt_edges = np.arange(0.0, 20.0+2.5, 2.5)
    n = len(cent_tags); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False); axes = axes.ravel()
    for i, tag in enumerate(cent_tags):
        df = run.rpa_vs_pt(tag, y_range=y_range, pt_edges=pt_edges)
        _, ax = _init_ax(axes[i], "pT [GeV]", "RpA", f"{tag}%  {y_range[0]}<y<{y_range[1]}")
        if {"lo","hi"}.issubset(df.columns):
            ax.fill_between(df["pt_right"], df["lo"], df["hi"], step="post", alpha=0.25)
        ax.step(df["pt_right"], df["RpA"], where="post")
    for j in range(i+1, axes.size): axes[j].set_visible(False)
    fig.tight_layout()
    return fig, axes[:n]
