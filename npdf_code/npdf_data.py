# -*- coding: utf-8 -*-
"""
npdf_data.py
============
TopDrawer I/O and nPDF dataset management.

This module handles:
- Reading TopDrawer .top files (differential cross sections)
- Parsing data sections with y-rapidity bins
- Managing pp/pA central + error sets
- Kick selection (pp baseline vs +dpt broadening)

Public API
----------
from npdf_data import (
    load_top_file, discover_by_number, read_topdrawer,
    TopData, NPDFSystem,
)

Author: Refactored module (2025)
"""
from __future__ import annotations

import os, re, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Sequence, Dict, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TopDrawer I/O
# ---------------------------------------------------------------------------

_NUM_RE = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

@dataclass
class TopData:
    """Container for parsed TopDrawer data."""
    df: pd.DataFrame               # columns: y, pt, val, err
    y_edges: np.ndarray            # left edges per y-slab
    y_centers: np.ndarray          # y-centers per slab
    header: str


def _sections(text: str) -> List[Tuple[int, int]]:
    """Extract (start, end) positions of data sections in TopDrawer text."""
    starts = [m.end() for m in re.finditer(r"SET ORDER X Y DY", text)]
    ends   = [m.end() for m in re.finditer(r"HIST SOLID", text)]
    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n)]


def _y_cut(block_text: str) -> Optional[Tuple[float, float]]:
    """Extract y-range (y_min, y_max) from a data block."""
    m = re.search(rf"({_NUM_RE})\s*<\s*y\s*<\s*({_NUM_RE})", block_text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _parse_numeric(block_text: str) -> np.ndarray:
    """Parse numeric lines into (pt, val, err) array."""
    rows = []
    for ln in block_text.splitlines():
        ln = ln.strip()
        if not ln or not re.match(rf"^{_NUM_RE}", ln):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except Exception:
            pass
    return np.asarray(rows, dtype=float)

def _pairwise(it: Iterable):
    """Iterator that yields consecutive pairs: (a,b), (c,d), ..."""
    it = iter(it)
    return zip(it, it)


def load_top_file(path: str | Path, kick: str = "pp", drop_last_pairs: int = 2) -> TopData:
    """
    Load a TopDrawer .top file.
    
    Parameters
    ----------
    path : str or Path
        Path to .top file
    kick : str, default "pp"
        Which kick to use:
        - "pp" / "no" / "none" / "baseline": pp kick (first in pair)
        - "dpt" / "+dpt" / "broad" / "broadening" / "pa" / "nuc": +dpt kick (second in pair)
    drop_last_pairs : int, default 2
        Number of trailing data pairs to drop (often contain summary info)
    
    Returns
    -------
    TopData
        Parsed data with columns [y, pt, val, err], y_edges, y_centers, header
    
    Notes
    -----
    TopDrawer files contain pairs of data sections:
    - First section: pp kick (baseline proton PDF)
    - Second section: +dpt kick (with pT broadening)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[TopDrawer] File not found: {path}")
    
    text = path.read_text()
    secs = _sections(text)
    if not secs:
        raise ValueError(f"[TopDrawer] No data sections found in {path}")

    # Extract data blocks and pair them
    blocks = [text[s:e] for (s, e) in secs]
    pairs  = list(_pairwise(blocks))
    pairs  = [p for p in pairs if len(p) == 2]
    
    if drop_last_pairs and len(pairs) > drop_last_pairs:
        pairs = pairs[:-drop_last_pairs]

    # Select kick
    k = kick.lower()
    if k in ("pp", "no", "none", "baseline"):
        sel = 0  # first in pair = "pp kick"
    elif k in ("dpt", "+dpt", "broad", "broadening", "pa", "nuc"):
        sel = 1  # second in pair = "+dpt kick"
    else:
        sel = 0
        print(f"[TopDrawer] Unknown kick='{kick}', defaulting to 'pp'.")

    chosen = [p[sel] for p in pairs]

    # Parse each y-slab
    y_cuts, slabs = [], []
    for blk in chosen:
        yc = _y_cut(blk) or _y_cut(blk[:400])
        if yc is None:
            raise ValueError(f"[TopDrawer] Could not find y-cut in a block of {path.name}")
        y_cuts.append(yc)

        arr = _parse_numeric(blk)
        if arr.size == 0:
            continue

        y_mean = 0.5 * (yc[0] + yc[1])
        ycol = np.full(arr.shape[0], y_mean, dtype=float)
        slabs.append(np.c_[ycol, arr])  # [y, pt, val, err]

    if not slabs:
        raise ValueError(f"[TopDrawer] No numeric data parsed from {path}")

    # Combine all slabs
    data = np.vstack(slabs)
    df = (pd.DataFrame(data, columns=["y", "pt", "val", "err"])
            .sort_values(["y", "pt"], kind="mergesort")
            .reset_index(drop=True))

    y_left_edges = np.array([lc for (lc, rc) in y_cuts], dtype=float)
    y_centers    = np.array(sorted(df["y"].unique()), dtype=float)

    # Extract header
    m = re.search(r"^\s*TITLE\s+(.+)$", text, flags=re.MULTILINE)
    header = m.group(1).strip() if m else path.name

    return TopData(df=df, y_edges=y_left_edges, y_centers=y_centers, header=header)


# Alias for backwards compatibility
read_topdrawer = load_top_file


def discover_by_number(folder: str | Path) -> List[Tuple[str, int]]:
    """
    Find all .top files matching pattern *e21NN*.top and return sorted by NN.
    
    Parameters
    ----------
    folder : str or Path
        Directory to search
    
    Returns
    -------
    list of (filepath, number)
        Sorted by number (NN from e21NN)
    
    Notes
    -----
    Expected file naming: *e21000diff.top (pp central), *e21001diff.top (pA central),
    *e21002diff.top, *e21003diff.top, ... (pA error sets)
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.top"))
    pat = re.compile(r".*e21(\d+)diff\.top$", re.IGNORECASE)
    out: List[Tuple[str, int]] = []
    for p in files:
        m = pat.match(p.name)
        if m:
            out.append((str(p), int(m.group(1))))
    out.sort(key=lambda x: x[1])
    return out


# ---------------------------------------------------------------------------
# NPDFSystem: Dataset container
# ---------------------------------------------------------------------------

@dataclass
class NPDFSystem:
    """
    Container for nPDF cross section predictions (pp + pA central + pA error sets).
    
    Attributes
    ----------
    pp_path : str
        Path to pp central .top file (e21000diff.top)
    pa_path : str
        Path to pA central .top file (e21001diff.top)
    error_paths : list of str
        Paths to pA error set .top files (e21002diff.top, e21003diff.top, ...)
    kick : str, default "pp"
        Which kick to use for all files
    name : str, default "system"
        Descriptive name for this dataset
    
    After calling .load():
    ----------------------
    df_pp : pd.DataFrame
        pp central cross sections [y, pt, val, err]
    df_pa : pd.DataFrame
        pA central cross sections [y, pt, val, err]
    df_errors : list of pd.DataFrame
        pA error sets, each with [y, pt, val, err]
    y_edges : pd.Series
        Left bin edges for y-slabs
    y_centers : pd.Series
        Centers of y-slabs
    
    Notes
    -----
    File ordering convention (from TopDrawer output):
      e21000diff.top: pp baseline (proton-proton)
      e21001diff.top: pA central (proton-nucleus)
      e21002diff.top, e21003diff.top, ...: pA error sets (EPPS21 Hessian members)
    """
    pp_path: str
    pa_path: str
    error_paths: List[str]
    kick: str = "pp"
    name: str = "system"

    # Populated by .load()
    df_pp: pd.DataFrame = field(init=False, repr=False)
    df_pa: pd.DataFrame = field(init=False, repr=False)
    df_errors: List[pd.DataFrame] = field(init=False, repr=False)
    y_edges: Optional[pd.Series] = field(init=False, default=None, repr=False)
    y_centers: Optional[pd.Series] = field(init=False, default=None, repr=False)

    def load(self) -> "NPDFSystem":
        """
        Load all files and normalize data.
        
        Returns
        -------
        self
            Allows chaining: system = NPDFSystem(...).load()
        """
        pp = load_top_file(self.pp_path, kick=self.kick)
        pa = load_top_file(self.pa_path, kick=self.kick)
        errs = [load_top_file(p, kick=self.kick).df for p in self.error_paths]

        # Normalize dtypes and sort
        for df in [pp.df, pa.df] + errs:
            for c in ("y", "pt", "val", "err"):
                if c not in df.columns:
                    raise ValueError(f"[NPDFSystem] Missing column '{c}' in data")
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.sort_values(["y", "pt"], inplace=True, kind="mergesort")
            df.reset_index(drop=True, inplace=True)

        self.df_pp = pp.df.copy()
        self.df_pa = pa.df.copy()
        self.df_errors = [d.copy() for d in errs]
        
        # Keep y-edges for plotting
        self.y_edges = pd.Series(pp.y_edges)
        self.y_centers = pd.Series(pp.y_centers)
        
        return self

    @staticmethod
    def from_folder(folder: str, kick: str = "pp", name: str = "system") -> "NPDFSystem":
        """
        Auto-discover and load all .top files from a folder.
        
        Parameters
        ----------
        folder : str
            Directory containing e21NNNdiff.top files
        kick : str, default "pp"
            Which kick to use
        name : str, default "system"
            Descriptive name
        
        Returns
        -------
        NPDFSystem
            Loaded system with pp, pA central, and error sets
        
        Raises
        ------
        ValueError
            If fewer than 2 files found (need at least pp + pA central)
        """
        files = discover_by_number(folder)
        if len(files) < 2:
            raise ValueError(
                f"[NPDFSystem] Need at least 2 files (pp + pA central) in {folder}, "
                f"found {len(files)}"
            )
        
        pp_central = files[0][0]
        pa_central = files[1][0]
        pa_errors  = [p for (p, _) in files[2:]]
        
        return NPDFSystem(pp_central, pa_central, pa_errors, kick=kick, name=name).load()

    def __repr__(self) -> str:
        loaded = hasattr(self, 'df_pp')
        status = "loaded" if loaded else "not loaded"
        n_err = len(self.error_paths)
        return f"NPDFSystem(name='{self.name}', kick='{self.kick}', error_sets={n_err}, {status})"

# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------

def style_axes(ax, xlab, ylab, grid: bool=False, logx: bool=False, logy: bool=False):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    if grid: ax.grid(True, which="both", alpha=0.25)
    # ---- MINOR TICKS (added) ----
    try:
        from matplotlib.ticker import AutoMinorLocator
        if not logx:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if not logy:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="both", which="major", length=5, width=1.0)
        ax.tick_params(axis="both", which="minor", length=3, width=0.8)
    except Exception:
        pass

def _stairs_xy(left_edges, y):
    """
    Ensure step/filled bands reach the last bin's right edge:
    returns x_plot (N+1) and y_plot (N+1) for where='post'.
    """
    x = np.asarray(left_edges, float)
    y = np.asarray(y, float)
    if x.size == 0:
        return x, y
    dx = np.diff(x)
    # prefer last spacing; fall back to median positive spacing; else 0.5
    if dx.size and np.isfinite(dx[-1]) and (dx[-1] > 0):
        dx_last = dx[-1]
    elif np.any(dx > 0):
        dx_last = np.median(dx[dx > 0])
    else:
        dx_last = 0.5
    x_full = np.r_[x, x[-1] + dx_last]
    y_full = np.r_[y, y[-1]]
    return x_full, y_full

def step_band_xy(ax, x_left, y_c, y_lo, y_hi, label: Optional[str]=None, color=None):
    xC, yC = _stairs_xy(x_left, y_c)
    _,  yL = _stairs_xy(x_left, y_lo)
    _,  yH = _stairs_xy(x_left, y_hi)
    lines = ax.step(xC, yC, where="post", label=label, color=color, linewidth=2)
    line_color = color if color is not None else lines[0].get_color()
    ax.fill_between(xC, yL, yH, step="post", alpha=0.25, color=line_color)

def centers_to_left_edges(centers: np.ndarray, width: Optional[float] = None) -> np.ndarray:
    c = np.asarray(centers, float); c = np.unique(np.sort(c))
    if len(c) < 2: return c.copy()
    left = np.empty_like(c)
    left[0]  = c[0] - 0.5*(c[1] - c[0])
    left[1:] = 0.5*(c[:-1] + c[1:])
    return left

def step_band_from_centers(ax, x_centers, y_c, y_lo, y_hi, **kwargs):
    x_left = centers_to_left_edges(x_centers)
    return step_band_xy(ax, x_left, y_c, y_lo, y_hi, **kwargs)

def step_band_from_left_edges(ax, left_edges, y_c, y_lo, y_hi, **kwargs):
    return step_band_xy(ax, np.asarray(left_edges, float), y_c, y_lo, y_hi, **kwargs)

def note_box(ax, text: str, loc: str = "upper right", size: float = 10, alpha: float = 0.85):
    from matplotlib.offsetbox import AnchoredText
    at = AnchoredText(text, loc=loc, prop=dict(size=size), frameon=True, borderpad=0.6)
    at.patch.set_alpha(alpha)
    ax.add_artist(at)
    return at

# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

JoinMode = Literal["intersect", "nearest"]

def _nearest_remap(src: pd.DataFrame, target_xy: pd.DataFrame) -> pd.DataFrame:
    a = src[["y","pt"]].to_numpy()
    b = target_xy[["y","pt"]].to_numpy()
    val = src["val"].to_numpy()
    res = np.empty(len(b), dtype=float)
    for i, (yy, pp) in enumerate(b):
        d2 = (a[:,0]-yy)**2 + (a[:,1]-pp)**2
        j = int(np.argmin(d2))
        res[i] = val[j]
    out = target_xy.copy()
    out["val"] = res
    return out

@dataclass
class RpAAnalysis:
    """RpA grid building + band propagation."""

    def compute_rpa_grid(self, df_pp, df_pa, df_errors, join: JoinMode = "intersect",
                         lowpt_policy: Literal["none","shift","interp","drop"] = "interp", include_members: bool = True,
                         pt_shift_min: float | None = None, 
                         shift_if_r_below: float | None = None) -> pd.DataFrame:
        """
        Return columns: [y, pt, r_central, r_lo, r_hi] and, if include_members=True,
        also r_mem_001... columns for each error-set member (exactly same order as df_errors).
        Error bands from Hessian: pairwise if even #members; else pos/neg envelope.
        """
        base = df_pa[["y","pt"]].drop_duplicates()

        if join == "intersect":
            base_all = base.merge(df_pp[["y","pt"]].drop_duplicates(), on=["y","pt"], how="inner")
            for dfe in df_errors:
                base_all = base_all.merge(dfe[["y","pt"]].drop_duplicates(), on=["y","pt"], how="inner")
            base = base_all.sort_values(["y","pt"]).reset_index(drop=True)

            pa_al = base.merge(df_pa[["y","pt","val"]], on=["y","pt"], how="inner")["val"].to_numpy(float)
            pp_al = base.merge(df_pp[["y","pt","val"]], on=["y","pt"], how="inner")["val"].to_numpy(float)
            r0 = np.divide(pa_al, pp_al, out=np.full_like(pa_al, np.nan), where=(pp_al!=0))

            mems: List[np.ndarray] = []
            for dfe in df_errors:
                tmp = base.merge(dfe[["y","pt","val"]], on=["y","pt"], how="inner")["val"].to_numpy(float)
                mems.append(np.divide(tmp, pp_al, out=np.full_like(tmp, np.nan), where=(pp_al!=0)))
        else:
            base = base.sort_values(["y","pt"]).reset_index(drop=True)
            pp_al = _nearest_remap(df_pp[["y","pt","val"]], base)["val"].to_numpy(float)
            pa_al = _nearest_remap(df_pa[["y","pt","val"]], base)["val"].to_numpy(float)
            r0 = np.divide(pa_al, pp_al, out=np.full_like(pa_al, np.nan), where=(pp_al!=0))
            mems: List[np.ndarray] = []
            for dfe in df_errors:
                tmp = _nearest_remap(dfe[["y","pt","val"]], base)["val"].to_numpy(float)
                mems.append(np.divide(tmp, pp_al, out=np.full_like(tmp, np.nan), where=(pp_al!=0)))

        # ---- OPTIONAL: low-pT policy (default OFF) -------------------------------
        if (lowpt_policy != "none") or (pt_shift_min is not None) or (shift_if_r_below is not None):
            yv = base["y"].to_numpy(); pv = base["pt"].to_numpy()
            yuniq = np.unique(yv)
            pt_star = float(pt_shift_min) if pt_shift_min is not None else None
            r_cut   = float(shift_if_r_below) if shift_if_r_below is not None else None

            for yy in yuniq:
                I = np.where(yv == yy)[0]
                if I.size == 0: continue

                # candidate “good” points at/above the floor with finite R
                J = I
                if pt_star is not None:
                    J = J[pv[J] >= pt_star]
                J = J[np.isfinite(r0[J])]
                if J.size == 0:
                    continue

                # j1: first good above floor; j2: next one (for interpolation)
                j1 = J[np.argmin(pv[J] - (pt_star if pt_star is not None else pv[J].min()))]
                gt = J[pv[J] > pv[j1]]
                j2 = gt[0] if gt.size else j1  # if only one, “interp” degenerates to “shift”

                for i in I:
                    bad_pt = (pt_star is not None) and (pv[i] < pt_star)
                    bad_r  = (r_cut   is not None) and (not np.isfinite(r0[i]) or (r0[i] < r_cut))
                    if not (bad_pt or bad_r):
                        continue

                    if lowpt_policy == "drop":
                        r0[i] = np.nan
                        for m in range(len(mems)): mems[m][i] = np.nan

                    elif lowpt_policy == "interp" and (j2 != j1):
                        t = (pv[i] - pv[j1]) / (pv[j2] - pv[j1])
                        r0[i] = (1-t)*r0[j1] + t*r0[j2]
                        for m in range(len(mems)):
                            mems[m][i] = (1-t)*mems[m][j1] + t*mems[m][j2]

                    else:  # "shift" or fallback when only one good point exists
                        r0[i] = r0[j1]
                        for m in range(len(mems)):
                            mems[m][i] = mems[m][j1]

        # Hessian band
        if mems:
            M = np.stack(mems, axis=0)
            if M.shape[0] % 2 == 0:
                D = M[0::2, :] - M[1::2, :]
                h = 0.5 * np.sqrt(np.sum(D*D, axis=0))
                err_plus = h; err_minus = h
            else:
                diff = M - r0[None, :]
                pos = np.maximum(diff, 0.0)
                neg = np.minimum(diff, 0.0)
                err_plus  = np.sqrt(np.sum(pos * pos, axis=0))
                err_minus = np.sqrt(np.sum(neg * neg, axis=0))
        else:
            M = np.empty((0, len(r0)))
            err_plus = np.zeros_like(r0); err_minus = np.zeros_like(r0)

        out = base.copy()
        out["r_central"] = r0
        out["r_lo"] = r0 - err_minus
        out["r_hi"] = r0 + err_plus

        # Optional: expose members as columns for exact downstream weighting/min-bias
        if include_members and M.size:
            for j in range(M.shape[0]):
                out[f"r_mem_{j+1:03d}"] = M[j]

        return out.sort_values(["y","pt"]).reset_index(drop=True)

    def rpa_binned(
        self,
        rgrid: pd.DataFrame,
        pa_central: pd.DataFrame,
        *,
        y_edges: Optional[Sequence[float]] = None,
        pt_edges: Optional[Sequence[float]] = None,
        weight_mode: Literal["pa","pp"] = "pa",
        pp_central: Optional[pd.DataFrame] = None,
        pt_floor: float = 0.0
    ) -> pd.DataFrame:
        """
        General binning along y, pT, or both. Returns weighted means (σ weights).
        If pt_floor>0, bins with right edge <= pt_floor are dropped (clean low-pT handling).
        Columns returned:
          - if 2D: y_left, pt_left, r_central, r_lo, r_hi
          - if 1D-y: y_left, r_central, r_lo, r_hi
          - if 1D-pt: pt_left, r_central, r_lo, r_hi
        """
        base = rgrid.copy()
        wtab = self._make_weight_table(rgrid, pa_central, df_pp=pp_central, mode=weight_mode)
        base = base.merge(wtab, on=["y","pt"], how="inner")

        # Apply cuts via bins
        y_bins = pd.IntervalIndex(pd.cut(base["y"], y_edges, right=False)) if y_edges is not None else None
        pt_bins = pd.IntervalIndex(pd.cut(base["pt"], pt_edges, right=False)) if pt_edges is not None else None

        rows = []
        if (y_bins is not None) and (pt_bins is not None):
            base = base.assign(y_bin=y_bins, pt_bin=pt_bins).dropna(subset=["y_bin","pt_bin"])
            # optional drop of lowest pT bin(s)
            if pt_floor > 0:
                base = base[base["pt_bin"].apply(lambda I: I.right > pt_floor)]
            grp = base.groupby(["y_bin","pt_bin"], sort=True)
            for (yb, pb), g in grp:
                w = g["w"].to_numpy()
                rows.append({
                    "y_left": float(yb.left),
                    "pt_left": float(pb.left),
                    "r_central": self._weighted_avg(g["r_central"].to_numpy(), w),
                    "r_lo":      self._weighted_avg(g["r_lo"].to_numpy(),      w),
                    "r_hi":      self._weighted_avg(g["r_hi"].to_numpy(),      w),
                })
            return pd.DataFrame(rows).sort_values(["y_left","pt_left"]).reset_index(drop=True)

        if y_bins is not None:
            base = base.assign(y_bin=y_bins).dropna(subset=["y_bin"])
            grp = base.groupby(["y_bin"], sort=True)
            for yb, g in grp:
                w = g["w"].to_numpy()
                rows.append({
                    "y_left": float(yb.left),
                    "r_central": self._weighted_avg(g["r_central"].to_numpy(), w),
                    "r_lo":      self._weighted_avg(g["r_lo"].to_numpy(),      w),
                    "r_hi":      self._weighted_avg(g["r_hi"].to_numpy(),      w),
                })
            return pd.DataFrame(rows).sort_values(["y_left"]).reset_index(drop=True)

        if pt_bins is not None:
            base = base.assign(pt_bin=pt_bins).dropna(subset=["pt_bin"])
            if pt_floor > 0:
                base = base[base["pt_bin"].apply(lambda I: I.right > pt_floor)]
            grp = base.groupby(["pt_bin"], sort=True)
            for pb, g in grp:
                w = g["w"].to_numpy()
                rows.append({
                    "pt_left": float(pb.left),
                    "r_central": self._weighted_avg(g["r_central"].to_numpy(), w),
                    "r_lo":      self._weighted_avg(g["r_lo"].to_numpy(),      w),
                    "r_hi":      self._weighted_avg(g["r_hi"].to_numpy(),      w),
                })
            return pd.DataFrame(rows).sort_values(["pt_left"]).reset_index(drop=True)

        # no bin edges provided — return original grid
        return base[["y","pt","r_central","r_lo","r_hi"]].copy()

    @staticmethod
    def _weighted_avg(series: np.ndarray, weights: np.ndarray) -> float:
        w = np.asarray(weights, float); x = np.asarray(series, float)
        m = (w > 0) & np.isfinite(x)
        if not np.any(m): return np.nan
        return float(np.sum(x[m]*w[m]) / np.sum(w[m]))
    
    @staticmethod
    def _make_weight_table(
        rgrid: pd.DataFrame,
        df_pa: pd.DataFrame,
        df_pp: Optional[pd.DataFrame] = None,
        mode: Literal["pa", "pp"] = "pa"
    ) -> pd.DataFrame:
        """
        Return a table with columns ['y','pt','w'] aligned to rgrid's (y,pt).
        - mode='pa' (default):  w = σ_pA
        - mode='pp':            w = σ_pp  (from df_pp if available, else σ_pA / R_central)
        """
        base = rgrid[["y","pt"]].drop_duplicates().sort_values(["y","pt"])
        if mode == "pa":
            wtab = base.merge(df_pa[["y","pt","val"]], on=["y","pt"], how="left").rename(columns={"val":"w"})
            wtab["w"] = wtab["w"].clip(lower=0).fillna(0.0)
            return wtab

        # mode == "pp"
        if (df_pp is not None) and {"y","pt","val"}.issubset(df_pp.columns):
            wtab = base.merge(df_pp[["y","pt","val"]], on=["y","pt"], how="left").rename(columns={"val":"w"})
            wtab["w"] = wtab["w"].clip(lower=0).fillna(0.0)
            return wtab

        # fallback: σ_pp ≈ σ_pA / R_central
        pa = base.merge(df_pa[["y","pt","val"]], on=["y","pt"], how="left").rename(columns={"val":"pa"})
        rc = base.merge(rgrid[["y","pt","r_central"]], on=["y","pt"], how="left")
        w = np.where(rc["r_central"] > 0, pa["pa"] / rc["r_central"], np.nan)
        wtab = base.copy()
        wtab["w"] = np.clip(w, 0, None)
        wtab["w"] = wtab["w"].fillna(0.0)
        return wtab

    def rpa_vs_y_pt_threshold(
        self,
        rgrid: pd.DataFrame,
        pa_central: pd.DataFrame,
        pt_min: float,
        *,
        weight_mode: Literal["pa","pp"] = "pa",
        pp_central: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        r = rgrid.copy()
        wtab = RpAAnalysis._make_weight_table(r, pa_central, df_pp=pp_central, mode=weight_mode)
        merged = r.merge(wtab, on=["y","pt"], how="inner")
        merged = merged[merged["pt"] >= float(pt_min)].copy()

        rows = []
        for y, g in merged.groupby("y", sort=True):
            rows.append({
                "y": float(y),
                "r_central": self._weighted_avg(g["r_central"], g["w"]),
                "r_lo":      self._weighted_avg(g["r_lo"],      g["w"]),
                "r_hi":      self._weighted_avg(g["r_hi"],      g["w"]),
            })
        return pd.DataFrame(rows).sort_values("y").reset_index(drop=True)

    def rpa_vs_pt_in_y(
        self,
        rgrid: pd.DataFrame,
        pa_central: pd.DataFrame,
        y_min: float,
        y_max: float,
        *,
        weight_mode: Literal["pa","pp"] = "pa",
        pp_central: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        r = rgrid[(rgrid["y"]>=y_min)&(rgrid["y"]<=y_max)].copy()
        wtab_all = RpAAnalysis._make_weight_table(rgrid, pa_central, df_pp=pp_central, mode=weight_mode)
        wtab = wtab_all[(wtab_all["y"]>=y_min)&(wtab_all["y"]<=y_max)]
        merged = r.merge(wtab, on=["y","pt"], how="inner")

        rows = []
        for pt, g in merged.groupby("pt", sort=True):
            rows.append({
                "pt": float(pt),
                "r_central": self._weighted_avg(g["r_central"], g["w"]),
                "r_lo":      self._weighted_avg(g["r_lo"],      g["w"]),
                "r_hi":      self._weighted_avg(g["r_hi"],      g["w"]),
            })
        return pd.DataFrame(rows).sort_values("pt").reset_index(drop=True)


    def coarse_bin_along_pt(self, rgrid, pa_central, block_size: int = 5):
        pdf = pa_central[["y","pt","val"]].rename(columns={"val":"w"})
        merged = rgrid.merge(pdf, on=["y","pt"], how="inner").sort_values(["y","pt"]).reset_index(drop=True)

        def _reduce(df):
            w = df["w"].to_numpy()
            return pd.Series({
                "y_bar": df["y"].mean(),
                "pt_bar": df["pt"].mean(),
                "r_central": self._weighted_avg(df["r_central"].to_numpy(), w),
                "r_lo":      self._weighted_avg(df["r_lo"].to_numpy(),      w),
                "r_hi":      self._weighted_avg(df["r_hi"].to_numpy(),      w),
                "sigma":     self._weighted_avg(df["w"].to_numpy(),         w),
            })

        idx = np.arange(len(merged))
        groups = (idx // block_size)
        binned = merged.groupby(groups, sort=True).apply(_reduce).reset_index(drop=True)

        rpa = binned[["y_bar","pt_bar","r_central","r_lo","r_hi"]].copy()
        xsc = binned[["y_bar","pt_bar","sigma"]].copy()
        return rpa, xsc

    def rpa_vs_pt_widebins(
        self,
        rgrid: pd.DataFrame,
        pa_central: pd.DataFrame,
        y_min: float,
        y_max: float,
        width: float = 2.5,
        *,
        weight_mode: Literal["pa","pp"] = "pa",
        pp_central: Optional[pd.DataFrame] = None,
        pt_floor: float = 0.0
    ) -> pd.DataFrame:
        r = rgrid[(rgrid["y"]>=y_min)&(rgrid["y"]<=y_max)].copy()
        wtab_all = RpAAnalysis._make_weight_table(rgrid, pa_central, df_pp=pp_central, mode=weight_mode)
        wtab = wtab_all[(wtab_all["y"]>=y_min)&(wtab_all["y"]<=y_max)]
        merged = r.merge(wtab, on=["y","pt"], how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["pt_left","r_central","r_lo","r_hi"])

        pts = np.sort(merged["pt"].unique())
        pmin, pmax = float(pts.min()), float(pts.max())
        # robust edges from the data range
        start = width * math.floor(pmin / width)
        stop  = width * math.ceil(pmax / width) + 1e-12
        edges = np.arange(start, stop, width)

        rows = []
        for i in range(len(edges)-1):
            left, right = float(edges[i]), float(edges[i+1])
            if right <= float(pt_floor):   # <— drop the troublesome lowest bin if desired
                continue
            g = merged[(merged["pt"]>=left)&(merged["pt"]<right)]
            if len(g)==0: continue
            w = g["w"].to_numpy()
            rows.append({
                "pt_left": left,
                "r_central": self._weighted_avg(g["r_central"].to_numpy(), w),
                "r_lo":      self._weighted_avg(g["r_lo"].to_numpy(),      w),
                "r_hi":      self._weighted_avg(g["r_hi"].to_numpy(),      w),
            })
        return pd.DataFrame(rows).sort_values("pt_left").reset_index(drop=True)

    def compute_rpa_members(self, df_pp, df_pa, df_errors, join="intersect",
                        lowpt_policy: Literal["none","shift","interp","drop"] = "none",
                        pt_shift_min: float | None = None,
                        shift_if_r_below: float | None = None):
        base = df_pa[["y","pt"]].drop_duplicates()
        if join == "intersect":
            base_all = base.merge(df_pp[["y","pt"]].drop_duplicates(), on=["y","pt"])
            for dfe in df_errors:
                base_all = base_all.merge(dfe[["y","pt"]].drop_duplicates(), on=["y","pt"])
            base = base_all.sort_values(["y","pt"]).reset_index(drop=True)
            pa0 = base.merge(df_pa[["y","pt","val"]], on=["y","pt"])["val"].to_numpy()
            pp0 = base.merge(df_pp[["y","pt","val"]], on=["y","pt"])["val"].to_numpy()
            with np.errstate(divide='ignore', invalid='ignore'):
                r0  = np.divide(pa0, pp0, out=np.full_like(pa0, np.nan), where=(pp0!=0))
            mems = []
            for dfe in df_errors:
                pa_k = base.merge(dfe[["y","pt","val"]], on=["y","pt"])["val"].to_numpy()
                mems.append(np.divide(pa_k, pp0, out=np.full_like(pa_k, np.nan), where=(pp0!=0)))
        else:
            base = base.sort_values(["y","pt"]).reset_index(drop=True)
            pp0 = _nearest_remap(df_pp[["y","pt","val"]], base)["val"].to_numpy()
            pa0 = _nearest_remap(df_pa[["y","pt","val"]], base)["val"].to_numpy()
            with np.errstate(divide='ignore', invalid='ignore'):
                r0  = np.divide(pa0, pp0, out=np.full_like(pa0, np.nan), where=(pp0!=0))
            mems = []
            for dfe in df_errors:
                pa_k = _nearest_remap(dfe[["y","pt","val"]], base)["val"].to_numpy()
                mems.append(np.divide(pa_k, pp0, out=np.full_like(pa_k, np.nan), where=(pp0!=0)))

        # ---- OPTIONAL: low-pT policy (default OFF) -------------------------------
        if (lowpt_policy != "none") or (pt_shift_min is not None) or (shift_if_r_below is not None):
            yv = base["y"].to_numpy(); pv = base["pt"].to_numpy()
            yuniq = np.unique(yv)
            pt_star = float(pt_shift_min) if pt_shift_min is not None else None
            r_cut   = float(shift_if_r_below) if shift_if_r_below is not None else None

            for yy in yuniq:
                I = np.where(yv == yy)[0]
                if I.size == 0: continue

                # candidate “good” points at/above the floor with finite R
                J = I
                if pt_star is not None:
                    J = J[pv[J] >= pt_star]
                J = J[np.isfinite(r0[J])]
                if J.size == 0:
                    continue

                # j1: first good above floor; j2: next one (for interpolation)
                j1 = J[np.argmin(pv[J] - (pt_star if pt_star is not None else pv[J].min()))]
                gt = J[pv[J] > pv[j1]]
                j2 = gt[0] if gt.size else j1  # if only one, “interp” degenerates to “shift”

                for i in I:
                    bad_pt = (pt_star is not None) and (pv[i] < pt_star)
                    bad_r  = (r_cut   is not None) and (not np.isfinite(r0[i]) or (r0[i] < r_cut))
                    if not (bad_pt or bad_r):
                        continue

                    if lowpt_policy == "drop":
                        r0[i] = np.nan
                        for m in range(len(mems)): mems[m][i] = np.nan

                    elif lowpt_policy == "interp" and (j2 != j1):
                        t = (pv[i] - pv[j1]) / (pv[j2] - pv[j1])
                        r0[i] = (1-t)*r0[j1] + t*r0[j2]
                        for m in range(len(mems)):
                            mems[m][i] = (1-t)*mems[m][j1] + t*mems[m][j2]

                    else:  # "shift" or fallback when only one good point exists
                        r0[i] = r0[j1]
                        for m in range(len(mems)):
                            mems[m][i] = mems[m][j1]

        M = np.stack(mems, axis=0) if mems else np.empty((0, len(r0)))
        return base, r0, M


    # --- add inside class RpAAnalysis ---------------------------------
    
    @staticmethod
    def _hessian_from_members(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Symmetric Hessian band from member array M with shape (Nmem, Npts).
        If Nmem is even (48), use pairwise master formula: Δ = 0.5*sqrt(sum (Δ_k)^2).
        Returns (err_minus, err_plus) with shapes (Npts,), symmetric here.
        """
        if M.size == 0:
            return np.array([]), np.array([])
        if M.shape[0] % 2 == 0:
            D = M[0::2, :] - M[1::2, :]
            h = 0.5 * np.sqrt(np.sum(D * D, axis=0))
            return h, h
        # fallback: envelope (rare for nPDF here)
        diff = M - np.nanmean(M, axis=0, keepdims=True)
        pos = np.maximum(diff, 0.0); neg = np.minimum(diff, 0.0)
        return np.sqrt(np.sum(neg * neg, axis=0)), np.sqrt(np.sum(pos * pos, axis=0))
    
    @staticmethod
    def _weights_with_yref(df_pa: pd.DataFrame, df_pp: pd.DataFrame | None,
                           base: pd.DataFrame, *,
                           mode: str = "pa",
                           weight_ref_y: float | str = "local") -> np.ndarray:
        """
        Returns weights aligned to 'base[["y","pt"]]'.
        mode: "pa" or "pp".
        weight_ref_y:
          - "local": use σ(y,pt) at each (y,pt)
          - float (e.g., 0.0): use σ(y_ref,pt) taken from the y-slab closest to y_ref
        """
        Y = base["y"].to_numpy(); P = base["pt"].to_numpy()
    
        def _nearest_y_table(df: pd.DataFrame, y_ref: float) -> dict[float, float]:
            # for each pt, take σ at the y-row whose center is closest to y_ref
            out = {}
            for ptv, g in df.groupby("pt"):
                yvals = g["y"].to_numpy()
                j = int(np.argmin(np.abs(yvals - y_ref)))
                out[float(ptv)] = float(g["val"].to_numpy()[j])
            return out
    
        if mode.lower() == "pa":
            if weight_ref_y == "local":
                wtab = base.merge(df_pa[["y","pt","val"]], on=["y","pt"], how="left")["val"].to_numpy()
            else:
                lut = _nearest_y_table(df_pa, float(weight_ref_y))
                wtab = np.array([lut.get(float(p), np.nan) for p in P], float)
            return np.clip(np.nan_to_num(wtab, nan=0.0), 0.0, None)
    
        # mode == "pp"
        if (df_pp is not None) and {"y","pt","val"}.issubset(df_pp.columns):
            if weight_ref_y == "local":
                wtab = base.merge(df_pp[["y","pt","val"]], on=["y","pt"], how="left")["val"].to_numpy()
            else:
                lut = _nearest_y_table(df_pp, float(weight_ref_y))
                wtab = np.array([lut.get(float(p), np.nan) for p in P], float)
            return np.clip(np.nan_to_num(wtab, nan=0.0), 0.0, None)
    
        # fallback: σ_pp ≈ σ_pA / R_central, use same y_ref rule
        if weight_ref_y == "local":
            pa = base.merge(df_pa[["y","pt","val"]], on=["y","pt"], how="left")["val"].to_numpy()
        else:
            lut = _nearest_y_table(df_pa, float(weight_ref_y))
            pa = np.array([lut.get(float(p), np.nan) for p in P], float)
        rc = base.merge(base[["y","pt","r_central"]], on=["y","pt"], how="left")["r_central"].to_numpy()
        w = np.where(rc > 0, pa / rc, np.nan)
        return np.clip(np.nan_to_num(w, nan=0.0), 0.0, None)
    
    def fuse_sigma_with_K(
        self,
        base: pd.DataFrame,
        r_sigma_central: np.ndarray,
        r_sigma_members: np.ndarray,    # shape (48, N)
        K_central: np.ndarray,
        K_members: np.ndarray           # shape (48, N)
    ) -> pd.DataFrame:
        """
        Elementwise fuse the σ-ratio with K(b): R_npdf = K(b)*R_sigma, member-by-member.
        Returns a DataFrame with columns [y, pt, r_central, r_lo, r_hi, r_mem_001..].
        """
        assert r_sigma_members.shape == K_members.shape, "Member count mismatch (σ vs K)"
        r0 = np.asarray(r_sigma_central, float) * np.asarray(K_central, float)
        M  = np.asarray(r_sigma_members, float) * np.asarray(K_members, float)
        em, ep = self._hessian_from_members(M)
        out = base[["y","pt"]].copy()
        out["r_central"] = r0
        out["r_lo"]      = r0 - em
        out["r_hi"]      = r0 + ep
        for j in range(M.shape[0]):
            out[f"r_mem_{j+1:03d}"] = M[j]
        return out
    
    def bin_49sets(
        self,
        df_49: pd.DataFrame,     # has r_central + r_mem_001.. columns
        sys: "NPDFSystem",
        *,
        y_range: tuple[float,float] | None = None,
        pt_range: tuple[float,float] | None = None,
        averaging: str = "weighted",      # "weighted" or "simple"
        weight_kind: str = "pa",          # "pa" or "pp"
        weight_ref_y: float | str = "local"
    ) -> dict:
        """
        One function to bin either a single set or all 49 sets (central + 48 members).
        Returns a dict with scalar r_central, r_lo, r_hi for the selected (y,pt) window.
        """
        g = df_49.copy()
        if y_range is not None:
            g = g[(g["y"] >= y_range[0]) & (g["y"] <= y_range[1])]
        if pt_range is not None:
            g = g[(g["pt"] >= pt_range[0]) & (g["pt"] <= pt_range[1])]
        if g.empty:
            return {"r_central": np.nan, "r_lo": np.nan, "r_hi": np.nan}
    
        if averaging == "simple":
            w = np.ones(len(g), float)
        else:
            w = self._weights_with_yref(
                sys.df_pa, getattr(sys, "df_pp", None), g[["y","pt"]],
                mode=weight_kind, weight_ref_y=weight_ref_y
            )
    
        def _wavg(a):
            m = np.isfinite(a) & (w > 0)
            return float(np.sum(a[m] * w[m]) / np.sum(w[m])) if np.any(m) else np.nan
    
        r0 = _wavg(g["r_central"].to_numpy())
        # member-by-member average, then Hessian on the *binned* numbers
        mem_cols = [c for c in g.columns if c.startswith("r_mem_")]
        if mem_cols:
            Mavg = np.array([_wavg(g[c].to_numpy()) for c in mem_cols], float)  # (48,)
            # pairwise (2k-1,2k) with the original order preserved
            if len(Mavg) % 2 == 0:
                D = Mavg[0::2] - Mavg[1::2]
                h = 0.5 * np.sqrt(np.sum(D * D))
                lo, hi = r0 - h, r0 + h
            else:
                # envelope fallback
                lo, hi = r0 - np.nanstd(Mavg), r0 + np.nanstd(Mavg)
        else:
            lo = hi = r0
        return {"r_central": r0, "r_lo": lo, "r_hi": hi}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # I/O functions
    "load_top_file",
    "read_topdrawer",
    "discover_by_number",
    
    # Data containers
    "TopData",
    "NPDFSystem",
]

# ------------------ USAGE -----------------------
def main():
    # --- setup (nPDF only) ---
    import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from pathlib import Path
    sys.path.append(".")

    from gluon_ratio import EPPS21Ratio, GluonEPPSProvider

    # I/O
    pPb5_dir = "../input/npdf/pPb5TeV"
    pPb8_dir = "../input/npdf/pPb8TeV"
    outdir   = Path("../output-npdf-comparisons"); outdir.mkdir(exist_ok=True)

    # energies (inelastic σ_NN in mb)
    sigma5, sigma8 = 67.0, 71.0

    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 12,
        "axes.grid": False,          # ← no grid by default
        "axes.spines.top": True,     # ← show top frame
        "axes.spines.right": True,   # ← show right frame
    })

    # analysis windows (edit as needed)
    CENT_EDGES = [0,20,40,60,80,100]      # centrality bins (%)
    Y_RANGES_THREE = [(-4.46,-2.96), (-1.37,0.43), (2.03,3.53)]  # Back, Mid, For
    Y_LABELS       = ["-4.46 < y < -2.96", "-1.37 < y < 0.43", "2.03 < y < 3.53"]
    PT_RANGE   = (0.0, 20.0)

    # --- load nPDF systems and grids (Hessian bands or raw lines) ---
    ana  = RpAAnalysis()  # data analyzer
    sys5 = NPDFSystem.from_folder(pPb5_dir, kick="pp", name="p+Pb 5.02 TeV")
    r5   = ana.compute_rpa_grid(sys5.df_pp, sys5.df_pa, sys5.df_errors, pt_shift_min=0, shift_if_r_below=0.0, lowpt_policy="drop", join="intersect")
    sys8 = NPDFSystem.from_folder(pPb8_dir, kick="pp", name="p+Pb 8.16 TeV")
    r8   = ana.compute_rpa_grid(sys8.df_pp, sys8.df_pa, sys8.df_errors, pt_shift_min=0, shift_if_r_below=0.0, lowpt_policy="drop", join="intersect")
    systems = [("5.02", sys5, r5), ("8.16", sys8, r8)]

    # ------- build results FIRST (for both energies), then plot -------
    pairs = [("5.02", sys5, r5), ("8.16", sys8, r8)]
    results = {}
    for tag, sysX, rgrid in pairs:
        df_1p0 = ana.rpa_vs_y_pt_threshold(rgrid, sysX.df_pa, pt_min=1.0)
        df_2p5 = ana.rpa_vs_y_pt_threshold(rgrid, sysX.df_pa, pt_min=2.5)
        results[(tag, "1.0")] = (df_1p0, sysX)
        results[(tag, "2.5")] = (df_2p5, sysX)

    def _x_edges(sysX, df):
        if getattr(sysX, "y_edges", None) is not None and len(sysX.y_edges) == len(df):
            return sysX.y_edges.values
        return centers_to_left_edges(df["y"].values)

    # ---- Figure with subfigures: top = RpA vs y (two panels), bottom = RpA vs pT (three y-windows) ----
    NOTE_LOC_Y  = "upper center"   # change to 'lower left', etc., as needed
    NOTE_LOC_PT = "lower right"

    y_panel_specs = [("2.5", r"$p_T \geq 2.5$ GeV"),
                    ("1.0", r"$p_T \geq 1.0$ GeV")]

    y_windows = [(-1.93, 1.93, "-1.93 < y < 1.93"),  
                 ( 1.5,  4.0, "1.5 < y < 4.0"),
                 (-5.0, -2.5, "-5.0 < y < -2.5")]

    fig = plt.figure(figsize=(12.5, 9.0), constrained_layout=True)
    sft, sfb = fig.subfigures(2, 1, height_ratios=[1.0, 1.2])

    # Keep your colors mapping (already present)
    colors = {"5.02": "tab:blue", "8.16": "tab:red"}

    # ----- TOP subfigure: RpA vs y (2 panels) -----
    axes_y = sft.subplots(1, 2, sharey=True)
    for ax, (ptkey, note_text) in zip(axes_y, y_panel_specs):
        for tag in ["5.02", "8.16"]:
            df, sysX = results[(tag, ptkey)]
            x = _x_edges(sysX, df)
            step_band_xy(ax, x, df["r_central"], df["r_lo"], df["r_hi"],
                        label=f"{tag} TeV", color=colors[tag])   # ← color fixed
        style_axes(ax, "y", r"$R_{pA}$", grid=False)
        ax.set_ylim(0.2, 1.2)
        ax.set_xlim(-5.0, 5.0)
        note_box(ax, note_text, loc=NOTE_LOC_Y)

    # One legend for the whole TOP subfigure
    h_top, l_top = axes_y[0].get_legend_handles_labels()
    sft.legend(h_top, l_top, loc="upper right", frameon=False, ncol=1, fontsize=11)


    # ----- BOTTOM subfigure: RpA vs pT (3 y-windows) -----
    axes_pt = sfb.subplots(1, len(y_windows), sharex=True, sharey=True)
    if len(y_windows) == 1:
        axes_pt = [axes_pt]

    for ax, (ymin, ymax, name) in zip(axes_pt, y_windows):
        for tag, sysX, rgrid in [("5.02", sys5, r5), ("8.16", sys8, r8)]:
            wide = ana.rpa_vs_pt_widebins(
                rgrid, sysX.df_pa, y_min=ymin, y_max=ymax, width=2.5
            )
            step_band_from_left_edges(ax, wide["pt_left"], wide["r_central"],
                                    wide["r_lo"], wide["r_hi"],
                                    label=f"{tag} TeV", color=colors[tag])  # ← color fixed
        style_axes(ax, r"$p_T$ [GeV]", r"$R_{pA}$", grid=False)
        ax.set_xlim(0, 20)
        ax.set_ylim(0.35, 1.25)
        note_box(ax, fr"{name}", loc=NOTE_LOC_PT)

    # One legend for the whole BOTTOM subfigure
    h_bot, l_bot = axes_pt[0].get_legend_handles_labels()
    sfb.legend(h_bot, l_bot, loc="upper center", frameon=False, ncol=2, fontsize=11)

    fig.savefig(f"{outdir}/rpa_y_pt_combined.pdf", bbox_inches="tight")
    plt.show()

    # --- helpers: raw per-member R_pA lines with faded error sets (NO bands) ---
    def _plot_raw_members_vs_y(ax, sysX, pt_min, color, central_label, weight_mode="pa"):
        # common grid & members
        base, r0, M = ana.compute_rpa_members(sysX.df_pp, sysX.df_pa, sysX.df_errors, join="intersect", lowpt_policy="drop", pt_shift_min=0, shift_if_r_below=0.0)

        # weights aligned to base
        rgrid_min = base.copy()
        rgrid_min["r_central"] = r0
        wtab = RpAAnalysis._make_weight_table(rgrid_min, sysX.df_pa, df_pp=sysX.df_pp, mode=weight_mode)
        wfull = wtab["w"].to_numpy()

        yvals = np.sort(base["y"].unique())
        yb = base["y"].to_numpy()
        pb = base["pt"].to_numpy()

        # ---- central line
        ys_c, Rs_c = [], []
        for yy in yvals:
            m = (yb == yy) & (pb >= float(pt_min))
            if not np.any(m): 
                continue
            R = r0[m]
            w = wfull[m]
            g = (w > 0) & np.isfinite(R)
            if np.any(g):
                ys_c.append(float(yy))
                Rs_c.append(float(np.sum(R[g]*w[g]) / np.sum(w[g])))
        if len(ys_c) >= 2:
            ax.plot(ys_c, Rs_c, "-", linewidth=2.0, color=color, label=central_label, zorder=5)

        # ---- error-member lines (faded)
        if M.size:
            for j in range(M.shape[0]):
                ys_j, Rs_j = [], []
                for yy in yvals:
                    m = (yb == yy) & (pb >= float(pt_min))
                    if not np.any(m):
                        continue
                    Rj = M[j, m]
                    w  = wfull[m]
                    g = (w > 0) & np.isfinite(Rj)
                    if np.any(g):
                        ys_j.append(float(yy))
                        Rs_j.append(float(np.sum(Rj[g]*w[g]) / np.sum(w[g])))
                if len(ys_j) >= 2:
                    ax.plot(ys_j, Rs_j, "-", linewidth=0.8, alpha=0.18, color=color, zorder=3)

    def _plot_raw_members_vs_pt(ax, sysX, y_min, y_max, color, central_label, weight_mode="pa"):
        # common grid & members (aligned base order)
        base, r0, M = ana.compute_rpa_members(sysX.df_pp, sysX.df_pa, sysX.df_errors, join="intersect",pt_shift_min=0, shift_if_r_below=0.0)

        rgrid_min = base.copy()
        rgrid_min["r_central"] = r0
        wtab = RpAAnalysis._make_weight_table(rgrid_min, sysX.df_pa, df_pp=sysX.df_pp, mode=weight_mode)
        w_full = base.merge(wtab, on=["y","pt"], how="left")["w"].to_numpy()

        yb = base["y"].to_numpy(); pb = base["pt"].to_numpy()
        mY = (yb >= float(y_min)) & (yb <= float(y_max))
        pts = np.sort(np.unique(pb[mY]))

        # central line
        Rc = []
        for p in pts:
            m = mY & (pb == p)
            if not np.any(m):
                Rc.append(np.nan); continue
            R = r0[m]; w = w_full[m]
            g = (w > 0) & np.isfinite(R)
            Rc.append(float(np.sum(R[g]*w[g]) / np.sum(w[g])) if np.any(g) else np.nan)
        ax.plot(pts, Rc, "-", linewidth=2.0, color=color, label=central_label, zorder=5)

        # members → individual thin lines (faded)
        if M.size:
            for j in range(M.shape[0]):
                Rj_all = []
                for p in pts:
                    m = mY & (pb == p)
                    if not np.any(m):
                        Rj_all.append(np.nan); continue
                    Rj = M[j, m]; w = w_full[m]
                    g = (w > 0) & np.isfinite(Rj)
                    Rj_all.append(float(np.sum(Rj[g]*w[g]) / np.sum(w[g])) if np.any(g) else np.nan)
                ax.plot(pts, np.asarray(Rj_all, float), "-", linewidth=0.8, alpha=0.18, color=color, zorder=3)

    # ===== EXTRA FIGURE (LINES): raw R_pA_mb_k vs y (top: pT thresholds) and vs pT (bottom: y-windows) =====
    fig2 = plt.figure(figsize=(12.5, 9.0), constrained_layout=True)
    sft2, sfb2 = fig2.subfigures(2, 1, height_ratios=[1.0, 1.2])

    # --- TOP: vs y at two pT thresholds ---
    axes2_y = sft2.subplots(1, 2, sharey=True)
    for ax, (ptmin, note) in zip(
        axes2_y,
        [(2.5, r"$p_T \geq 2.5$ GeV"), (1.0, r"$p_T \geq 1.0$ GeV")]
    ):
        _plot_raw_members_vs_y(ax, sys5, ptmin, colors["5.02"], "5.02 TeV")
        _plot_raw_members_vs_y(ax, sys8, ptmin, colors["8.16"], "8.16 TeV")
        style_axes(ax, "y", r"$R_{pA}$", grid=False)
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(0.2, 1.2)
        note_box(ax, note, loc="upper center")
    # single legend for top
    h2t, l2t = axes2_y[0].get_legend_handles_labels()
    sft2.legend(h2t, l2t, loc="upper right", frameon=False, ncol=2, fontsize=11)

    # --- BOTTOM: vs pT in three y-windows (all member lines) ---
    y_windows = [(-1.93, 1.93, "-1.93 < y < 1.93"),
                 ( 1.5 , 4.0 , "1.5 < y < 4.0"),
                 (-5.0 , -2.5, "-5.0 < y < -2.5")]

    axes2_pt = sfb2.subplots(1, len(y_windows), sharex=True, sharey=True)
    if len(y_windows) == 1:
        axes2_pt = [axes2_pt]

    for ax, (ymin, ymax, name) in zip(axes2_pt, y_windows):
        _plot_raw_members_vs_pt(ax, sys5, ymin, ymax, colors["5.02"], "5.02 TeV")
        _plot_raw_members_vs_pt(ax, sys8, ymin, ymax, colors["8.16"], "8.16 TeV")
        style_axes(ax, r"$p_T$ [GeV]", r"$R_{pA}$", grid=False)
        ax.set_xlim(0, 20)
        ax.set_ylim(0.35, 1.25)
        note_box(ax, fr"{name}", loc="lower right")
    # single legend for bottom
    h2b, l2b = axes2_pt[0].get_legend_handles_labels()
    sfb2.legend(h2b, l2b, loc="upper center", frameon=False, ncol=2, fontsize=11)

    fig2.savefig(f"{outdir}/rpa_raw_members_vs_y_ptpanels.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
