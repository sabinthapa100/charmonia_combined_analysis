# -*- coding: utf-8 -*-
"""
npdf_module.py
==============
Concise, single-file module that merges the previously separate helpers:
- io_topdrawer.py
- datasets.py
- analysis.py
- plotting.py
- centrality.py
- utils.py

Public API (stable names preserved)
-----------------------------------
from npdf_module import (
    # datasets & analysis
    NPDFSystem, RpAAnalysis,
    # plotting
    style_axes, step_band_xy, overlay_error_members, slice_nearest_pt_for_each_y,
    band_xy, step_band_from_centers, step_band_from_left_edges, centers_to_left_edges,
    plot_rpa_vs_centrality_hzerr,
    # centrality
    WoodsSaxonPb, CentralityModel, GluonFromGrid, GluonRatioTable,
    # optional TopDrawer reader (alias)
    read_topdrawer, load_top_file, discover_by_number,
    # misc utils
    ensure_dir, ensure_out, round_grid, weighted_average, GridStats,
)

Notes
-----
- Kept behavior of prior modules verbatim where possible.
- Added a tiny compatibility alias `read_topdrawer = load_top_file`.
- Added `plot_rpa_vs_centrality_hzerr` to draw binned centrality points with
  vertical error bars using the table from `CentralityModel.rpa_vs_centrality_integrated`.
"""
from __future__ import annotations

import os, re, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Sequence, Dict, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# utils.py (merged)
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def ensure_out(tag: str) -> str:
    """Create an output folder per collision system/tag (e.g., '5.02', '8.16', 'dAu200')."""
    folder = f"./output-{tag}"
    return ensure_dir(folder)

def round_grid(df: pd.DataFrame, y_dec=3, pt_dec=3) -> pd.DataFrame:
    """Round y/pt to stabilize joins (floating tiny drifts)."""
    out = df.copy()
    out["y_r"] = out["y"].round(y_dec)
    out["pt_r"] = out["pt"].round(pt_dec)
    return out

def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights)
    v = np.asarray(values)
    s = w.sum()
    if s == 0:
        return np.nan
    return float((v * w).sum() / s)

@dataclass
class GridStats:
    nrows: int
    ny: int
    npt: int
    y_min: float
    y_max: float
    pt_min: float
    pt_max: float

    @staticmethod
    def from_df(df: pd.DataFrame) -> "GridStats":
        ys = np.unique(df["y"])
        pts = np.unique(df["pt"])
        return GridStats(
            nrows=len(df),
            ny=len(ys),
            npt=len(pts),
            y_min=float(np.min(ys)),
            y_max=float(np.max(ys)),
            pt_min=float(np.min(pts)),
            pt_max=float(np.max(pts)),
        )

# ---------------------------------------------------------------------------
# io_topdrawer.py (merged)  — FINAL
# ---------------------------------------------------------------------------

# Numeric token usable inside f-strings (DO NOT over-escape in raw strings)
_NUM_RE = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

@dataclass
class TopData:
    """Parsed Topdrawer payload."""
    df: pd.DataFrame               # columns: y, pt, val, err
    y_edges: np.ndarray            # left-step edges for y step-plots
    y_centers: np.ndarray          # center values per y-slab
    header: str                    # first header found (for reference)

def _sections(text: str) -> List[Tuple[int, int]]:
    """
    Return [ (start,end) ] for every data block between 'SET ORDER X Y DY'
    and 'HIST SOLID' lines.
    """
    starts = [m.end() for m in re.finditer(r"SET ORDER X Y DY", text)]
    ends   = [m.end() for m in re.finditer(r"HIST SOLID", text)]
    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n)]

def _y_cut(block_text: str) -> Optional[Tuple[float, float]]:
    # e.g. "-4.0 < y < -3.5" or "2 < y < 2.5"
    m = re.search(rf"({_NUM_RE})\s*<\s*y\s*<\s*({_NUM_RE})", block_text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def _parse_numeric(block_text: str) -> np.ndarray:
    """
    Read rows of: X Y DY  (pt, value, error).
    Ignores non-numeric lines.
    """
    rows = []
    for ln in block_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if not re.match(rf"^{_NUM_RE}", ln):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            vals = [float(parts[0]), float(parts[1]), float(parts[2])]
            rows.append(vals)
        except Exception:
            # ignore malformed numeric lines
            pass
    return np.asarray(rows, dtype=float)

def _pairwise(iterable: Iterable):
    it = iter(iterable)
    return zip(it, it)

def load_top_file(path: str | Path, kick: str = "pp", drop_last_pairs: int = 2) -> TopData:
    """
    Parse a *.top file produced by Ramona's code.

    Parameters
    ----------
    path : str or Path
        Topdrawer file.
    kick : {"pp","dpt"}
        Which of the two consecutive blocks to keep:
        - 'pp'  → first of each pair  (matches Mathematica ptkickOption=1)
        - 'dpt' → second of each pair (pt-kick variant)
    drop_last_pairs : int
        The final two y-slabs are decorative/duplicate in these files and
        produce spikes if kept. Mathematica drops them with [[1;;-3]] —
        do the same here by default.

    Returns
    -------
    TopData
        DataFrame with columns [y, pt, val, err], plus y-edges/centers/header.
    """
    path = Path(path)
    text = path.read_text()

    secs = _sections(text)
    if not secs:
        raise ValueError(f"No data sections found in {path}")

    # Turn contiguous sections into blocks and then into y-slab pairs
    blocks = [text[s:e] for (s, e) in secs]
    pairs = list(_pairwise(blocks))

    # Drop the decorative trailing pairs by default
    if drop_last_pairs:
        pairs = pairs[:-drop_last_pairs]

    # Select first or second of each pair
    sel = 0 if kick.lower() in ("pp", "no", "none") else 1
    chosen = [p[sel] for p in pairs if len(p) == 2]  # guard against odd count

    y_cuts, slabs = [], []
    for blk in chosen:
        yc = _y_cut(blk)
        if yc is None:
            # some files have the y-cut very near the top
            yc = _y_cut(blk[:400])
        if yc is None:
            raise ValueError(f"Could not find y-cut in a block of {path.name}")
        y_cuts.append(yc)

        arr = _parse_numeric(blk)
        if arr.size == 0:
            continue

        y_mean = 0.5 * (yc[0] + yc[1])
        ycol = np.full(arr.shape[0], y_mean, dtype=float)
        slabs.append(np.c_[ycol, arr])  # [y, pt, val, err]

    if not slabs:
        raise ValueError(f"No numeric data parsed from {path}")

    data = np.vstack(slabs)
    df = (pd.DataFrame(data, columns=["y", "pt", "val", "err"])
            .sort_values(["y", "pt"], kind="mergesort")
            .reset_index(drop=True))

    # Keep left-edges in y for step plotting, and centers for convenience
    y_left_edges = np.array([lc for (lc, rc) in y_cuts], dtype=float)
    y_centers    = np.array(sorted(df["y"].unique()), dtype=float)

    # Header/title
    m = re.search(r"^\s*TITLE\s+(.+)$", text, flags=re.MULTILINE)
    header = m.group(1).strip() if m else path.name

    return TopData(df=df, y_edges=y_left_edges, y_centers=y_centers, header=header)

def discover_by_number(folder: str | Path) -> List[Tuple[str, int]]:
    """
    Return [(file_path, set_number)] sorted by the integer after 'e21'
    in the filename, e.g.:
        jpsi_ppb5_e21{NN}diff.top  → NN
        jpsi_ppb8_e21{NN}diff.top  → NN

    This finder is tolerant to any prefix before 'e21' and is case-insensitive.
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

# Backward-compat alias for notebooks that used this name
read_topdrawer = load_top_file


# ---------------------------------------------------------------------------
# datasets.py (merged)
# ---------------------------------------------------------------------------

@dataclass
class NPDFSystem:
    """
    Container for one collision system directory that contains
    - first file:  pp central
    - second file: pA central
    - remaining:   pA error sets (eigenvector members)
    Files are sorted by the integer after 'e21' just like in the Mathematica code.
    """
    pp_path: str
    pa_path: str
    error_paths: List[str]
    kick: str = "pp"          # {"pp", "dpt"}
    name: str = "system"

    # Filled by .load()
    df_pp: pd.DataFrame = field(init=False)
    df_pa: pd.DataFrame = field(init=False)
    df_errors: List[pd.DataFrame] = field(init=False)
    y_edges: Optional[pd.Series] = field(init=False, default=None)
    y_centers: Optional[pd.Series] = field(init=False, default=None)

    def load(self) -> "NPDFSystem":
        pp = load_top_file(self.pp_path, kick=self.kick)
        pa = load_top_file(self.pa_path, kick=self.kick)
        errs = [load_top_file(p, kick=self.kick).df for p in self.error_paths]

        # Keep only necessary columns (and ensure dtypes)
        for df in [pp.df, pa.df] + errs:
            for c in ("y", "pt", "val", "err"):
                assert c in df.columns, f"Missing column {c}"
            df.sort_values(["y", "pt"], inplace=True, kind="mergesort")
            df.reset_index(drop=True, inplace=True)

        self.df_pp = pp.df.copy()
        self.df_pa = pa.df.copy()
        self.df_errors = [d.copy() for d in errs]
        self.y_edges = pd.Series(pp.y_edges)   # step left edges
        self.y_centers = pd.Series(pp.y_centers)
        return self

    @staticmethod
    def from_folder(folder: str, kick: str = "pp", name: str = "system") -> "NPDFSystem":
        files = discover_by_number(folder)
        if len(files) < 2:
            raise ValueError(f"Need at least 2 files (pp central + pA central) in {folder}")
        pp_central = files[0][0]
        pa_central = files[1][0]
        pa_errors  = [p for (p, _) in files[2:]]
        return NPDFSystem(pp_central, pa_central, pa_errors, kick=kick, name=name).load()

# ---------------------------------------------------------------------------
# analysis.py (merged)
# ---------------------------------------------------------------------------

JoinMode = Literal["intersect", "nearest"]

def _nearest_remap(src: pd.DataFrame, target_xy: pd.DataFrame) -> pd.DataFrame:
    """
    Map values from src(y,pt) onto target (y,pt) using nearest-neighbour in (y,pt).
    Avoids SciPy. Complexity is O(N*M) – fine for ~O(2e3).
    """
    a = src[["y", "pt"]].to_numpy()
    b = target_xy[["y", "pt"]].to_numpy()
    val = src["val"].to_numpy()

    res = np.empty(len(b), dtype=float)
    for i, (yy, pp) in enumerate(b):
        d2 = (a[:, 0] - yy) ** 2 + (a[:, 1] - pp) ** 2
        j = int(np.argmin(d2))
        res[i] = val[j]
    out = target_xy.copy()
    out["val"] = res
    return out

@dataclass
class RpAAnalysis:
    """Computation utilities to reproduce the Mathematica workflow."""

    # ---------- 3) RpA grid with error band ----------
    def compute_rpa_grid(
        self,
        df_pp: pd.DataFrame,
        df_pa: pd.DataFrame,
        df_errors: List[pd.DataFrame],
        join: JoinMode = "intersect"
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
        [y, pt, r_central, r_lo, r_hi],
        where r_lo/hi are the Hessian-like +/- bands computed from error members
        (sum in quadrature of positive/negative deviations).
        """

        # Base grid: start from pA central grid
        base = df_pa[["y", "pt"]].drop_duplicates()

        if join == "intersect":
            # Single intersection across *all* sets up-front
            base_all = base.merge(df_pp[["y", "pt"]].drop_duplicates(), on=["y", "pt"], how="inner")
            for dfe in df_errors:
                base_all = base_all.merge(dfe[["y", "pt"]].drop_duplicates(), on=["y", "pt"], how="inner")
            base = base_all.sort_values(["y", "pt"]).reset_index(drop=True)

            pa_aligned = base.merge(df_pa[["y", "pt", "val"]], on=["y", "pt"], how="inner")["val"].to_numpy()
            pp_aligned = base.merge(df_pp[["y", "pt", "val"]], on=["y", "pt"], how="inner")["val"].to_numpy()
            r0 = pa_aligned / pp_aligned

            mems: List[np.ndarray] = []
            for dfe in df_errors:
                tmp = base.merge(dfe[["y", "pt", "val"]], on=["y", "pt"], how="inner")["val"].to_numpy()
                mems.append(tmp / pp_aligned)

        else:  # join == "nearest"
            base = base.sort_values(["y", "pt"]).reset_index(drop=True)
            pp_aligned = _nearest_remap(df_pp[["y", "pt", "val"]], base)["val"].to_numpy()
            pa_aligned = _nearest_remap(df_pa[["y", "pt", "val"]], base)["val"].to_numpy()
            r0 = pa_aligned / pp_aligned

            mems: List[np.ndarray] = []
            for dfe in df_errors:
                tmp = _nearest_remap(dfe[["y", "pt", "val"]], base)["val"].to_numpy()
                mems.append(tmp / pp_aligned)

        # Hessian-style +/- errors
        if mems:
            M = np.stack(mems, axis=0)          # [nMembers, nPoints]
            diff = M - r0[None, :]              # deviations from central
            pos = np.maximum(diff, 0.0)
            neg = np.minimum(diff, 0.0)
            err_plus  = np.sqrt(np.sum(pos * pos, axis=0))
            err_minus = np.sqrt(np.sum(neg * neg, axis=0))
        else:
            err_plus = np.zeros_like(r0)
            err_minus = np.zeros_like(r0)

        out = base.copy()
        out["r_central"] = r0
        out["r_lo"] = r0 - err_minus
        out["r_hi"] = r0 + err_plus
        return out.sort_values(["y", "pt"]).reset_index(drop=True)

    # ---------- helpers for weighted averages ----------
    @staticmethod
    def _weighted_avg(series: np.ndarray, weights: np.ndarray) -> float:
        w = np.asarray(weights, float)
        x = np.asarray(series, float)
        m = (w > 0) & np.isfinite(x)
        if not np.any(m):
            return np.nan
        return float(np.sum(x[m] * w[m]) / np.sum(w[m]))

    # ---------- RpA vs y for pT >= threshold ----------
    def rpa_vs_y_pt_threshold(self, rgrid: pd.DataFrame, pa_central: pd.DataFrame, pt_min: float) -> pd.DataFrame:
        """Weighted by σ_pA(y,pt) as in the Mathematica notebook."""
        r = rgrid.copy()
        pdf = pa_central[["y", "pt", "val"]].rename(columns={"val": "w"})
        merged = r.merge(pdf, on=["y", "pt"], how="inner")
        merged = merged[merged["pt"] >= pt_min].copy()

        rows = []
        for y, g in merged.groupby("y", sort=True):
            rows.append({
                "y": float(y),
                "r_central": self._weighted_avg(g["r_central"], g["w"]),
                "r_lo":      self._weighted_avg(g["r_lo"],      g["w"]),
                "r_hi":      self._weighted_avg(g["r_hi"],      g["w"]),
            })
        return pd.DataFrame(rows).sort_values("y").reset_index(drop=True)

    # ---------- RpA vs pT in a rapidity window ----------
    def rpa_vs_pt_in_y(self, rgrid: pd.DataFrame, pa_central: pd.DataFrame, y_min: float, y_max: float) -> pd.DataFrame:
        r = rgrid[(rgrid["y"] >= y_min) & (rgrid["y"] <= y_max)].copy()
        pdf = pa_central[(pa_central["y"] >= y_min) & (pa_central["y"] <= y_max)][["y", "pt", "val"]]
        pdf = pdf.rename(columns={"val": "w"})
        merged = r.merge(pdf, on=["y", "pt"], how="inner")

        rows = []
        for pt, g in merged.groupby("pt", sort=True):
            rows.append({
                "pt": float(pt),
                "r_central": self._weighted_avg(g["r_central"], g["w"]),
                "r_lo":      self._weighted_avg(g["r_lo"],      g["w"]),
                "r_hi":      self._weighted_avg(g["r_hi"],      g["w"]),
            })
        return pd.DataFrame(rows).sort_values("pt").reset_index(drop=True)

    # ---------- Coarse binning along pT ----------
    def coarse_bin_along_pt(self, rgrid: pd.DataFrame, pa_central: pd.DataFrame, block_size: int = 5
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reproduce the Mathematica 'size=5' grouping: flatten by (y,pt),
        then average in consecutive blocks, weighted by σ_pA.
        Returns:
          - binned RpA: columns [y_bar, pt_bar, r_central, r_lo, r_hi]
          - binned XSC: columns [y_bar, pt_bar, sigma]
        """
        pdf = pa_central[["y", "pt", "val"]].rename(columns={"val": "w"})
        merged = rgrid.merge(pdf, on=["y", "pt"], how="inner").sort_values(["y", "pt"]).reset_index(drop=True)

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

        rpa = binned[["y_bar", "pt_bar", "r_central", "r_lo", "r_hi"]].copy()
        xsc = binned[["y_bar", "pt_bar", "sigma"]].copy()
        return rpa, xsc

    # ---------- Wide pT bands by GeV width ----------
    def rpa_vs_pt_widebins(self, rgrid: pd.DataFrame, pa_central: pd.DataFrame, y_min: float, y_max: float, width: float = 2.5
                           ) -> pd.DataFrame:
        """
        Average R_pA over [y_min,y_max] inside fixed pT bins of given width (GeV),
        weights = σ_pA, and return LEFT edges for step plotting.
        Columns: [pt_left, r_central, r_lo, r_hi]
        """
        r = rgrid[(rgrid["y"] >= y_min) & (rgrid["y"] <= y_max)].copy()
        pdf = pa_central[(pa_central["y"] >= y_min) & (pa_central["y"] <= y_max)][["y", "pt", "val"]]
        pdf = pdf.rename(columns={"val": "w"})
        merged = r.merge(pdf, on=["y", "pt"], how="inner")

        if merged.empty:
            return pd.DataFrame(columns=["pt_left", "r_central", "r_lo", "r_hi"])

        pts = np.sort(merged["pt"].unique())
        pmin, pmax = float(pts.min()), float(pts.max())
        start = width * np.floor(pmin / width)
        stop  = width * np.ceil(pmax / width) + 1e-9
        edges = np.arange(start, stop, width)

        out_rows = []
        for i in range(len(edges)):
            left = edges[i]
            right = edges[i] + width
            g = merged[(merged["pt"] >= left) & (merged["pt"] < right)]
            if len(g) == 0:
                continue
            w = g["w"].to_numpy()
            out_rows.append({
                "pt_left": left,
                "r_central": self._weighted_avg(g["r_central"].to_numpy(), w),
                "r_lo":      self._weighted_avg(g["r_lo"].to_numpy(),      w),
                "r_hi":      self._weighted_avg(g["r_hi"].to_numpy(),      w),
            })
        return pd.DataFrame(out_rows).sort_values("pt_left").reset_index(drop=True)

# ---------------------------------------------------------------------------
# plotting.py (merged)
# ---------------------------------------------------------------------------

def style_axes(ax, xlab, ylab, grid: bool=True, logx: bool=False, logy: bool=False, title: Optional[str]=None):
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    if grid:
        ax.grid(True, which="both", alpha=0.25)
    if title:
        ax.set_title(title)

def step_band_xy(ax, x, y_c, y_lo, y_hi, label: Optional[str]=None, color=None):
    """Draw a step band (like Mathematica ListStepPlot with filling).
    Ensures the band color matches the midline color even when `color=None`.
    """
    # draw the line first and read back the resolved color
    lines = ax.step(x, y_c, where="post", label=label, color=color, linewidth=2)
    line_color = color if color is not None else lines[0].get_color()
    # fill with exactly the same color
    ax.fill_between(x, y_lo, y_hi, step="post", alpha=0.25, color=line_color)

def overlay_error_members(ax, xs, members: np.ndarray, color=None, alpha: float=0.12, lw: float=1.0):
    """Overlay thin member curves (each row = a member)."""
    if members.ndim != 2:
        return
    for i in range(members.shape[0]):
        ax.plot(xs, members[i], color=color, alpha=alpha, lw=lw)

def slice_nearest_pt_for_each_y(df: pd.DataFrame, pt_target: float) -> pd.DataFrame:
    """
    For a given target pt, pick for each y the row whose pt is nearest to pt_target.
    This eliminates 'spiky' artifacts from picking mismatched rows.
    """
    out = []
    for y, g in df.groupby("y"):
        j = (g["pt"] - pt_target).abs().idxmin()
        out.append(df.loc[j])
    return pd.DataFrame(out).sort_values("y")

def band_xy(ax, x, y_c, y_lo, y_hi, label: Optional[str]=None, color=None):
    """
    Regular (non-step) band: good for pT where points represent bin centers.
    Ensures the band color matches the midline color even when `color=None`.
    """
    x = np.asarray(x, float)
    order = np.argsort(x)
    x   = x[order]
    y_c = np.asarray(y_c, float)[order]
    y_lo = np.asarray(y_lo, float)[order]
    y_hi = np.asarray(y_hi, float)[order]

    # draw the line first and read back the resolved color
    (line_obj,) = ax.plot(x, y_c, lw=2, label=label, color=color)
    line_color = color if color is not None else line_obj.get_color()
    # fill with exactly the same color
    ax.fill_between(x, y_lo, y_hi, alpha=0.25, color=line_color)

def centers_to_left_edges(centers: np.ndarray) -> np.ndarray:
    """
    Convert unevenly spaced centers -> left edges for step='post'.
    Length is the same as centers (last right-edge is implicit).
    """
    c = np.asarray(centers, float)
    c = np.unique(np.sort(c))
    if len(c) < 2:
        return c.copy()
    left = np.empty_like(c)
    left[0]  = c[0] - 0.5*(c[1] - c[0])
    left[1:] = 0.5*(c[:-1] + c[1:])
    return left

def _centers_to_left_edges(centers):
    c = np.unique(np.asarray(centers, float))
    if len(c) < 2:
        return c.copy()
    left = np.empty_like(c)
    left[0]  = c[0] - 0.5*(c[1]-c[0])
    left[1:] = 0.5*(c[:-1] + c[1:])
    return left

def step_band_from_centers(ax, x_centers, y_c, y_lo, y_hi, **kwargs):
    """Convert bin centers -> left edges, then call step_band_xy."""
    x_left = _centers_to_left_edges(x_centers)
    return step_band_xy(ax, x_left, y_c, y_lo, y_hi, **kwargs)

def step_band_from_left_edges(ax, left_edges, y_c, y_lo, y_hi, **kwargs):
    """For wide pT bands (fixed bin width), we already have left edges. Call step-band helper."""
    return step_band_xy(ax, np.asarray(left_edges, float), y_c, y_lo, y_hi, **kwargs)

def _parse_centbin(s: str) -> Tuple[float,float]:
    """Parse '0-5%' -> (0.0, 5.0)."""
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*%", str(s))
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))

def plot_rpa_vs_centrality_hzerr(ax, df: pd.DataFrame, label: Optional[str]=None, color=None, markers: bool=True):
    """
    Plot R_pA vs centrality bins with vertical error bars at bin centers.
    Expects columns: ['cent_bin','r_central','r_lo','r_hi'] produced by
    CentralityModel.rpa_vs_centrality_integrated(...).
    """
    if df is None or len(df)==0:
        return ax
    # Build x as bin centers in percent
    edges = np.array([_parse_centbin(s) for s in df["cent_bin"].tolist()], dtype=float)
    x = 0.5*(edges[:,0] + edges[:,1])
    y = df["r_central"].to_numpy(float)
    ylo = df["r_lo"].to_numpy(float)
    yhi = df["r_hi"].to_numpy(float)
    yerr = np.vstack([y - ylo, yhi - y])

    ax.errorbar(x, y, yerr=yerr, fmt="o" if markers else None, capsize=3, label=label, color=color)
    ax.plot(x, y, lw=2, color=color)
    ax.set_xlim(left=min(x)-2.5, right=max(x)+2.5)
    return ax

# ---------------------------------------------------------------------------
# centrality.py (merged)
# ---------------------------------------------------------------------------

def _searchsorted_clamped(arr: np.ndarray, x: float) -> int:
    i = np.searchsorted(arr, x) - 1
    return int(np.clip(i, 0, len(arr) - 2))

class Bilinear2D:
    def __init__(self, x_grid: np.ndarray, q_grid: np.ndarray, V_qx: np.ndarray):
        assert V_qx.shape == (len(q_grid), len(x_grid))
        self.x = np.array(x_grid, dtype=float)
        self.q = np.array(q_grid, dtype=float)
        self.V = np.array(V_qx, dtype=float)

    def __call__(self, x: float, q: float) -> float:
        x = float(np.clip(x, self.x[0], self.x[-1]))
        q = float(np.clip(q, self.q[0], self.q[-1]))
        i = _searchsorted_clamped(self.x, x)
        j = _searchsorted_clamped(self.q, q)
        x1, x2 = self.x[i], self.x[i+1]
        q1, q2 = self.q[j], self.q[j+1]
        fx = 0.0 if x2 == x1 else (x - x1) / (x2 - x1)
        fq = 0.0 if q2 == q1 else (q - q1) / (q2 - q1)
        v11 = self.V[j,   i  ]
        v21 = self.V[j,   i+1]
        v12 = self.V[j+1, i  ]
        v22 = self.V[j+1, i+1]
        return ((1-fx)*(1-fq)*v11 + fx*(1-fq)*v21 + (1-fx)*fq*v12 + fx*fq*v22)

def _read_xq_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        engine="python",
        sep=r"[\\s,]+",
        dtype=str
    )
    df = df.dropna(how="all")
    if df.shape[1] < 3:
        raise RuntimeError(f"{path} does not look like 3-column x Q S table.")
    df = df.iloc[:, :3]
    df.columns = ["x", "Qcol", "S"]
    for col in ("x", "Qcol", "S"):
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r"[;,]+$", "", regex=True)
        df[col] = df[col].str.replace("D", "E", regex=False)
    for col in ("x", "Qcol", "S"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{path} parsed to an empty numeric table after cleaning.")
    return df

@dataclass
class GluonRatioTable:
    path: Path
    sqrt_sNN_GeV: float
    m_jpsi_GeV: float = 3.548
    q_is_q2: Optional[bool] = True
    y_sign_for_xA: int = -1
    _interp: Optional[Bilinear2D] = None
    _x_grid: Optional[np.ndarray] = None
    _q_grid: Optional[np.ndarray] = None

    def load(self) -> "GluonRatioTable":
        df = _read_xq_table(self.path)
        # Drop non-physical/negative S entries
        df = df[df["S"] > 0].copy()

        if self.q_is_q2 is None:
            qvals = np.sort(df["Qcol"].unique())
            self.q_is_q2 = (np.median(np.diff(qvals)) > 1.0)

        x_vals = np.sort(df["x"].unique())
        q_vals = np.sort(df["Qcol"].unique())
        V = (
            df.pivot(index="Qcol", columns="x", values="S")
              .reindex(index=q_vals, columns=x_vals)
              .sort_index()
              .sort_index(axis=1)
        )
        V = V.interpolate(axis=0, limit_direction="both") \
             .interpolate(axis=1, limit_direction="both")
        V_np = V.to_numpy(dtype=float)

        self._interp = Bilinear2D(x_vals, q_vals, V_np)
        self._x_grid, self._q_grid = x_vals, q_vals

        Smin, Smax = np.nanmin(V_np), np.nanmax(V_np)
        if not np.isfinite(Smin) or not np.isfinite(Smax):
            raise RuntimeError(f"{self.path} produced non-finite S_A grid.")
        if Smin <= 0:
            print(f"[GluonRatioTable] WARNING: S_A grid has values ≤ 0 (min={Smin:.3g}). Will clip at evaluation.")
        return self

    # ---- kinematics & evaluation ----
    def x_of(self, y: float, pT: float) -> float:
        mT = math.hypot(self.m_jpsi_GeV, pT)
        return (2.0 * mT / self.sqrt_sNN_GeV) * math.exp(self.y_sign_for_xA * y)

    def Qgrid_of(self, pT: float) -> float:
        mT = math.hypot(self.m_jpsi_GeV, pT)
        return mT*mT if self.q_is_q2 else mT

    def SA_ypt(self, y: float, pT: float) -> float:
        assert self._interp is not None, "Call .load() first"
        x = self.x_of(y, pT)
        q = self.Qgrid_of(pT)
        S = float(self._interp(x, q))
        return max(S, 1e-8)

    # ---- quick grids for plotting ----
    def grid_SA_xQ(self, nx: int = 120, nq: int = 120) -> pd.DataFrame:
        assert self._x_grid is not None and self._q_grid is not None
        x0, x1 = float(self._x_grid[0]), float(self._x_grid[-1])
        q0, q1 = float(self._q_grid[0]), float(self._q_grid[-1])
        xs = np.geomspace(max(x0, 1e-8), x1, nx) if x0 > 0 else np.linspace(x0, x1, nx)
        qs = np.linspace(q0, q1, nq)
        X, Q = np.meshgrid(xs, qs)
        S = np.vectorize(lambda a,b: float(self._interp(a,b)))(X, Q)
        return pd.DataFrame({"x": X.ravel(), "Qgrid": Q.ravel(), "SA": S.ravel()})

    def grid_SA_ypt(self, y_edges: Sequence[float], pt_edges: Sequence[float], samples_per_bin: Tuple[int,int]=(5,5)) -> pd.DataFrame:
        yc = 0.5*(np.array(y_edges[:-1]) + np.array(y_edges[1:]))
        pc = 0.5*(np.array(pt_edges[:-1]) + np.array(pt_edges[1:]))
        rows = []
        for i in range(len(yc)):
            yl, yr = y_edges[i], y_edges[i+1]
            ys = np.linspace(yl, yr, max(2, samples_per_bin[0]))
            for j in range(len(pc)):
                pl, pr = pt_edges[j], pt_edges[j+1]
                ps = np.linspace(pl, pr, max(2, samples_per_bin[1]))
                Y, P = np.meshgrid(ys, ps)
                SA = np.vectorize(self.SA_ypt)(Y, P)
                rows.append({
                    "y": float(yc[i]),
                    "pt": float(pc[j]),
                    "y_left": float(yl), "y_right": float(yr),
                    "pt_left": float(pl), "pt_right": float(pr),
                    "SA": float(np.mean(SA))
                })
        return pd.DataFrame(rows)

@dataclass
class GluonFromGrid:
    df_pp: pd.DataFrame
    df_pa: pd.DataFrame

    def make_SA_on_xy(self) -> pd.DataFrame:
        a = self.df_pp[["y","pt","val"]].rename(columns={"val":"pp"})
        b = self.df_pa[["y","pt","val"]].rename(columns={"val":"pa"})
        m = a.merge(b, on=["y","pt"], how="inner")
        m["SA"] = np.where(m["pp"] > 0, m["pa"]/m["pp"], np.nan)
        return m[["y","pt","SA"]]

    def SA_ypt(self, y: float, pT: float) -> float:
        gpp = self.df_pp[["y","pt","val"]].to_numpy()
        d2 = (gpp[:,0]-y)**2 + (gpp[:,1]-pT)**2
        j = int(np.argmin(d2))
        m = self.make_SA_on_xy()
        sub = m[(m["y"]==gpp[j,0]) & (m["pt"]==gpp[j,1])]
        if not sub.empty and np.isfinite(sub["SA"].values[0]):
            return float(sub["SA"].values[0])
        return 1.0

@dataclass
class WoodsSaxonPb:
    A: int = 208
    R: float = 6.624
    a: float = 0.549
    rho0: float = 0.17
    zmax_mult: float = 10.0
    nz: int = 4001

    def rho(self, r: float) -> float:
        return self.rho0 / (1.0 + math.exp((r - self.R)/self.a))

    def thickness(self, b: float) -> float:
        zmax = self.zmax_mult * self.R
        z = np.linspace(-zmax, zmax, self.nz)
        r = np.hypot(b, z)
        rho_line = self.rho0 / (1.0 + np.exp((r - self.R)/self.a))
        return float(np.trapezoid(rho_line, z))

    def make_T_grid(self, b_max: float = 12.0, nb: int = 601) -> pd.DataFrame:
        b = np.linspace(0.0, b_max, nb)
        T = np.array([self.thickness(float(bi)) for bi in b])
        return pd.DataFrame({"b": b, "T": T})

    def normalization_N(self, T_grid: pd.DataFrame) -> float:
        b = T_grid["b"].to_numpy()
        T = T_grid["T"].to_numpy()
        T0 = float(T[0])
        I_T2 = float(np.trapezoid(2.0*math.pi * b * (T**2), b))
        return (self.A * T0) / I_T2

    @staticmethod
    def sigma_mb_to_fm2(sigma_mb: float) -> float:
        return 0.1 * float(sigma_mb)

    def P_inel(self, T_b: float, sigmaNN_mb: float = 70.0) -> float:
        s = self.sigma_mb_to_fm2(sigmaNN_mb)
        return 1.0 - math.exp(- s * T_b)

    def Npart_pA(self, T_b: float, sigmaNN_mb: float = 70.0) -> float:
        s = self.sigma_mb_to_fm2(sigmaNN_mb)
        tgt = self.A * (1.0 - math.exp(-s*T_b/self.A))
        return 1.0 + tgt

    def b_pdf(self, T_grid: pd.DataFrame, sigmaNN_mb: float = 70.0) -> pd.DataFrame:
        b = T_grid["b"].to_numpy()
        T = T_grid["T"].to_numpy()
        pinel = np.array([self.P_inel(Ti, sigmaNN_mb) for Ti in T], dtype=float)
        w = 2.0 * math.pi * b * pinel
        Z = float(np.trapezoid(w, b))
        if Z > 0: w = w / Z
        return pd.DataFrame({"b": b, "T": T, "P_inel": pinel, "w": w})

    def b_edges_for_percentiles(self, T_grid: pd.DataFrame, percentiles: Sequence[float], sigmaNN_mb: float = 70.0
                                ) -> np.ndarray:
        pdf = self.b_pdf(T_grid, sigmaNN_mb=sigmaNN_mb)
        b = pdf["b"].to_numpy(); w = pdf["w"].to_numpy()
        cdf = np.cumsum((w[:-1] + w[1:]) * 0.5 * np.diff(b))
        cdf = np.r_[0.0, cdf]
        if cdf[-1] > 0: cdf = cdf / cdf[-1]
        out = []
        for p in percentiles:
            t = float(p) / 100.0
            j = int(np.searchsorted(cdf, t, side="left"))
            if j <= 0:
                out.append(float(b[0])); continue
            if j >= len(b):
                out.append(float(b[-1])); continue
            denom = (cdf[j] - cdf[j-1])
            frac = 0.0 if denom == 0 else (t - cdf[j-1]) / denom
            out.append(float((1.0-frac)*b[j-1] + frac*b[j]))
        return np.asarray(out, dtype=float)

@dataclass
class CentralityModel:
    """
    RpA(y,pT;b) = [1 + N (S_A-1) T(b)/T(0)] / S_A × RpA_inclusive(y,pT).
    """
    gluon: object
    geom: WoodsSaxonPb
    T_grid: pd.DataFrame
    Nnorm: float

    _pdf_cache: Optional[pd.DataFrame] = None
    _alpha_cache: Dict[Tuple[Tuple[int,...], float, str], List[Tuple[str,float]]] = None

    @classmethod
    def from_inputs(cls, gluon: object, geom: WoodsSaxonPb, b_max: float = 12.0, nb: int = 601, N_override: Optional[float] = None
                    ) -> "CentralityModel":
        T_grid = geom.make_T_grid(b_max=b_max, nb=nb)
        Nnorm = geom.normalization_N(T_grid) if N_override is None else float(N_override)
        return cls(gluon=gluon, geom=geom, T_grid=T_grid, Nnorm=Nnorm, _pdf_cache=None, _alpha_cache={})

    # ---------- fast precomputations ----------
    def pdf_table(self, sigmaNN_mb: float) -> pd.DataFrame:
        if (self._pdf_cache is None
            or "sigmaNN_mb" not in self._pdf_cache.attrs
            or self._pdf_cache.attrs["sigmaNN_mb"] != sigmaNN_mb):
            pdf = self.geom.b_pdf(self.T_grid, sigmaNN_mb=sigmaNN_mb)
            pdf.attrs["sigmaNN_mb"] = sigmaNN_mb
            self._pdf_cache = pdf
        return self._pdf_cache

    def _alphas_for_bins(self, cent_edges_pct: Sequence[float], sigmaNN_mb: float = 70.0, weight: str = "inelastic"
                         ) -> List[Tuple[str, float]]:
        key = (tuple(map(int, cent_edges_pct)), float(sigmaNN_mb), weight)
        if self._alpha_cache and key in self._alpha_cache:
            return self._alpha_cache[key]

        pdf = self.pdf_table(sigmaNN_mb)
        b = pdf["b"].to_numpy(); T = pdf["T"].to_numpy(); T0 = float(T[0])
        W = T if weight == "thickness" else pdf["w"].to_numpy()

        edges_b = self.geom.b_edges_for_percentiles(self.T_grid, cent_edges_pct, sigmaNN_mb=sigmaNN_mb)
        out = []
        for i in range(len(edges_b)-1):
            bl, br = edges_b[i], edges_b[i+1]
            m = (b >= bl) & (b < br)
            if not np.any(m):
                out.append((f"{cent_edges_pct[i]}-{cent_edges_pct[i+1]}%", np.nan))
                continue
            num = float(np.trapezoid((T[m]/T0)*W[m], b[m]))
            den = float(np.trapezoid(W[m], b[m]))
            alpha = num/den if den > 0 else np.nan
            out.append((f"{cent_edges_pct[i]}-{cent_edges_pct[i+1]}%", alpha))
        self._alpha_cache[key] = out
        return out

    def centrality_table(self, cent_edges_pct: Sequence[float], sigmaNN_mb: float = 70.0) -> pd.DataFrame:
        pdf = self.pdf_table(sigmaNN_mb)
        b = pdf["b"].to_numpy(); T = pdf["T"].to_numpy(); w = pdf["w"].to_numpy()
        T0 = float(T[0])
        edges_b = self.geom.b_edges_for_percentiles(self.T_grid, cent_edges_pct, sigmaNN_mb=sigmaNN_mb)
        rows = []
        for i in range(len(edges_b)-1):
            bl, br = float(edges_b[i]), float(edges_b[i+1])
            m = (b >= bl) & (b < br)
            if not np.any(m):
                continue
            den = float(np.trapezoid(w[m], b[m]))
            bbar = float(np.trapezoid(b[m]*w[m], b[m]) / den) if den>0 else np.nan
            alpha = float(np.trapezoid((T[m]/T0)*w[m], b[m]) / den) if den>0 else np.nan
            Np = np.array([self.geom.Npart_pA(Ti, sigmaNN_mb) for Ti in T[m]], float)
            Np_bar = float(np.trapezoid(Np*w[m], b[m]) / den) if den>0 else np.nan
            # <N_coll> = <sigmaNN * T(b)> with sigma converted to fm^2
            s_fm2 = self.geom.sigma_mb_to_fm2(sigmaNN_mb)
            Ncoll_bar = float(np.trapezoid((s_fm2*T[m])*w[m], b[m]) / den) if den>0 else np.nan
            rows.append({
                "cent_bin": f"{cent_edges_pct[i]}-{cent_edges_pct[i+1]}%",
                "b_left": bl, "b_right": br, "b_mean": bbar,
                "alpha": alpha, "N_part": Np_bar, "N_coll": Ncoll_bar
            })
        return pd.DataFrame(rows)

    # ---------- basic S_A attachment ----------
    def attach_SA_to_grid(self, rgrid: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        rg = rgrid.copy()
        if hasattr(self.gluon, "make_SA_on_xy"):
            tbl = self.gluon.make_SA_on_xy()
            def _round(df):
                out = df.copy()
                out["y_r"]  = out["y"].round(3)
                out["pt_r"] = out["pt"].round(3)
                return out
            rg2  = _round(rg); tbl2 = _round(tbl)
            rg2 = rg2.merge(tbl2[["y_r","pt_r","SA"]], on=["y_r","pt_r"], how="left")
            miss = int(rg2["SA"].isna().sum())
            if verbose and miss:
                print(f"[attach_SA] WARNING: {miss} points missing SA; filling with 1.0")
            rg2["SA"] = rg2["SA"].fillna(1.0)
            return rg2.drop(columns=["y_r","pt_r"])
        else:
            # vectorized SA on (y,pt)
            YP = rg[["y","pt"]].to_numpy(dtype=float)
            SA = np.array([float(self.gluon.SA_ypt(y, p)) for y, p in YP])

            # set SA=1.0 for (x,Q) outside the S_A table domain
            if hasattr(self.gluon, "_x_grid") and self.gluon._x_grid is not None:
                xg = self.gluon._x_grid; qg = self.gluon._q_grid
                xv = np.array([self.gluon.x_of(y, p)  for y, p in YP])
                qv = np.array([self.gluon.Qgrid_of(p) for _, p in YP])
                inside = (xv >= xg[0]) & (xv <= xg[-1]) & (qv >= qg[0]) & (qv <= qg[-1])
                SA = np.where(inside & np.isfinite(SA) & (SA > 0), SA, 1.0)

            rg["SA"] = SA
            return rg

    # ---------- pointwise α(b), S_AWS ----------
    def alpha_of_b(self, b_val: float) -> float:
        b = self.T_grid["b"].to_numpy()
        T = self.T_grid["T"].to_numpy()
        T0 = float(T[0])
        Tb = float(np.interp(b_val, b, T))
        return Tb / T0 if T0 > 0 else 0.0

    def SAWS_ypt_b(self, y: float, pT: float, b_val: float) -> float:
        SA = float(self.gluon.SA_ypt(y, pT))
        alpha = self.alpha_of_b(b_val)
        return 1.0 + self.Nnorm * (SA - 1.0) * alpha

    # ---------- scaling by α (centrality bins) ----------
    def _scale_grid_by_alpha(self, rg_with_SA: pd.DataFrame, alpha: float) -> pd.DataFrame:
        SA = rg_with_SA["SA"].to_numpy()
        SA_safe = np.clip(SA, 1e-12, None)
        K = (1.0 + self.Nnorm * (SA - 1.0) * float(alpha)) / SA_safe

        out = rg_with_SA.copy()
        for col in ("r_central", "r_lo", "r_hi"):
            if col in out.columns:
                out[col] = out[col].to_numpy() * K

        # enforce ordered envelopes so errorbars are always non-negative
        if set(["r_central","r_lo","r_hi"]).issubset(out.columns):
            rc = out["r_central"].to_numpy()
            lo = out["r_lo"].to_numpy()
            hi = out["r_hi"].to_numpy()
            out["r_lo"] = np.minimum.reduce([lo, rc, hi])
            out["r_hi"] = np.maximum.reduce([lo, rc, hi])

        return out

    # ---------- R_pA vs impact parameter b ----------
    def rpa_vs_b(self, rgrid: pd.DataFrame, df_pa: pd.DataFrame, y_min: float, y_max: float, pt_min: float, pt_max: float,
                 b_values: Optional[Sequence[float]] = None, sigmaNN_mb: float = 70.0, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print(f"[rpa_vs_b] window: y∈[{y_min},{y_max}], pT∈[{pt_min},{pt_max}]")

        rg = self.attach_SA_to_grid(rgrid, verbose=False)
        m = (rg["y"]>=y_min)&(rg["y"]<=y_max)&(rg["pt"]>=pt_min)&(rg["pt"]<=pt_max)
        rg = rg.loc[m].copy()
        if rg.empty:
            if verbose: print("  -> no points in window; returning empty DataFrame.")
            return pd.DataFrame(columns=["b","alpha","N_part","r_central","r_lo","r_hi"])

        wtab = df_pa[["y","pt","val"]].rename(columns={"val":"w"})
        rg = rg.merge(wtab, on=["y","pt"], how="left")
        rg["w"] = rg["w"].clip(lower=0).fillna(0.0)

        if b_values is None:
            b_values = self.T_grid["b"].to_numpy()

        SA = rg["SA"].to_numpy()
        SA_safe = np.clip(SA, 1e-12, None)
        base = {k: rg[k].to_numpy() for k in ("r_central","r_lo","r_hi") if k in rg.columns}
        w = rg["w"].to_numpy()

        rows = []
        for bv in b_values:
            alpha = self.alpha_of_b(float(bv))
            K = (1.0 + self.Nnorm * (SA - 1.0) * alpha) / SA_safe
            out_vals = {}
            for k, arr in base.items():
                v = arr * K
                s = w.sum()
                out_vals[k] = float((v*w).sum()/s) if s>0 else np.nan
            Tb = float(np.interp(bv, self.T_grid["b"].to_numpy(), self.T_grid["T"].to_numpy()))
            Np = self.geom.Npart_pA(Tb, sigmaNN_mb=sigmaNN_mb)
            rows.append({"b": float(bv), "alpha": float(alpha), "N_part": float(Np),
                         "r_central": out_vals.get("r_central", np.nan),
                         "r_lo": out_vals.get("r_lo", np.nan),
                         "r_hi": out_vals.get("r_hi", np.nan)})
        return pd.DataFrame(rows)

    # ---------- public physics helpers (bins & averages) ----------
    def rpa_vs_centrality_integrated(self, rgrid: pd.DataFrame, df_pa: pd.DataFrame, centrality_edges_pct: Sequence[float],
                                     y_min: float, y_max: float, pt_min: float = 0.0, pt_max: float = 20.0,
                                     sigmaNN_mb: float = 70.0, weight: str = "inelastic",
                                     verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print(f"[rpa_vs_centrality_integrated] window: y∈[{y_min},{y_max}], pT∈[{pt_min},{pt_max}]")
        rg = self.attach_SA_to_grid(rgrid, verbose=False)
        m = (rg["y"]>=y_min)&(rg["y"]<=y_max)&(rg["pt"]>=pt_min)&(rg["pt"]<=pt_max)
        rgw = rg.loc[m].copy()
        if rgw.empty:
            if verbose: print("  -> no points in window; returning empty DataFrame.")
            return pd.DataFrame(columns=["cent_bin","r_central","r_lo","r_hi"])

        wtab = df_pa[["y","pt","val"]].rename(columns={"val":"w"})
        rgw = rgw.merge(wtab, on=["y","pt"], how="left")
        rgw["w"] = rgw["w"].clip(lower=0).fillna(0.0)

        alphas = self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight)
        rows = []
        for label, alpha in alphas:
            if not np.isfinite(alpha):
                if verbose: print(f"  [warn] alpha NaN for bin {label}; skipping.")
                continue
            g = self._scale_grid_by_alpha(rgw, alpha)
            w = g["w"].to_numpy()
            def _wavg(col):
                v = g[col].to_numpy()
                s = w.sum()
                return float((v*w).sum()/s) if s>0 else np.nan
            rows.append({"cent_bin":label,
                         "r_central": _wavg("r_central"),
                         "r_lo":      _wavg("r_lo"),
                         "r_hi":      _wavg("r_hi")})
            if verbose:
                print(f"  bin {label}: alpha={alpha:.6f}, points={len(g)}, wsum={w.sum():.3e}")
        return pd.DataFrame(rows)

    def rpa_vs_y_in_centrality_bins(self, rgrid: pd.DataFrame, df_pa: pd.DataFrame, centrality_edges_pct: Sequence[float],
                                    y_width: float = 0.5, pt_min: Optional[float] = 0.0, pt_max: Optional[float] = None,
                                    sigmaNN_mb: float = 70.0, weight: str = "inelastic", verbose: bool = True,
                                    pt_guard: Optional[float] = None, K_cap: Optional[float] = None
                                    ) -> List[Tuple[str, pd.DataFrame]]:
        rg = self.attach_SA_to_grid(rgrid, verbose=False)
        df_pa_w = df_pa[["y","pt","val"]].rename(columns={"val":"w"})
        alphas = self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight)
        out = []
        for label, alpha in alphas:
            if not np.isfinite(alpha):
                if verbose: print(f"  [warn] alpha NaN for bin {label}; skipping.")
                continue
            g = self._scale_grid_by_alpha(rg, alpha).merge(df_pa_w, on=["y","pt"], how="left")

            # Flexible pT window (only apply if provided)
            if pt_min is not None:
                g = g[g["pt"] >= float(pt_min)]
            if pt_max is not None:
                g = g[g["pt"] <= float(pt_max)]
            g = g.copy()
            if g.empty:
                out.append((label, pd.DataFrame(columns=["y","y_left","y_right","r_central","r_lo","r_hi"])));
                continue

            # Careful handling of backward, very-high-pT corner
            if (pt_guard is not None) or (K_cap is not None):
                SA = g["SA"].to_numpy()
                SA_safe = np.clip(SA, 1e-12, None)
                K = (1.0 + self.Nnorm * (SA - 1.0) * float(alpha)) / SA_safe
                bad = np.zeros(len(g), dtype=bool)
                if pt_guard is not None:
                    bad |= (g["y"].to_numpy() < 0.0) & (g["pt"].to_numpy() > float(pt_guard))
                if K_cap is not None:
                    bad |= (K > float(K_cap))
                if bad.any():
                    # revert those rows to inclusive (i.e., divide out the applied K → set K=1)
                    for col in ("r_central","r_lo","r_hi"):
                        if col in g.columns:
                            arr = g[col].to_numpy()
                            arr[bad] = arr[bad] / K[bad]
                            g[col] = arr

            # Bin in y
            y = g["y"].to_numpy()
            vmin, vmax = float(np.min(y)), float(np.max(y))
            y_edges = np.arange(math.floor(vmin/y_width)*y_width,
                                math.ceil(vmax/y_width)*y_width + 1e-12, y_width)
            ids = np.digitize(y, y_edges) - 1
            g["ybin"] = ids

            def _wavg(df, col):
                w = np.clip(df["w"].to_numpy(), 0, None); v = df[col].to_numpy(); s = w.sum()
                return float((v*w).sum()/s) if s>0 else np.nan

            rows = []
            for bidx in sorted(np.unique(ids)):
                sub = g[g["ybin"]==bidx]
                y_left, y_right = y_edges[bidx], y_edges[bidx+1]
                rows.append({
                    "y": 0.5*(y_left+y_right),
                    "y_left": y_left, "y_right": y_right,
                    "r_central": _wavg(sub,"r_central"),
                    "r_lo":      _wavg(sub,"r_lo"),
                    "r_hi":      _wavg(sub,"r_hi")
                })
            out.append((label, pd.DataFrame(rows)))
            if verbose: print(f"  {label}: y-bins={len(rows)} (Δy={y_width})")
        return out

    def rpa_vs_pt_in_centrality_bins(self, rgrid: pd.DataFrame, df_pa: pd.DataFrame, centrality_edges_pct: Sequence[float],
                                     y_min: float, y_max: float, pt_width: float = 2.5, sigmaNN_mb: float = 70.0,
                                     weight: str = "inelastic", verbose: bool = True
                                     ) -> List[Tuple[str, pd.DataFrame]]:
        rg = self.attach_SA_to_grid(rgrid, verbose=False)
        df_pa_w = df_pa[["y","pt","val"]].rename(columns={"val":"w"})
        rg = rg[(rg["y"]>=y_min)&(rg["y"]<=y_max)].copy()
        rg = rg.merge(df_pa_w, on=["y","pt"], how="left")
        if rg.empty:
            return [(f"{centrality_edges_pct[i]}-{centrality_edges_pct[i+1]}%", pd.DataFrame()) for i in range(len(centrality_edges_pct)-1)]

        alphas = self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight)
        out = []
        pt = rg["pt"].to_numpy()
        pmin, pmax = float(np.min(pt)), float(np.max(pt))
        pt_edges = np.arange(math.floor(pmin/pt_width)*pt_width,
                             math.ceil(pmax/pt_width)*pt_width + 1e-12, pt_width)
        ids = np.digitize(pt, pt_edges) - 1
        rg["ptbin"] = ids

        for label, alpha in alphas:
            if not np.isfinite(alpha):
                if verbose: print(f"  [warn] alpha NaN for bin {label}; skipping.")
                continue
            g = self._scale_grid_by_alpha(rg, alpha)
            def _wavg(df, col):
                w = np.clip(df["w"].to_numpy(), 0, None); v = df[col].to_numpy(); s = w.sum()
                return float((v*w).sum()/s) if s>0 else np.nan
            rows = []
            for bidx in sorted(np.unique(ids)):
                sub = g[g["ptbin"]==bidx]
                if sub.empty: continue
                pt_left, pt_right = pt_edges[bidx], pt_edges[bidx+1]
                rows.append({"pt":0.5*(pt_left+pt_right),"pt_left":pt_left,"pt_right":pt_right,
                             "r_central":_wavg(sub,"r_central"),"r_lo":_wavg(sub,"r_lo"),"r_hi":_wavg(sub,"r_hi")})
            out.append((label, pd.DataFrame(rows)))
            if verbose: print(f"  {label}: pT-bins={len(rows)} in y∈[{y_min},{y_max}] (ΔpT={pt_width})")
        return out

# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------

__all__ = [
    # datasets & analysis
    "NPDFSystem", "RpAAnalysis",
    # plotting
    "style_axes", "step_band_xy", "overlay_error_members", "slice_nearest_pt_for_each_y",
    "band_xy", "step_band_from_centers", "step_band_from_left_edges", "centers_to_left_edges",
    "plot_rpa_vs_centrality_hzerr",
    # centrality
    "WoodsSaxonPb", "CentralityModel", "GluonFromGrid", "GluonRatioTable",
    # TopDrawer
    "read_topdrawer", "load_top_file", "discover_by_number",
    # misc utils
    "ensure_dir", "ensure_out", "round_grid", "weighted_average", "GridStats",
]