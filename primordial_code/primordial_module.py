# -*- coding: utf-8 -*-
"""
primordial_module
=================
Read TAMU primordial charmonium output (datafile.gz), manage Glauber/centrality
maps, apply feed-down, and compute publication-ready R_pA observables.

Highlights
----------
- Works with 1, 2, or 3+ runs. With ≥2 runs: central=mean (or median for ≥3),
  band=envelope [min,max] with swap guard. Backwards compatible τ1/τ2 helpers.
- Direct outputs for R_pA vs b, vs Npart, vs y, vs pT; plus per-b scans.
- Double ratios vs b/Npart/y/pT.
- Default: feed-down ON; y flipped (p-going positive) ON.
- Optional √s scaling: σ_pp^dir weights × (√s/5.023)^0.5 (e.g., 8.16 TeV).
- Minimal, clean plotting helpers for central curve + theory band (step style).
- Robust parsing, clear errors, and “debug” prints.

Dependencies: NumPy, Pandas, Matplotlib (optional for plotting).
"""

from __future__ import annotations

import gzip, math, os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Matplotlib optional
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    mpl = None
    plt = None


# ------------------------------------------------------------------ utils ----

def _ensure_1d(a: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)

def _nanmean_and_sem(x: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if w is None:
        x = x[np.isfinite(x)]
        n = x.size
        if n == 0: return (np.nan, np.nan)
        mean = float(np.nanmean(x))
        sem  = float(np.nanstd(x, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        return (mean, sem)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    x, w = x[m], w[m]
    if x.size == 0: return (np.nan, np.nan)
    s = w.sum()
    if s <= 0 or not np.isfinite(s): return _nanmean_and_sem(x, None)
    mean = float(np.sum(w * x) / s)
    sem  = float(math.sqrt(np.sum((w**2) * (x - mean) ** 2)) / s)
    return (mean, sem)

def _safe_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    xp = _ensure_1d(xp); fp = _ensure_1d(fp); x = _ensure_1d(x)
    if xp.size == 0 or fp.size == 0: return np.full_like(x, np.nan)
    if xp.size == 1: return np.full_like(x, fp[0])
    order = np.argsort(xp)
    return np.interp(x, xp[order], fp[order], left=fp[order][0], right=fp[order][-1])

def _as_bins_from_edges(edges: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out = []
    for lo, hi in edges:
        lo, hi = float(lo), float(hi)
        if not (hi > lo): raise ValueError(f"Invalid bin ({lo}, {hi}); need hi > lo.")
        out.append((lo, hi))
    return out

def make_bins_from_width(start: float, stop: float, width: float) -> List[Tuple[float, float]]:
    start, stop, width = float(start), float(stop), float(width)
    if width <= 0: raise ValueError("width must be positive")
    edges = list(np.arange(start, stop, width))
    if not edges or edges[-1] < stop: edges.append(stop)
    return [(float(edges[i]), float(edges[i+1])) for i in range(len(edges)-1)]


# ------------------------------------------------------------ styling (opt) ----

class Style:
    linewidth: float = 2.0
    framewidth: float = 1.5
    tick_fs: int = 18
    title_fs: int = 20
    label_fs: int = 20
    font_family: str = "DejaVu Sans"

    @classmethod
    def apply(cls):
        if mpl is None: return
        mpl.rcParams.update({
            "axes.spines.top": True, "axes.spines.right": True,
            "axes.linewidth": cls.framewidth,
            "axes.titlesize": cls.title_fs,
            "axes.labelsize": cls.label_fs,
            "font.family": cls.font_family,
            "xtick.labelsize": cls.tick_fs, "ytick.labelsize": cls.tick_fs,
            "lines.linewidth": cls.linewidth,
            "figure.autolayout": True,
        })


# ----------------------------------------------------- config & feed-down ----

STATE_NAMES_DEFAULT = ["jpsi_1S", "chic0_1P", "chic1_1P", "chic2_1P", "psi_2S"]

@dataclass
class ReaderConfig:
    state_names: List[str] = field(default_factory=lambda: list(STATE_NAMES_DEFAULT))
    norm_factor: float = 1.0
    debug: bool = False
    def __post_init__(self):
        if len(self.state_names) != 5:
            raise ValueError("Expected exactly 5 state columns in suppression rows.")

@dataclass
class FeedDown:
    """Feed-down matrix F such that sigma_obs = F @ sigma_dir."""
    F: np.ndarray = field(default_factory=lambda: np.array([
        [1.0, 0.0141, 0.343, 0.195, 0.615 ],
        [0.0, 1.0,    0.0,   0.0,   0.0977],
        [0.0, 0.0,    1.0,   0.0,   0.0975],
        [0.0, 0.0,    0.0,   1.0,   0.0936],
        [0.0, 0.0,    0.0,   0.0,   1.0   ],
    ], dtype=float))
    def __post_init__(self):
        F = np.asarray(self.F, dtype=float)
        if F.shape != (5, 5): raise ValueError("FeedDown matrix must be 5x5.")
        self.F = F
        self.F_inv = np.linalg.inv(self.F)
    def direct_from_observed(self, sigma_obs: Sequence[float]) -> np.ndarray:
        v = _ensure_1d(sigma_obs)
        if v.size != 5: raise ValueError("Expected 5 observed cross sections.")
        return self.F_inv @ v
    def observed_from_direct(self, sigma_dir: Sequence[float]) -> np.ndarray:
        v = _ensure_1d(sigma_dir)
        if v.size != 5: raise ValueError("Expected 5 direct cross sections.")
        return self.F @ v


# ---------------------------------------------------------- centrality maps ----

@dataclass
class CentralityMaps:
    """Hold b↦Npart/Nbin/c and c↦b mappings; any subset may be provided."""
    b_grid: Optional[np.ndarray] = None
    npart_vals: Optional[np.ndarray] = None
    nbin_vals: Optional[np.ndarray] = None
    c_vals: Optional[np.ndarray] = None
    c_grid: Optional[np.ndarray] = None
    b_from_c_vals: Optional[np.ndarray] = None

    def set_b2npart(self, b, npart): self.b_grid = _ensure_1d(b); self.npart_vals = _ensure_1d(npart)
    def set_b2nbin(self, b, nbin):   self.b_grid = _ensure_1d(b); self.nbin_vals = _ensure_1d(nbin)

    def set_b2centrality(self, b, c):
        self.b_grid = _ensure_1d(b); self.c_vals = _ensure_1d(c)
        order = np.argsort(self.c_vals)
        c_sorted, b_sorted = self.c_vals[order], self.b_grid[order]
        if np.allclose(c_sorted, c_sorted[0]):
            self.c_grid = np.array([c_sorted[0]]); self.b_from_c_vals = np.array([np.nanmean(b_sorted)])
        else:
            self.c_grid = c_sorted; self.b_from_c_vals = b_sorted

    def b_to_npart(self, b) -> np.ndarray:
        if self.b_grid is None or self.npart_vals is None: raise KeyError("b→Npart map not set.")
        return _safe_interp(_ensure_1d(b), self.b_grid, self.npart_vals)
    def b_to_nbin(self, b) -> np.ndarray:
        if self.b_grid is None or self.nbin_vals is None: raise KeyError("b→Nbin map not set.")
        return _safe_interp(_ensure_1d(b), self.b_grid, self.nbin_vals)
    def b_to_c(self, b) -> np.ndarray:
        if self.b_grid is None or self.c_vals is None: raise KeyError("b→c map not set.")
        return _safe_interp(_ensure_1d(b), self.b_grid, self.c_vals)
    def c_to_b(self, c) -> np.ndarray:
        if self.c_grid is None or self.b_from_c_vals is None:
            if self.c_vals is None or self.b_grid is None: raise KeyError("c→b map not available.")
            order = np.argsort(self.c_vals)
            return _safe_interp(_ensure_1d(c), self.c_vals[order], self.b_grid[order])
        return _safe_interp(_ensure_1d(c), self.c_grid, self.b_from_c_vals)


# -------------------------------------------------------------- data ingest ----

@dataclass
class PrimordialDataset:
    """
    Loader for `datafile.gz` produced by TAMU code.

    Rows come in pairs:
      meta_row: [b, ..., pT, ..., y, ...]
      sup_row : first 5 entries = suppressions for states in cfg.state_names
    """
    path: str
    cfg: ReaderConfig = field(default_factory=ReaderConfig)
    meta_idx_b: Optional[int] = None
    meta_idx_pt: Optional[int] = None
    meta_idx_y: Optional[int] = None

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"datafile not found: {self.path}")

        def _rows(p):
            with gzip.open(p, "rt") as f:
                for ln, line in enumerate(f, 1):
                    s = line.strip()
                    if not s: continue
                    try:
                        yield [float(x) for x in s.split()]
                    except Exception:
                        if self.cfg.debug: print(f"[WARN] Non-numeric line {ln} skipped.")
                        continue

        rows = list(_rows(self.path))
        if len(rows) < 2: raise RuntimeError("file has too few rows to parse.")
        if len(rows) % 2 != 0:
            if self.cfg.debug: print("[WARN] Odd number of rows; dropping last line.")
            rows = rows[:-1]

        meta = np.asarray(rows[0::2], dtype=float)
        sup  = np.asarray(rows[1::2], dtype=float)
        if meta.shape[0] != sup.shape[0]:
            raise RuntimeError("meta/suppression row-count mismatch (file corrupted?).")

        b_idx = 0 if self.meta_idx_b is None else int(self.meta_idx_b)
        pt_idx = 4 if self.meta_idx_pt is None else int(self.meta_idx_pt)
        y_idx  = 6 if self.meta_idx_y is None else int(self.meta_idx_y)
        if meta.shape[1] <= max(b_idx, pt_idx, y_idx):
            raise IndexError("meta rows missing required columns for b/pt/y.")

        b, pt, y = meta[:, b_idx], meta[:, pt_idx], meta[:, y_idx]
        nstates = len(self.cfg.state_names)
        if sup.shape[1] < nstates:
            raise IndexError(f"suppression rows have {sup.shape[1]} columns; need ≥ {nstates}.")
        S = sup[:, :nstates]

        data = {"b": b, "pt": pt, "y": y}
        for j, name in enumerate(self.cfg.state_names):
            data[name] = S[:, j]

        df = pd.DataFrame(data)
        if self.cfg.debug:
            print(f"[INFO] Loaded {len(df):,} events from '{self.path}'. "
                  f"b∈[{np.nanmin(b):.2f},{np.nanmax(b):.2f}], "
                  f"pT∈[{np.nanmin(pt):.4g},{np.nanmax(pt):.4g}], "
                  f"y∈[{np.nanmin(y):.2f},{np.nanmax(y):.2f}]")
        return df


# -------------------------------------------------------------- core analysis ----

@dataclass
class PrimordialAnalysis:
    """
    Compute R_pA with/without feed-down. DataFrames have one x-column plus, per
    state, `<state>` and `<state>_err` columns. Defaults: feed-down ON, flip y ON.
    """
    df: pd.DataFrame
    centrality: CentralityMaps = field(default_factory=CentralityMaps)
    feeddown: FeedDown = field(default_factory=FeedDown)
    cfg: ReaderConfig = field(default_factory=ReaderConfig)
    sigma_exp_pp: np.ndarray = field(default_factory=lambda: np.array([21.91, 0.377, 0.386, 0.355, 3.26], dtype=float))
    pp_sigma_scale: float = 1.0  # optional √s scaling of σ_pp^dir weights

    def __post_init__(self):
        self.sigma_dir_pp = self.feeddown.direct_from_observed(self.sigma_exp_pp) * float(self.pp_sigma_scale)

    def set_energy_scaling_by_sqrts(self, sqrts_NN: float, ref: float = 5.023, power: float = 0.5):
        """Set σ_pp^dir × (sqrts_NN/ref)^power. Example: 8.16 TeV ⇒ power=0.5."""
        self.pp_sigma_scale = (float(sqrts_NN) / float(ref)) ** float(power)
        self.sigma_dir_pp = self.feeddown.direct_from_observed(self.sigma_exp_pp) * self.pp_sigma_scale

    @property
    def state_names(self) -> List[str]: return self.cfg.state_names

    # -------------------- helpers

    def _select_by(self, *, b_sel=None, pt_sel=None, y_sel=None) -> pd.DataFrame:
        m = pd.Series(True, index=self.df.index)
        if b_sel is not None:  lo, hi = b_sel;  m &= (self.df["b"]  >= lo) & (self.df["b"]  <= hi)
        if pt_sel is not None: lo, hi = pt_sel; m &= (self.df["pt"] >= lo) & (self.df["pt"] <= hi)
        if y_sel is not None:  lo, hi = y_sel;  m &= (self.df["y"]  >= lo) & (self.df["y"]  <= hi)
        return self.df.loc[m]

    def _weights_from_centrality(self, b_vals: np.ndarray, mode: str = "flat",
                                 p_of_c: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        if mode == "flat": return np.ones_like(b_vals, dtype=float)
        try: c = self.centrality.b_to_c(b_vals)
        except Exception: return np.ones_like(b_vals, dtype=float)
        p_raw = np.exp(-c / 0.25) if p_of_c is None else np.asarray(p_of_c(c), dtype=float)
        s = np.sum(p_raw)
        return (p_raw * (p_raw.size / s)) if (s > 0 and np.isfinite(s)) else np.ones_like(p_raw)

    def _apply_feeddown(self, R_mean: np.ndarray, R_err: np.ndarray,
                        nbin_weight: Optional[Union[float, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        sig = np.asarray(self.sigma_dir_pp, dtype=float)
        R_mean, R_err = np.asarray(R_mean, dtype=float), np.asarray(R_err, dtype=float)
        if nbin_weight is None:
            w = np.ones_like(sig)
        else:
            w = np.asarray(nbin_weight, dtype=float)
            if w.ndim == 0: w = np.full_like(sig, float(w))
            if w.size != sig.size: raise ValueError("nbin_weight size mismatch.")
        num = self.feeddown.F @ (w * sig * R_mean)
        den = self.feeddown.F @ (w * sig)
        with np.errstate(divide="ignore", invalid="ignore"): R_obs = num / den
        cov_dir = np.diag(R_err**2); J = self.feeddown.F @ np.diag(w * sig)
        cov_num = J @ cov_dir @ J.T
        var_obs = np.diag(cov_num) / (den**2)
        return R_obs, np.sqrt(np.maximum(0.0, var_obs))

    @staticmethod
    def _inject_chi1P_average(df: pd.DataFrame) -> pd.DataFrame:
        need = {"chic0_1P", "chic1_1P", "chic2_1P"}
        if not need.issubset(df.columns): return df
        vals = df[["chic0_1P", "chic1_1P", "chic2_1P"]].to_numpy(dtype=float)
        df["chicJ_1P"] = np.nanmean(vals, axis=1)
        err_cols = [c for c in df.columns if c.endswith("_err")]
        if {"chic0_1P_err","chic1_1P_err","chic2_1P_err"}.issubset(err_cols):
            errs = df[["chic0_1P_err","chic1_1P_err","chic2_1P_err"]].to_numpy(dtype=float)
            df["chicJ_1P_err"] = np.sqrt(np.nansum(errs**2, axis=1)) / 3.0
        return df

    # -------------------- public computations

    def rpa_vs_b(self, *, with_feeddown=True, use_nbin=True,
                 weight_mode="flat", p_of_c=None, verbose=False) -> pd.DataFrame:
        rows = []
        for i, (b_val, d) in enumerate(self.df.groupby("b", sort=True)):
            if verbose or self.cfg.debug: print(f"[rpa_vs_b] b={b_val:.3f} ({i+1})  n={len(d):,}")
            w = self._weights_from_centrality(d["b"].to_numpy(), mode=weight_mode, p_of_c=p_of_c)
            means, errs = [], []
            for name in self.state_names:
                m, e = _nanmean_and_sem(d[name].to_numpy(), w); means.append(m); errs.append(e)
            means, errs = np.array(means), np.array(errs)
            if with_feeddown:
                nbin_vec = None
                if use_nbin:
                    try: nbin_vec = self.cfg.norm_factor * self.centrality.b_to_nbin(np.array([b_val]))[0]
                    except Exception: nbin_vec = None
                means, errs = self._apply_feeddown(means, errs, nbin_weight=nbin_vec)
            row = {"b": float(b_val)}
            for k, name in enumerate(self.state_names): row[name] = means[k]; row[f"{name}_err"] = errs[k]
            rows.append(row)

        out = pd.DataFrame(rows).sort_values("b").reset_index(drop=True)
        try: out["Npart"] = self.centrality.b_to_npart(out["b"].to_numpy())
        except Exception: pass
        return self._inject_chi1P_average(out)

    def rpa_vs_npart(self, **kwargs) -> pd.DataFrame:
        dfb = self.rpa_vs_b(**kwargs)
        if "Npart" not in dfb.columns: raise RuntimeError("Npart map not available.")
        cols = [c for c in dfb.columns if c not in ("b", "Npart")]
        return dfb[["Npart"] + cols].sort_values("Npart").reset_index(drop=True)

    def rpa_vs_pt(self, *, y_window: Tuple[float, float], pt_bins: Sequence[Tuple[float, float]],
                  with_feeddown=True, use_nbin=True, weight_mode="flat", p_of_c=None,
                  verbose=False) -> pd.DataFrame:
        bins = _as_bins_from_edges(pt_bins); rows = []
        for j, (lo, hi) in enumerate(bins):
            dsel = self._select_by(pt_sel=(lo, hi), y_sel=y_window)
            if dsel.empty: continue
            if verbose or self.cfg.debug: print(f"[rpa_vs_pt] bin {j+1}: pT∈[{lo:.2f},{hi:.2f}]  n={len(dsel):,}")
            w = self._weights_from_centrality(dsel["b"].to_numpy(), mode=weight_mode, p_of_c=p_of_c)
            means, errs = [], []
            for name in self.state_names:
                m, e = _nanmean_and_sem(dsel[name].to_numpy(), w); means.append(m); errs.append(e)
            means, errs = np.array(means), np.array(errs)
            if with_feeddown:
                nbin_vec = None
                if use_nbin:
                    try:
                        bbar = float(np.nanmean(dsel["b"].to_numpy()))
                        nbin_vec = self.cfg.norm_factor * self.centrality.b_to_nbin(np.array([bbar]))[0]
                    except Exception: nbin_vec = None
                means, errs = self._apply_feeddown(means, errs, nbin_weight=nbin_vec)
            row = {"pt": 0.5 * (lo + hi)}
            for k, name in enumerate(self.state_names): row[name] = means[k]; row[f"{name}_err"] = errs[k]
            rows.append(row)
        return self._inject_chi1P_average(pd.DataFrame(rows).sort_values("pt").reset_index(drop=True))

    def rpa_vs_y(self, *, pt_window: Tuple[float, float], y_bins: Sequence[Tuple[float, float]],
                 with_feeddown=True, use_nbin=True, weight_mode="flat", p_of_c=None,
                 flip_y=True, verbose=False) -> pd.DataFrame:
        bins = _as_bins_from_edges(y_bins); rows = []
        for j, (lo, hi) in enumerate(bins):
            dsel = self._select_by(y_sel=(lo, hi), pt_sel=pt_window)
            if dsel.empty: continue
            if verbose or self.cfg.debug: print(f"[rpa_vs_y] bin {j+1}: y∈[{lo:.2f},{hi:.2f}]  n={len(dsel):,}")
            w = self._weights_from_centrality(dsel["b"].to_numpy(), mode=weight_mode, p_of_c=p_of_c)
            means, errs = [], []
            for name in self.state_names:
                m, e = _nanmean_and_sem(dsel[name].to_numpy(), w); means.append(m); errs.append(e)
            means, errs = np.array(means), np.array(errs)
            if with_feeddown:
                nbin_vec = None
                if use_nbin:
                    try:
                        bbar = float(np.nanmean(dsel["b"].to_numpy()))
                        nbin_vec = self.cfg.norm_factor * self.centrality.b_to_nbin(np.array([bbar]))[0]
                    except Exception: nbin_vec = None
                means, errs = self._apply_feeddown(means, errs, nbin_weight=nbin_vec)
            ymid = 0.5 * (lo + hi)
            row = {"y": (-ymid if flip_y else ymid)}
            for k, name in enumerate(self.state_names): row[name] = means[k]; row[f"{name}_err"] = errs[k]
            rows.append(row)
        return self._inject_chi1P_average(pd.DataFrame(rows).sort_values("y").reset_index(drop=True))

    # per-b scans

    def rpa_vs_y_per_b(self, *, pt_window: Tuple[float, float], y_bins: Sequence[Tuple[float, float]],
                       with_feeddown=True, use_nbin=True, flip_y=True, verbose=False) -> pd.DataFrame:
        bins = _as_bins_from_edges(y_bins); out = []
        for ib, (b_val, dfb) in enumerate(self.df.groupby("b", sort=True)):
            if verbose or self.cfg.debug: print(f"[rpa_vs_y_per_b] b={b_val:.3f} ({ib+1})")
            for lo, hi in bins:
                dsel = dfb[(dfb["y"] >= lo) & (dfb["y"] <= hi) &
                           (dfb["pt"] >= pt_window[0]) & (dfb["pt"] <= pt_window[1])]
                if dsel.empty: continue
                means, errs = [], []
                for name in self.state_names:
                    m, e = _nanmean_and_sem(dsel[name].to_numpy(), None); means.append(m); errs.append(e)
                means, errs = np.array(means), np.array(errs)
                if with_feeddown:
                    nbin_vec = None
                    if use_nbin:
                        try: nbin_vec = self.cfg.norm_factor * self.centrality.b_to_nbin(np.array([b_val]))[0]
                        except Exception: nbin_vec = None
                    means, errs = self._apply_feeddown(means, errs, nbin_weight=nbin_vec)
                row = {"b": float(b_val), "y": (-0.5 * (lo + hi) if flip_y else 0.5 * (lo + hi))}
                for k, name in enumerate(self.state_names): row[name] = means[k]; row[f"{name}_err"] = errs[k]
                out.append(row)
        return self._inject_chi1P_average(pd.DataFrame(out).sort_values(["b", "y"]).reset_index(drop=True))

    def rpa_vs_pt_per_b(self, *, y_window: Tuple[float, float], pt_bins: Sequence[Tuple[float, float]],
                        with_feeddown=True, use_nbin=True, verbose=False) -> pd.DataFrame:
        bins = _as_bins_from_edges(pt_bins); out = []
        for ib, (b_val, dfb) in enumerate(self.df.groupby("b", sort=True)):
            if verbose or self.cfg.debug: print(f"[rpa_vs_pt_per_b] b={b_val:.3f} ({ib+1})")
            for lo, hi in bins:
                dsel = dfb[(dfb["pt"] >= lo) & (dfb["pt"] <= hi) &
                           (dfb["y"] >= y_window[0]) & (dfb["y"] <= y_window[1])]
                if dsel.empty: continue
                means, errs = [], []
                for name in self.state_names:
                    m, e = _nanmean_and_sem(dsel[name].to_numpy(), None); means.append(m); errs.append(e)
                means, errs = np.array(means), np.array(errs)
                if with_feeddown:
                    nbin_vec = None
                    if use_nbin:
                        try: nbin_vec = self.cfg.norm_factor * self.centrality.b_to_nbin(np.array([b_val]))[0]
                        except Exception: nbin_vec = None
                    means, errs = self._apply_feeddown(means, errs, nbin_weight=nbin_vec)
                row = {"b": float(b_val), "pt": 0.5 * (lo + hi)}
                for k, name in enumerate(self.state_names): row[name] = means[k]; row[f"{name}_err"] = errs[k]
                out.append(row)
        return self._inject_chi1P_average(pd.DataFrame(out).sort_values(["b", "pt"]).reset_index(drop=True))

    # double ratios

    def _double_ratio_from_df(self, df: pd.DataFrame, xcol: str, *,
                              num_state="psi_2S", den_state="jpsi_1S") -> pd.DataFrame:
        if num_state not in df.columns or den_state not in df.columns:
            raise KeyError("Requested states not present.")
        pp_ratio = float(self.sigma_exp_pp[self.state_names.index(num_state)] /
                         self.sigma_exp_pp[self.state_names.index(den_state)])
        num, den = df[num_state].to_numpy(), df[den_state].to_numpy()
        en = df.get(f"{num_state}_err", pd.Series(np.zeros_like(num))).to_numpy()
        ed = df.get(f"{den_state}_err", pd.Series(np.zeros_like(den))).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (num / den) / pp_ratio
            rel2 = np.zeros_like(ratio)
            m1 = (num > 0) & np.isfinite(en); rel2[m1] += (en[m1] / num[m1]) ** 2
            m2 = (den > 0) & np.isfinite(ed); rel2[m2] += (ed[m2] / den[m2]) ** 2
            ratio_err = np.abs(num / den) * np.sqrt(rel2) / pp_ratio
        return pd.DataFrame({xcol: df[xcol], "ratio": ratio, "ratio_err": ratio_err})

    def double_ratio_vs_b(self, **kwargs) -> pd.DataFrame:
        return self._double_ratio_from_df(self.rpa_vs_b(**kwargs), "b")
    def double_ratio_vs_npart(self, **kwargs) -> pd.DataFrame:
        return self._double_ratio_from_df(self.rpa_vs_npart(**kwargs), "Npart")
    def double_ratio_vs_y(self, *, pt_window, y_bins, **kwargs) -> pd.DataFrame:
        return self._double_ratio_from_df(self.rpa_vs_y(pt_window=pt_window, y_bins=y_bins, **kwargs), "y")
    def double_ratio_vs_pt(self, *, y_window, pt_bins, **kwargs) -> pd.DataFrame:
        return self._double_ratio_from_df(self.rpa_vs_pt(y_window=y_window, pt_bins=pt_bins, **kwargs), "pt")


# ---------------------------------------------------------- N-run ensemble ----

@dataclass
class BandEnsemble:
    """Collect multiple PrimordialAnalysis runs (e.g., τ1, τ2, ...), build bands."""
    runs: Dict[str, PrimordialAnalysis] = field(default_factory=dict)
    def add_run(self, tag: str, analysis: PrimordialAnalysis):
        if tag in self.runs: raise KeyError(f"Run tag '{tag}' already exists.")
        self.runs[tag] = analysis

    # -- generic combiner --
    def _combine(self, frames: Dict[str, pd.DataFrame], keys: Sequence[str],
                 center_method: str = "auto") -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Outer merge across all runs
        merged = None
        for tag, df in frames.items():
            cols = [c for c in df.columns if c not in keys]
            df2 = df.copy()
            df2.columns = [*keys] + [f"{c}:{tag}" for c in cols]
            merged = df2 if merged is None else pd.merge(merged, df2, on=list(keys), how="outer", sort=True)

        # Interpolate numeric gaps
        for c in merged.columns:
            if c in keys: continue
            if merged[c].dtype.kind in "fiu":
                merged[c] = merged[c].interpolate("linear", limit_direction="both")

        # Prepare outputs
        base_states = sorted({c for c in frames[next(iter(frames))].columns if c not in keys and not c.endswith("_err")})
        # Build envelope and center
        res_center, res_band = {k: merged[k] for k in keys}, {k: merged[k] for k in keys}

        # Decide center method
        tags = list(frames.keys())
        if center_method == "auto":
            if any(t.lower() in ("mid", "center", "central") for t in tags):
                center_method = "tag"
            elif len(tags) >= 3:
                center_method = "median"
            else:
                center_method = "mean"

        for s in base_states:
            # Collect across tags
            cols = [f"{s}:{t}" for t in tags if f"{s}:{t}" in merged.columns]
            stack = np.vstack([merged[c].to_numpy() for c in cols])  # shape (n_runs, n_points)
            lo = np.nanmin(stack, axis=0); hi = np.nanmax(stack, axis=0)
            res_band[f"{s}_lo"], res_band[f"{s}_hi"] = lo, hi

            if center_method == "tag":
                sel = next(t for t in tags if t.lower() in ("mid", "center", "central"))
                res_center[s] = merged[f"{s}:{sel}"]
                # combine provided SEMs if present
                ecols = [f"{s}_err:{sel}"] if f"{s}_err:{sel}" in merged.columns else []
            elif center_method == "median":
                res_center[s] = np.nanmedian(stack, axis=0)
                ecols = [f"{s}_err:{t}" for t in tags if f"{s}_err:{t}" in merged.columns]
            else:  # mean
                res_center[s] = np.nanmean(stack, axis=0)
                ecols = [f"{s}_err:{t}" for t in tags if f"{s}_err:{t}" in merged.columns]

            if ecols:
                errs = np.vstack([merged[c].to_numpy() for c in ecols])
                # conservative: RMS/len(ecols)**0.5 approximates SEM of the mean; median → use RMS
                if center_method == "median":
                    res_center[f"{s}_err"] = np.sqrt(np.nanmean(errs**2, axis=0))
                else:
                    res_center[f"{s}_err"] = np.sqrt(np.nansum(errs**2, axis=0)) / max(1, len(ecols))
            else:
                res_center[f"{s}_err"] = 0.0

        return pd.DataFrame(res_center), pd.DataFrame(res_band)

    # ----- public helpers that mirror single-run API -----

    def _collect(self, fn: str, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        return {tag: getattr(ana, fn)(*args, **kwargs) for tag, ana in self.runs.items()}

    def central_and_band_vs_b(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfs = self._collect("rpa_vs_b", **kwargs);      return self._combine(dfs, keys=["b"])
    def central_and_band_vs_npart(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfs = self._collect("rpa_vs_npart", **kwargs);  return self._combine(dfs, keys=["Npart"])
    def central_and_band_vs_y(self, *, pt_window, y_bins, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfs = self._collect("rpa_vs_y", pt_window=pt_window, y_bins=y_bins, **kwargs); return self._combine(dfs, keys=["y"])
    def central_and_band_vs_y_per_b(self, *, pt_window, y_bins, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfs = self._collect("rpa_vs_y_per_b", pt_window=pt_window, y_bins=y_bins, **kwargs); return self._combine(dfs, keys=["b","y"])
    def central_and_band_vs_pt_per_b(self, *, y_window, pt_bins, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfs = self._collect("rpa_vs_pt_per_b", y_window=y_window, pt_bins=pt_bins, **kwargs); return self._combine(dfs, keys=["b","pt"])

    def central_and_band_vs_pt(self, *, y_window, pt_bins, **kwargs):
        """
        Center + band R_pA(pT) for each state, combining all runs.

        Parameters
        ----------
        y_window : (y_min, y_max)
            Rapidity window matching your CNM y-window.
        pt_bins : list of (pt_min, pt_max)
            Must match the CNM pT binning (DEFAULT_P_EDGES).

        Other kwargs are passed to PrimordialAnalysis.rpa_vs_pt.
        """
        dfs = self._collect("rpa_vs_pt", y_window=y_window, pt_bins=pt_bins, **kwargs)
        return self._combine(dfs, keys=["pt"])

# --------------------------------------------------------------- plotting ----

def plot_with_bands(df_center: pd.DataFrame, df_band: Optional[pd.DataFrame],
                    xcol: str, states: Sequence[str], *,
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    outfile: Optional[str] = None,
                    ytick_step: Optional[float] = None,
                    xtick_step: Optional[float] = None,
                    y_minor: bool = True,
                    x_minor: bool = True,
                    minor_divisions: int = 2,
                    step: bool = True,
                    flip_x: bool = False,
                    legend_labels: Optional[Dict[str, str]] = None,
                    xlim: Optional[Tuple[float, float]] = None,
                    ylim: Optional[Tuple[float, float]] = None):
    if plt is None: return
    Style.apply()
    dc = df_center.copy()
    band = df_band.copy() if df_band is not None else None
    if flip_x:
        dc[xcol] = -dc[xcol]
        if band is not None: band[xcol] = -band[xcol]
    if xlabel is None:
        xlabel = {"y": r"$y$", "pt": r"$p_T$ [\mathrm{GeV}]", "Npart": r"$N_{\mathrm{part}}$", "b": r"$b\ [\mathrm{fm}]$"}.get(xcol, xcol)
    if ylabel is None: ylabel = r"$R_{pA}$"

    fig, ax = plt.subplots()

    if band is not None:
        xs = band[xcol].to_numpy()
        for s in states:
            lo = band.get(f"{s}_lo", None); hi = band.get(f"{s}_hi", None)
            if lo is None or hi is None: continue
            ax.fill_between(xs, lo, hi, alpha=0.25, step="mid" if step else None)

    for s in states:
        x = dc[xcol].to_numpy(); y = dc[s].to_numpy()
        yerr = dc.get(f"{s}_err", pd.Series(np.zeros_like(y))).to_numpy()
        lab = (legend_labels or {}).get(s, {"jpsi_1S": r"$J/\psi(1S)$", "chicJ_1P": r"$\chi_c(1P)$", "psi_2S": r"$\psi(2S)$"}.get(s, s))
        if step:
            ax.step(x, y, where="mid", label=lab)
            ax.errorbar(x, y, yerr=yerr, fmt="none", lw=1.0, capsize=2)
        else:
            ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", label=lab)

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if xtick_step: ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xtick_step))
    if ytick_step: ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ytick_step))
    if x_minor: ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=minor_divisions))
    if y_minor: ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=minor_divisions))
    ax.tick_params(which="major", length=6, width=1.0)
    ax.tick_params(which="minor", length=3, width=0.8)
    if title: ax.set_title(title)
    ax.legend(frameon=True)
    if outfile: fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig, ax


def subplot_grid_rpa_y_per_b(df_center: pd.DataFrame, df_band: Optional[pd.DataFrame],
                             state: str, *, ncols: int = 3,
                             drop_b: Optional[Sequence[float]] = None,
                             xtick_step: float = 1.0, ytick_step: float = 0.1,
                             ylim: Tuple[float, float] = (0.3, 1.0),
                             title_prefix: str = r"$R_{pA}(y)$ at $b=$",
                             sharey: bool = True):
    if plt is None: return
    Style.apply()
    bs = np.unique(df_center["b"].to_numpy())
    if drop_b is not None:
        drop = set(float(b) for b in drop_b); bs = [b for b in bs if float(b) not in drop]
    n = len(bs); ncols = max(1, int(ncols)); nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.8*nrows),
                             squeeze=False, sharey=sharey)

    for idx, bval in enumerate(bs):
        r, c = divmod(idx, ncols); ax = axes[r][c]
        slc_c = df_center[df_center["b"] == bval].sort_values("y")
        if df_band is not None:
            slc_b = df_band[df_band["b"] == bval].sort_values("y")
            ax.fill_between(slc_b["y"], slc_b[f"{state}_lo"], slc_b[f"{state}_hi"], alpha=0.25, step="mid")
        ax.step(slc_c["y"], slc_c[state], where="mid")
        yerr = slc_c.get(f"{state}_err", pd.Series(np.zeros(len(slc_c)))).to_numpy()
        ax.errorbar(slc_c["y"], slc_c[state], yerr=yerr, fmt="none", marker=None, ms=0,
                    lw=1.0, capsize=2)
        ax.set_title(f"{title_prefix}{bval:.2f} fm"); ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xtick_step))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ytick_step))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))

    for j in range(n, nrows*ncols): r, c = divmod(j, ncols); axes[r][c].set_visible(False)
    for ax in axes[-1]: ax.set_xlabel(r"$y$")
    for ax in axes[:, 0]: ax.set_ylabel(r"$R_{pA}$")
    fig.tight_layout(); return fig, axes


def subplot_grid_rpa_pt_per_b(df_center: pd.DataFrame, df_band: Optional[pd.DataFrame],
                              state: str, *, ncols: int = 3,
                              drop_b: Optional[Sequence[float]] = None,
                              xtick_step: float = 2.5, ytick_step: float = 0.1,
                              ylim: Tuple[float, float] = (0.3, 1.0),
                              title_prefix: str = r"$R_{pA}(p_T)$ at $b=$",
                              sharey: bool = True,
                              note: Optional[str] = None):
    if plt is None: return
    Style.apply()
    bs = np.unique(df_center["b"].to_numpy())
    if drop_b is not None:
        drop = set(float(b) for b in drop_b); bs = [b for b in bs if float(b) not in drop]
    n = len(bs); ncols = max(1, int(ncols)); nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.8*nrows),
                             squeeze=False, sharey=sharey)

    for idx, bval in enumerate(bs):
        r, c = divmod(idx, ncols); ax = axes[r][c]
        slc_c = df_center[df_center["b"] == bval].sort_values("pt")
        if df_band is not None:
            slc_b = df_band[df_band["b"] == bval].sort_values("pt")
            ax.fill_between(slc_b["pt"], slc_b[f"{state}_lo"], slc_b[f"{state}_hi"], alpha=0.25, step="mid")
        ax.step(slc_c["pt"], slc_c[state], where="mid")
        yerr = slc_c.get(f"{state}_err", pd.Series(np.zeros(len(slc_c)))).to_numpy()
        ax.errorbar(slc_c["pt"], slc_c[state], yerr=yerr, fmt="none", marker=None, ms=0,
                    lw=1.0, capsize=2)
        ax.set_title(f"{title_prefix}{bval:.2f} fm"); ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xtick_step))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ytick_step))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=2))
        if note:
            ax.text(0.98, 0.02, note, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=Style.tick_fs-2,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    for j in range(n, nrows*ncols): r, c = divmod(j, ncols); axes[r][c].set_visible(False)
    for ax in axes[-1]: ax.set_xlabel(r"$p_T$ [GeV]")
    for ax in axes[:, 0]: ax.set_ylabel(r"$R_{pA}$")
    fig.tight_layout(); return fig, axes


# ----------------------------------------------------------- centrality I/O ----

def _load_two_col_tsv(path):
    rows = []
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if (not s) or s.startswith("#"): continue
            parts = [p for p in s.replace(",", " ").split() if p]
            if len(parts) < 2: continue
            try: rows.append((float(parts[0]), float(parts[1])))
            except Exception: continue
    if not rows: raise ValueError(f"No usable data in {path}")
    return np.asarray(rows, dtype=float)

_DEFAULT_PPB5_MBVALS = [0, 1.52647, 2.79103, 3.61425, 4.27999, 4.85473, 5.36824, 5.83783, 6.27931, 6.72798, 7.42169]
_DEFAULT_PPB5_NPART = [15.720849737871513, 15.214024855878275, 14.131458212934922, 12.921264444118433, 11.556973627423103,
                       10.038473881729283, 8.410715519698396, 6.758654727694391, 5.166849166284715, 3.639621838387811,
                       1.8282093640749786]

class CentralityIO:
    """Load centrality maps either from a folder or individual files."""
    @staticmethod
    def load_maps_from_folder(root: str) -> "CentralityMaps":
        from pathlib import Path
        rootp = Path(root)
        if not rootp.exists(): raise FileNotFoundError(f"Centrality folder not found: {root}")
        maps = CentralityMaps()

        # b <-> c
        bvsc = None
        for nm in ("bvscData.tsv", "b_vs_c.tsv", "bvsc.tsv"):
            p = rootp / nm
            if p.exists(): bvsc = p; break
        if bvsc is None: raise FileNotFoundError(f"Missing bvscData file in {root}.")
        arr = _load_two_col_tsv(str(bvsc))
        a0max, a1max = float(arr[:,0].max()), float(arr[:,1].max())
        if a0max <= 1.2 and a1max > 1.2:  cvals, bvals = arr[:,0], arr[:,1]
        elif a1max <= 1.2 and a0max > 1.2: bvals, cvals = arr[:,0], arr[:,1]
        else:                               bvals, cvals = arr[:,0], arr[:,1]
        maps.set_b2centrality(bvals, cvals)

        # Nbin(b)
        for nm in ("nbinvsbData.tsv", "nbin_vs_b.tsv", "Nbin_vs_b.tsv"):
            p = rootp / nm
            if p.exists():
                arr = _load_two_col_tsv(str(p)); maps.set_b2nbin(arr[:,0], arr[:,1]); break

        # Npart(b) optional
        have_npart = False
        for nm in ("npartvsbData.tsv", "npart_vs_b.tsv", "Npart_vs_b.tsv"):
            p = rootp / nm
            if p.exists():
                arr = _load_two_col_tsv(str(p)); maps.set_b2npart(arr[:,0], arr[:,1]); have_npart = True; break
        if not have_npart:
            maps.set_b2npart(_DEFAULT_PPB5_MBVALS, _DEFAULT_PPB5_NPART)
        return maps

    @staticmethod
    def load_maps_from_files(*, bvsc: Optional[str] = None, nbin: Optional[str] = None, npart: Optional[str] = None) -> "CentralityMaps":
        maps = CentralityMaps()
        if bvsc and os.path.exists(bvsc):
            arr = _load_two_col_tsv(bvsc)
            a0max, a1max = float(arr[:,0].max()), float(arr[:,1].max())
            if a0max <= 1.2 and a1max > 1.2:  cvals, bvals = arr[:,0], arr[:,1]
            elif a1max <= 1.2 and a0max > 1.2: bvals, cvals = arr[:,0], arr[:,1]
            else:                               bvals, cvals = arr[:,0], arr[:,1]
            maps.set_b2centrality(bvals, cvals)
        if nbin and os.path.exists(nbin):
            arr = _load_two_col_tsv(nbin); maps.set_b2nbin(arr[:,0], arr[:,1])
        if npart and os.path.exists(npart):
            arr = _load_two_col_tsv(npart); maps.set_b2npart(arr[:,0], arr[:,1])
        return maps


# ---------------------------------------------------------- handy wrappers ----

Y_WINDOW_ALL      = (-5.0, 5.0)
# Y_WINDOW_FORWARD  = (1.5, 4.0)
# Y_WINDOW_BACKWARD = (-5.0, -2.5)
# Y_WINDOW_CENTRAL  = (-1.93, 1.93)

Y_WINDOW_FORWARD  = (2.03,3.53)
Y_WINDOW_BACKWARD = (-4.46,-2.96)
Y_WINDOW_CENTRAL  = (-1.37,0.43)

def rpa_vs_pt_binned(analysis: PrimordialAnalysis, *, y_window, pt_max=20.0, pt_width=2.5,
                     with_feeddown=True, use_nbin=True, weight_mode="flat", p_of_c=None, verbose=False):
    bins = make_bins_from_width(0.0, float(pt_max), float(pt_width))
    return analysis.rpa_vs_pt(y_window=y_window, pt_bins=bins, with_feeddown=with_feeddown,
                              use_nbin=use_nbin, weight_mode=weight_mode, p_of_c=p_of_c, verbose=verbose)

def rpa_vs_y_binned(analysis: PrimordialAnalysis, *, pt_window, y_window=Y_WINDOW_ALL,
                    y_width=0.5, with_feeddown=True, use_nbin=True, weight_mode="flat",
                    p_of_c=None, flip_y=True, verbose=False):
    lo, hi = y_window
    bins = make_bins_from_width(lo, hi, float(y_width))
    return analysis.rpa_vs_y(pt_window=pt_window, y_bins=bins, with_feeddown=with_feeddown,
                             use_nbin=use_nbin, weight_mode=weight_mode, p_of_c=p_of_c,
                             flip_y=flip_y, verbose=verbose)

def build_ensemble(base_prefix: str, centrality_root: str, tags: Sequence[str] = ("tau1", "tau2"),
                   *, cfg: Optional[ReaderConfig] = None, sqrts_NN: Optional[float] = None) -> Tuple[BandEnsemble, Dict[str, PrimordialAnalysis]]:
    """
    Load multiple runs like base_prefix + f"_{tag}/datafile.gz", share Glauber maps,
    return (ensemble, {tag: PrimordialAnalysis}). If sqrts_NN is given, apply √s scaling.
    """
    cfg = cfg or ReaderConfig(debug=True)
    maps = CentralityIO.load_maps_from_folder(centrality_root)
    analyses: Dict[str, PrimordialAnalysis] = {}
    for tag in tags:
        path = os.path.join(base_prefix + f"_{tag}", "datafile.gz")
        if not os.path.exists(path): raise FileNotFoundError(f"Missing file for {tag}: {path}")
        df = PrimordialDataset(path, cfg=cfg).load()
        ana = PrimordialAnalysis(df, centrality=maps, cfg=cfg)
        if sqrts_NN is not None: ana.set_energy_scaling_by_sqrts(sqrts_NN)
        analyses[tag] = ana
    ens = BandEnsemble()
    for tag, ana in analyses.items(): ens.add_run(tag, ana)
    return ens, analyses


__all__ = [
    "ReaderConfig", "FeedDown", "CentralityMaps", "CentralityIO", "PrimordialDataset",
    "PrimordialAnalysis", "BandEnsemble", "Style",
    "make_bins_from_width", "plot_with_bands",
    "subplot_grid_rpa_y_per_b", "subplot_grid_rpa_pt_per_b",
    "Y_WINDOW_ALL", "Y_WINDOW_FORWARD", "Y_WINDOW_BACKWARD", "Y_WINDOW_CENTRAL",
    "rpa_vs_pt_binned", "rpa_vs_y_binned",
    "build_ensemble",
]

