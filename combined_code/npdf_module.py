# -*- coding: utf-8 -*-
"""
npdf_module.py
==============
Concise, single-file module merging I/O, analysis, plotting, and centrality.

Public API (stable)
-------------------
from npdf_module import (
    # datasets & analysis
    NPDFSystem, RpAAnalysis,
    # plotting
    style_axes, step_band_xy, overlay_error_members, slice_nearest_pt_for_each_y,
    band_xy, step_band_from_centers, step_band_from_left_edges, centers_to_left_edges,
    plot_rpa_vs_centrality_hzerr,
    # centrality
    WoodsSaxonPb, CentralityModel, GluonFromGrid, GluonRatioTable,
    # TopDrawer I/O
    read_topdrawer, load_top_file, discover_by_number,
    # misc
    ensure_dir, ensure_out, round_grid, weighted_average, GridStats,
)

What we compute
---------------
1) Parse TopDrawer differential cross sections:
   R^{(k)}_0(y,pT) = σ_pA^(k)(y,pT)/σ_pp(y,pT) for k∈{central, Hessian members}
2) Build RpA^(k) = K^(k)(b,y,pT) * R^{(k)}_0(y,pT) (k = 1 => central, other sets are error sets & give band via Hessian rules, k∈{central, Hessian members}).
3) K^(k)(b,y,pT) = S_AWS^{(k)}(b,y,pT)/S_A^(k)(y,pT) where:
    Attach S_A^(k)(y,pT) are the gluon pdf sets computed from gluon_ratio module. We can use these to get S_AWS^(k)(b,y,pT) = 1 + N (S^(k)_A-1) α(b).
     -- set-matched EPPS21 through GluonEPPSProvider (recommended) (either central set or all sets (central + error sets))
4) Map to centrality with Woods–Saxon thickness: (for each sets of gluon ratios sets)
   K^(k) = [1 + N (S^(k)_A-1) α(b)] / S^(k)_A and integrate with σ^(k)_pA weights.

Matched default
---------------
Once you call:
    cm.enable_matched_from_analysis(ana, df_pp, df_pa, df_errors, epps_provider)
all public RpA-vs-(b/centrality/y/pT) helpers will use the same EPPS set id
as each σ-member. Disable with `cm.disable_matched()`.
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
# utils
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def ensure_out(tag: str) -> str:
    folder = f"./output-{tag}"
    return ensure_dir(folder)

def round_grid(df: pd.DataFrame, y_dec=3, pt_dec=3) -> pd.DataFrame:
    out = df.copy()
    out["y_r"] = out["y"].round(y_dec)
    out["pt_r"] = out["pt"].round(pt_dec)
    return out

def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    v = np.asarray(values,  float)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
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
            nrows=len(df), ny=len(ys), npt=len(pts),
            y_min=float(np.min(ys)), y_max=float(np.max(ys)),
            pt_min=float(np.min(pts)), pt_max=float(np.max(pts)),
        )

# ---------------------------------------------------------------------------
# TopDrawer I/O
# ---------------------------------------------------------------------------

_NUM_RE = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

@dataclass
class TopData:
    df: pd.DataFrame               # columns: y, pt, val, err
    y_edges: np.ndarray            # left edges per y-slab
    y_centers: np.ndarray          # y-centers per slab
    header: str

def _sections(text: str) -> List[Tuple[int, int]]:
    starts = [m.end() for m in re.finditer(r"SET ORDER X Y DY", text)]
    ends   = [m.end() for m in re.finditer(r"HIST SOLID", text)]
    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n)]

def _y_cut(block_text: str) -> Optional[Tuple[float, float]]:
    m = re.search(rf"({_NUM_RE})\s*<\s*y\s*<\s*({_NUM_RE})", block_text)
    if not m: return None
    return float(m.group(1)), float(m.group(2))

def _parse_numeric(block_text: str) -> np.ndarray:
    rows = []
    for ln in block_text.splitlines():
        ln = ln.strip()
        if not ln or not re.match(rf"^{_NUM_RE}", ln): continue
        parts = ln.split()
        if len(parts) < 3: continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except Exception:
            pass
    return np.asarray(rows, dtype=float)

def _pairwise(it: Iterable):
    it = iter(it)
    return zip(it, it)

def load_top_file(path: str | Path, kick: str = "pp", drop_last_pairs: int = 2) -> TopData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[TopDrawer] File not found: {path}")
    text = path.read_text()

    secs = _sections(text)
    if not secs:
        raise ValueError(f"[TopDrawer] No data sections found in {path}")

    blocks = [text[s:e] for (s, e) in secs]
    pairs  = list(_pairwise(blocks))
    pairs  = [p for p in pairs if len(p) == 2]
    if drop_last_pairs and len(pairs) > drop_last_pairs:
        pairs = pairs[:-drop_last_pairs]

    # NEW (drop-in):
    k = kick.lower()
    if k in ("pp", "no", "none", "baseline"):
        sel = 0                      # first in pair = "pp kick"
        # print(f"[TopDrawer] Using kick='{kick}' => 'pp' selected.")
    elif k in ("dpt", "+dpt", "broad", "broadening", "pa", "nuc"):
        sel = 1                      # second in pair = "+dpt kick"
        # print(f"[TopDrawer] Using kick='{kick}' => '+dpt' selected.")
    else:
        sel = 0
        print(f"[TopDrawer] Unknown kick='{kick}', defaulting to 'pp'.")

    chosen = [p[sel] for p in pairs]

    y_cuts, slabs = [], []
    for blk in chosen:
        yc = _y_cut(blk) or _y_cut(blk[:400])
        if yc is None:
            raise ValueError(f"[TopDrawer] Could not find y-cut in a block of {path.name}")
        y_cuts.append(yc)

        arr = _parse_numeric(blk)
        if arr.size == 0: continue

        y_mean = 0.5 * (yc[0] + yc[1])
        ycol = np.full(arr.shape[0], y_mean, dtype=float)
        slabs.append(np.c_[ycol, arr])  # [y, pt, val, err]

    if not slabs:
        raise ValueError(f"[TopDrawer] No numeric data parsed from {path}")

    data = np.vstack(slabs)
    df = (pd.DataFrame(data, columns=["y","pt","val","err"])
            .sort_values(["y","pt"], kind="mergesort")
            .reset_index(drop=True))

    y_left_edges = np.array([lc for (lc, rc) in y_cuts], dtype=float)
    y_centers    = np.array(sorted(df["y"].unique()), dtype=float)

    m = re.search(r"^\s*TITLE\s+(.+)$", text, flags=re.MULTILINE)
    header = m.group(1).strip() if m else path.name

    return TopData(df=df, y_edges=y_left_edges, y_centers=y_centers, header=header)

def discover_by_number(folder: str | Path) -> List[Tuple[str, int]]:
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

read_topdrawer = load_top_file  

# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

@dataclass
class NPDFSystem:
    """
    Folder layout (sorted by e21NN):
      0: pp central, 1: pA central, 2..: pA error sets
    """
    pp_path: str
    pa_path: str
    error_paths: List[str]
    kick: str = "pp" # dpt
    name: str = "system"

    df_pp: pd.DataFrame = field(init=False)
    df_pa: pd.DataFrame = field(init=False)
    df_errors: List[pd.DataFrame] = field(init=False)
    y_edges: Optional[pd.Series] = field(init=False, default=None)
    y_centers: Optional[pd.Series] = field(init=False, default=None)

    def load(self) -> "NPDFSystem":
        pp = load_top_file(self.pp_path, kick=self.kick)
        pa = load_top_file(self.pa_path, kick=self.kick)
        errs = [load_top_file(p, kick=self.kick).df for p in self.error_paths]

        # normalize dtypes & sort
        for df in [pp.df, pa.df] + errs:
            for c in ("y","pt","val","err"):
                if c not in df.columns:
                    raise ValueError(f"[NPDFSystem] Missing column {c}")
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.sort_values(["y","pt"], inplace=True, kind="mergesort")
            df.reset_index(drop=True, inplace=True)

        self.df_pp = pp.df.copy()
        self.df_pa = pa.df.copy()
        self.df_errors = [d.copy() for d in errs]
        # keep these for plotting (your notebook uses them)
        self.y_edges = pd.Series(pp.y_edges)
        self.y_centers = pd.Series(pp.y_centers)
        return self

    @staticmethod
    def from_folder(folder: str, kick: str = "pp", name: str = "system") -> "NPDFSystem":
        files = discover_by_number(folder)
        if len(files) < 2:
            raise ValueError(f"[NPDFSystem] Need at least 2 files (pp central + pA central) in {folder}")
        pp_central = files[0][0]
        pa_central = files[1][0]
        pa_errors  = [p for (p, _) in files[2:]]
        return NPDFSystem(pp_central, pa_central, pa_errors, kick=kick, name=name).load()

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
                         include_members: bool = True) -> pd.DataFrame:
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

    def compute_rpa_members(self, df_pp, df_pa, df_errors, join="intersect"):
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
        M = np.stack(mems, axis=0) if mems else np.empty((0, len(r0)))
   
        return base, r0, M

# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

def style_axes(ax, xlab, ylab, grid: bool=True, logx: bool=False, logy: bool=False, title: Optional[str]=None):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    if grid: ax.grid(True, which="both", alpha=0.25)
    if title: ax.set_title(title)

def step_band_xy(ax, x, y_c, y_lo, y_hi, label: Optional[str]=None, color=None):
    lines = ax.step(x, y_c, where="post", label=label, color=color, linewidth=2)
    line_color = color if color is not None else lines[0].get_color()
    ax.fill_between(x, y_lo, y_hi, step="post", alpha=0.25, color=line_color)

def overlay_error_members(ax, xs, members: np.ndarray, color=None, alpha: float=0.12, lw: float=1.0):
    if members.ndim != 2: return
    for i in range(members.shape[0]):
        ax.plot(xs, members[i], color=color, alpha=alpha, lw=lw)

def slice_nearest_pt_for_each_y(df: pd.DataFrame, pt_target: float) -> pd.DataFrame:
    out = []
    for y, g in df.groupby("y"):
        j = (g["pt"] - pt_target).abs().idxmin()
        out.append(df.loc[j])
    return pd.DataFrame(out).sort_values("y")

def band_xy(ax, x, y_c, y_lo, y_hi, label: Optional[str]=None, color=None):
    x = np.asarray(x, float); order = np.argsort(x)
    x   = x[order]
    y_c = np.asarray(y_c, float)[order]
    y_lo = np.asarray(y_lo, float)[order]
    y_hi = np.asarray(y_hi, float)[order]
    (line_obj,) = ax.plot(x, y_c, lw=2, label=label, color=color)
    line_color = color if color is not None else line_obj.get_color()
    ax.fill_between(x, y_lo, y_hi, alpha=0.25, color=line_color)

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

def _parse_centbin(s: str) -> Tuple[float,float]:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*%", str(s))
    if not m: return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))

def plot_rpa_vs_centrality_hzerr(ax, df: pd.DataFrame, label: Optional[str]=None, color=None, markers: bool=True):
    if df is None or len(df)==0: return ax
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

def plot_K_vs_b(ax, cm: "CentralityModel", y: float, pT: float, b_values=None, *, label=None, color=None):
    ktab = cm.K_vs_b(y, pT, b_values=b_values)
    ax.plot(ktab["b"], ktab["K"], lw=2, label=label, color=color)
    style_axes(ax, r"$b$ [fm]", r"$K(b;y,p_T)$", grid=True)
    return ax

def step_K_vs_y(ax, df: pd.DataFrame, *, color=None, label=None):
    edges = np.r_[df["y_left"].to_numpy(float), df["y_right"].iloc[-1]]
    vals  = df["K"].to_numpy(float)
    x, y  = edges, np.r_[vals, vals[-1]]
    ax.plot(x, y, drawstyle="steps-post", lw=2, color=color, label=label)
    style_axes(ax, r"$y$", r"$K$", grid=True); ax.set_xlim(edges[0], edges[-1])
    return ax
# ---------------------------------------------------------------------------
# centrality & gluon attachments
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
    try:
        df = pd.read_csv(path, comment="#", header=None, engine="python",
                         sep=r"[\s,]+", dtype=str, on_bad_lines="skip")
    except Exception:
        df = pd.DataFrame()
    df = df.dropna(how="all")
    if df.shape[1] < 3:
        try:
            df = pd.read_csv(path, comment="#", header=None, engine="python",
                             sep=r"\t+|\s{2,}|,", dtype=str, on_bad_lines="skip").dropna(how="all")
        except Exception:
            pass
    if df.shape[1] < 3:
        raise RuntimeError(f"{path} does not look like 3-column x Q S table.")
    df = df.iloc[:, :3]; df.columns = ["x","Qcol","S"]
    for col in ("x","Qcol","S"):
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r"[;,]+$", "", regex=True)
        df[col] = df[col].str.replace("D","E", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{path} parsed to an empty numeric table after cleaning.")
    return df

@dataclass
class GluonRatioTable:
    path: Path
    sqrt_sNN_GeV: float
    m_jpsi_GeV: float = 3.43
    q_is_q2: Optional[bool] = False
    y_sign_for_xA: int = -1
    _interp: Optional[Bilinear2D] = None
    _x_grid: Optional[np.ndarray] = None
    _q_grid: Optional[np.ndarray] = None

    def load(self) -> "GluonRatioTable":
        df = _read_xq_table(self.path)
        df = df[df["S"] > 0].copy()

        if self.q_is_q2 is None:
            qvals = np.sort(df["Qcol"].unique())
            self.q_is_q2 = (np.nanmax(qvals) > 100.0)

        x_vals = np.sort(df["x"].unique())
        q_vals = np.sort(df["Qcol"].unique())

        V = (df.pivot(index="Qcol", columns="x", values="S")
               .reindex(index=q_vals, columns=x_vals).sort_index().sort_index(axis=1))
        V = V.interpolate(axis=0, limit_direction="both").interpolate(axis=1, limit_direction="both")
        V_np = V.to_numpy(dtype=float)

        self._interp = Bilinear2D(x_vals, q_vals, V_np)
        self._x_grid, self._q_grid = x_vals, q_vals

        Smin, Smax = np.nanmin(V_np), np.nanmax(V_np)
        if not np.isfinite(Smin) or not np.isfinite(Smax):
            raise RuntimeError(f"{self.path} produced non-finite S_A grid.")
        if Smin <= 0:
            print(f"[GluonRatioTable] WARNING: S_A grid has values ≤ 0 (min={Smin:.3g}). Will clip at evaluation.")
        return self

    # in GluonRatioTable dataclass add:
    kinematics: str = "2to2"  # "2to2" (default), "2to1" (legacy)

    # --- convention hook (optional; defaults remain intact) ---
    def set_convention(self, *, y_sign_for_xA: int = -1, kinematics: str = "2to2") -> "GluonRatioTable":
        """
        Ensure (y, pT) -> x_A uses the same convention as the σ tables.
        - y_sign_for_xA=-1 : A at negative rapidity (your default, p in +y)
        - kinematics="2to2" : x_A = 2 m_T / sqrt(s) * exp(sign*y)  (your default)
        """
        self.y_sign_for_xA = int(y_sign_for_xA)
        self.kinematics = str(kinematics)
        return self
    
    def x_of(self, y: float, pT: float) -> float:
        """
        Map (y, pT) -> x_A (target-side Bjorken x).

        Convention:
        - If y_sign_for_xA = -1 (default: A at negative rapidity), use exp(-y).
        - If y_sign_for_xA = +1 (A at positive rapidity), use exp(+y).

        Kinematics:
        - "2to1": x_A = (m_T / sqrt(s)) * exp(s*y)
        - "2to2": x_A = (2 m_T / sqrt(s)) * exp(s*y)
        """
        mT = math.hypot(self.m_jpsi_GeV, float(pT))
        sgn = float(self.y_sign_for_xA)  # -1 if A at negative y, +1 if A at positive y
        kin = (self.kinematics or "2to2").lower()

        if kin.startswith("2to1"):
            x = (mT / self.sqrt_sNN_GeV) * math.exp(sgn * y)
        elif kin.startswith("2to2"):
            x = (2.0 * mT / self.sqrt_sNN_GeV) * math.exp(sgn * y)
        else:
            # Fallback to 2->2 as the default physics choice
            x = (2.0 * mT / self.sqrt_sNN_GeV) * math.exp(sgn * y)

        # Clip to table/grid domain and to physical [~0, 1]
        xmin = float(self._x_grid[0])  if self._x_grid is not None else 1e-7
        xmax = float(self._x_grid[-1]) if self._x_grid is not None else 1.0
        x = float(np.clip(x, xmin, min(xmax, 1.0)))

        return x


    def Qgrid_of(self, pT: float) -> float:
        mT = math.hypot(self.m_jpsi_GeV, pT)
        return mT*mT if self.q_is_q2 else mT

    def SA_ypt(self, y: float, pT: float) -> float:
        assert self._interp is not None, "Call .load() first"
        x = self.x_of(y, pT)
        q = self.Qgrid_of(pT)
        S = float(self._interp(x, q))
        return max(S, 1e-8)

    # quick grids for plotting
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
                    "y": float(yc[i]), "pt": float(pc[j]),
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

    def P_inel(self, T_b: float, sigmaNN_mb: float = 71.0) -> float:
        s = self.sigma_mb_to_fm2(sigmaNN_mb)
        return 1.0 - math.exp(- s * T_b)

    def Npart_pA(self, T_b: float, sigmaNN_mb: float = 71.0) -> float:
        s = self.sigma_mb_to_fm2(sigmaNN_mb)
        tgt = self.A * (1.0 - math.exp(-s*T_b/self.A))
        return 1.0 + tgt

    def b_pdf(self, T_grid: pd.DataFrame, sigmaNN_mb: float = 71.0) -> pd.DataFrame:
        b = T_grid["b"].to_numpy()
        T = T_grid["T"].to_numpy()
        pinel = np.array([self.P_inel(Ti, sigmaNN_mb) for Ti in T], dtype=float)
        w = 2.0 * math.pi * b * pinel
        Z = float(np.trapezoid(w, b))
        if Z > 0: w = w / Z
        return pd.DataFrame({"b": b, "T": T, "P_inel": pinel, "w": w})

    def b_edges_for_percentiles(self, T_grid: pd.DataFrame, percentiles: Sequence[float], sigmaNN_mb: float = 71.0
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
            if j <= 0: out.append(float(b[0])); continue
            if j >= len(b): out.append(float(b[-1])); continue
            denom = (cdf[j] - cdf[j-1])
            frac = 0.0 if denom == 0 else (t - cdf[j-1]) / denom
            out.append(float((1.0-frac)*b[j-1] + frac*b[j]))
        return np.asarray(out, dtype=float)

def _wavg_safe(df, col):
    v = np.asarray(df[col], float)
    w = np.clip(np.asarray(df["w"], float), 0.0, None)
    m = np.isfinite(v) & (w > 0.0)
    s = float(w[m].sum())
    return float((v[m] * w[m]).sum() / s) if s > 0.0 else np.nan

@dataclass
class CentralityModel:
    """
    RpA^npdf (k set) (y,pT;b) = [1 + N (S_A-1) (k set) α(b)] / S_A (k set) × (σ_pA/σ_pp).
    """
    gluon: object
    geom: WoodsSaxonPb
    T_grid: pd.DataFrame
    Nnorm: float

    _pdf_cache: Optional[pd.DataFrame] = None
    _alpha_cache: Dict[Tuple[Tuple[int,...], float, str], List[Tuple[str,float]]] = None
    _matched: Optional[dict] = None
    _W_BUILDER = staticmethod(RpAAnalysis._make_weight_table)  # reuse weight logic (pa/pp)
    @classmethod
    def from_inputs(cls, gluon: object, geom: WoodsSaxonPb, b_max: float = 12.0, nb: int = 601, N_override: Optional[float] = None
                    ) -> "CentralityModel":
        T_grid = geom.make_T_grid(b_max=b_max, nb=nb)
        Nnorm = geom.normalization_N(T_grid) if N_override is None else float(N_override)
        return cls(gluon=gluon, geom=geom, T_grid=T_grid, Nnorm=Nnorm, _pdf_cache=None, _alpha_cache={}, _matched=None)

    # --- matched-mode wiring ---
    def enable_matched(self, base_xy: pd.DataFrame, r0: np.ndarray, members: np.ndarray,
                       epps_provider, set_ids=None, *, centralK: bool = False):
        if not isinstance(base_xy, pd.DataFrame) or not {"y","pt"}.issubset(base_xy.columns):
            raise ValueError("[enable_matched] base_xy must have ['y','pt']")
        r0 = np.asarray(r0, float)
        members = np.asarray(members, float)
        if members.ndim != 2 or members.shape[1] != r0.shape[0]:
            raise ValueError(f"[enable_matched] members shape {members.shape} incompatible with r0 {r0.shape}")
        if set_ids is not None and len(set_ids) != members.shape[0]:
            raise ValueError("[enable_matched] len(set_ids) must equal number of member rows")
        self._matched = {
            "base_xy": base_xy.copy().reset_index(drop=True),
            "r0":      r0.copy(),
            "members": members.copy(),
            "epps":    epps_provider,
            "set_ids": list(set_ids) if set_ids is not None else None,
            "centralK": bool(centralK),
        }

        # Make EPPS provider the active gluon source by default
        try:
            from gluon_ratio import GluonEPPSProvider
            if isinstance(epps_provider, GluonEPPSProvider):
                self.gluon = epps_provider
        except Exception:
            # Fall back silently; user can still set .gluon manually
            self.gluon = epps_provider

        return self
    
    ##------Debugger for point evaluations -----
    def debug_point(self, y, pt, b, df_pp, df_pa, df_errors=None, gluon=None, msg="DEBUG"):
        # Nearest (y,pt) on the pp grid:
        gpp = df_pp[["y","pt","val"]].to_numpy(float)
        d2  = (gpp[:,0]-y)**2 + (gpp[:,1]-pt)**2
        j   = int(np.argmin(d2))
        y0, p0 = float(gpp[j,0]), float(gpp[j,1])

        # Values at that grid point
        pp  = float(df_pp[(df_pp["y"]==y0)&(df_pp["pt"]==p0)]["val"].values[0])
        pa  = float(df_pa[(df_pa["y"]==y0)&(df_pa["pt"]==p0)]["val"].values[0])
        r0  = pa/pp if pp != 0.0 else float("nan")

        # Members at that point (if provided)
        mem_vals = []
        if df_errors:
            for k, dfe in enumerate(df_errors, start=1):
                val = float(dfe[(dfe["y"]==y0)&(dfe["pt"]==p0)]["val"].values[0])
                mem_vals.append(val/pp if pp>0 else float("nan"))

        # SA (central + maybe one error set to spot-check)
        if gluon is None: gluon = getattr(self, "gluon", None)
        SA_c = float(gluon.SA_ypt(y0, p0)) if hasattr(gluon,"SA_ypt") \
            else float(gluon.SA_ypt_set([y0],[p0], set_id=1)[0]) if hasattr(gluon,"SA_ypt_set") else 1.0

        # Geometry
        alpha = float(self.alpha_of_b(b))
        Kc    = (1.0 + self.Nnorm*(SA_c - 1.0)*alpha) / max(SA_c, 1e-12)
        RpA_c = r0 * Kc

        print(f"[{msg}] at (y,pt,b)=({y0:.3f},{p0:.3f},{b:.2f})")
        print(f"  pp={pp:.6e}  pa={pa:.6e}  r0=pa/pp={r0:.6f}")
        print(f"  SA_c={SA_c:.6f}  alpha(b)={alpha:.6f}  Nnorm={self.Nnorm:.6f}  Kc={Kc:.6f}")
        print(f"  RpA_c = r0*Kc = {RpA_c:.6f}")
        if mem_vals:
            print(f"  #members={len(mem_vals)}; first 6 r0(mem)={mem_vals[:6]}")

    def enable_matched_from_analysis(self, analysis, df_pp, df_pa, df_errors, epps_provider, join: str = "intersect",
                                     set_ids=None, *, centralK: bool = False):
        base_xy, r0, members = analysis.compute_rpa_members(df_pp, df_pa, df_errors, join=join)
        if set_ids is None and hasattr(epps_provider, "nuclear_set_ids"):
            set_ids = epps_provider.nuclear_set_ids()[:members.shape[0]]  # e.g., [2..49]
        return self.enable_matched(base_xy, r0, members, epps_provider, set_ids=set_ids, centralK=centralK)

    def disable_matched(self):
        self._matched = None
        return self

    # --- cached geometry ---
    def pdf_table(self, sigmaNN_mb: float) -> pd.DataFrame:
        if (self._pdf_cache is None
            or "sigmaNN_mb" not in self._pdf_cache.attrs
            or self._pdf_cache.attrs["sigmaNN_mb"] != sigmaNN_mb):
            pdf = self.geom.b_pdf(self.T_grid, sigmaNN_mb=sigmaNN_mb)
            pdf.attrs["sigmaNN_mb"] = sigmaNN_mb
            self._pdf_cache = pdf
        return self._pdf_cache

    # --- Hessian combiner (pairwise if even members; NaN-safe) ---
    def _hessian_band(self, member_vals, central, *, pairwise=True):
        mv = np.asarray(member_vals, float)
        mv = mv[np.isfinite(mv)]
        if mv.size == 0:
            return float(central), float(central)
        if pairwise and (mv.size % 2 == 0):
            D = mv[0::2] - mv[1::2]
            h = 0.5 * np.sqrt(np.sum(D * D))
            return float(central - h), float(central + h)
        diff = mv - central
        err_plus  = np.sqrt(np.sum(np.clip(diff,  0, None) ** 2))
        err_minus = np.sqrt(np.sum(np.clip(-diff, 0, None) ** 2))
        return float(central - err_minus), float(central + err_plus)

    # --- robust Hessian via explicit (+,-) pairs ---
    def _get_hessian_pairs(self, epps_provider, set_ids: list[int]) -> list[tuple[int,int]]:
        """
        Return explicit (plus_id, minus_id) pairs for Hessian combination.
        Priority:
          1) epps_provider.hessian_pairs   (e.g., [(2,3),(4,5),...])
          2) default: consecutive pairs starting at 2 based on len(set_ids)
        """
        # Provider-defined pairs?
        pairs = getattr(epps_provider, "hessian_pairs", None)
        if pairs:
            return [(int(a), int(b)) for (a, b) in pairs]

        # Fallback to conventional EPPS21 ordering:
        # central=1, then (2,3), (4,5), ..., covering all error members.
        # We only keep pairs that are actually present in set_ids.
        sids = set(int(s) for s in set_ids)
        out = []
        # try to cover as many as possible in the typical pattern
        for a in range(2, 2 + 2*len(set_ids), 2):
            b = a + 1
            if (a in sids) and (b in sids):
                out.append((a, b))
        # If the above produced nothing (exotic ordering), fall back to consecutive IDs in set_ids.
        if not out and len(set_ids) >= 2:
            for i in range(0, len(set_ids)-1, 2):
                out.append((int(set_ids[i]), int(set_ids[i+1])))
        return out

    def _hessian_from_pairs(self, values_by_id: dict[int, float], pairs: list[tuple[int,int]], central: float
                            ) -> tuple[float, float]:
        """
        Proper Hessian 1-sigma from explicit (+,-) pairs:
            Δ_i = (R_i+ - R_i-) / 2,   σ = sqrt(sum_i Δ_i^2)
        Returns (lo, hi) = (central - σ, central + σ).
        Falls back to envelope if no valid pairs found.
        """
        deltas = []
        for (p, m) in pairs:
            if (p in values_by_id) and (m in values_by_id):
                deltas.append(0.5 * (float(values_by_id[p]) - float(values_by_id[m])))
        if deltas:
            sigma = float(np.sqrt(np.sum(np.square(deltas))))
            return (float(central - sigma), float(central + sigma))

        # Fallback: symmetric envelope around central
        diffs = [float(values_by_id[k]) - float(central) for k in values_by_id]
        if not diffs:
            return (float(central), float(central))
        pos = np.sqrt(np.sum(np.square([max(d, 0.0) for d in diffs])))
        neg = np.sqrt(np.sum(np.square([min(d, 0.0) for d in diffs])))
        return (float(central - neg), float(central + pos))

    def _alphas_for_bins(self, cent_edges_pct, sigmaNN_mb: float = 71.0, weight: str = "inelastic"):
        """
        Return [(label, alpha_bin), ...] with alpha_bin = <T/T0>_bin averaged using the
        *N_coll* weight 2π b T(b). This is the only choice consistent with the
        normalization used to define Nnorm via ∫ 2π b T^2(b) db.

        NOTE: The `weight` argument here still controls only how the *bin edges*
        are defined via the cdf (inelastic percentiles), but the alpha average
        inside each bin is always performed with 2π b T(b).
        """
        edges_key = tuple(np.round(np.asarray(cent_edges_pct, dtype=float), 6))
        key = (edges_key, float(sigmaNN_mb), str(weight), "ncoll-alpha")
        if self._alpha_cache and key in self._alpha_cache:
            return self._alpha_cache[key]

        pdf = self.pdf_table(sigmaNN_mb)           # has columns: b, T, Pinel, w (=2π b Pinel normalized), cdf
        b = pdf["b"].to_numpy()
        T = pdf["T"].to_numpy()
        T0 = float(T[0])
        twopi_b = 2.0 * np.pi * b

        # use *N_coll* weight for the alpha average (independent of `weight`)
        W_alpha = twopi_b * T                      # proportional to Ncoll(b)

        # centrality bin edges in b are still defined from the inelastic cdf
        edges_b = self.geom.b_edges_for_percentiles(self.T_grid, cent_edges_pct, sigmaNN_mb=sigmaNN_mb)

        out = []
        for i in range(len(edges_b) - 1):
            bl, br = edges_b[i], edges_b[i + 1]
            m = (b >= bl) & (b < br) if i < len(edges_b) - 2 else (b >= bl) & (b <= br)
            if not np.any(m):
                out.append((f"{cent_edges_pct[i]}-{cent_edges_pct[i+1]}%", np.nan))
                continue

            den = float(np.trapezoid(W_alpha[m], b[m]))
            num = float(np.trapezoid((T[m] / T0) * W_alpha[m], b[m]))
            a_bin = (num / den) if den > 0.0 else np.nan
            out.append((f"{cent_edges_pct[i]}-{cent_edges_pct[i+1]}%", a_bin))

        self._alpha_cache = self._alpha_cache or {}
        self._alpha_cache[key] = out
        return out


    def centrality_table(self, cent_edges_pct: Sequence[float], sigmaNN_mb: float = 71.0) -> pd.DataFrame:
        pdf = self.pdf_table(sigmaNN_mb)
        b = pdf["b"].to_numpy(); T = pdf["T"].to_numpy(); w = pdf["w"].to_numpy()
        T0 = float(T[0])
        edges_b = self.geom.b_edges_for_percentiles(self.T_grid, cent_edges_pct, sigmaNN_mb=sigmaNN_mb)
        rows = []
        for i in range(len(edges_b)-1):
            bl, br = float(edges_b[i]), float(edges_b[i+1])
            m = (b >= bl) & (b < br)
            if not np.any(m): continue
            den = float(np.trapezoid(w[m], b[m]))
            bbar = float(np.trapezoid(b[m]*w[m], b[m]) / den) if den>0 else np.nan
            alpha = float(np.trapezoid((T[m]/T0)*w[m], b[m]) / den) if den>0 else np.nan
            Np = np.array([self.geom.Npart_pA(Ti, sigmaNN_mb) for Ti in T[m]], float)
            Np_bar = float(np.trapezoid(Np*w[m], b[m]) / den) if den>0 else np.nan
            s_fm2 = self.geom.sigma_mb_to_fm2(sigmaNN_mb)
            Ncoll_bar = float(np.trapezoid((s_fm2*T[m])*w[m], b[m]) / den) if den>0 else np.nan
            rows.append({
                "cent_bin": f"{cent_edges_pct[i]}-{cent_edges_pct[i+1]}%",
                "b_left": bl, "b_right": br, "b_mean": bbar,
                "alpha": alpha, "N_part": Np_bar, "N_coll": Ncoll_bar
            })
        return pd.DataFrame(rows)

    # --- vectorized S_A (x,Q)  ---
    def attach_SA_to_grid(self, rgrid: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Attach S_A^g(y,pT) to rgrid.

        Preference order (no breaking changes):
        1) gluon_ratio provider exposing SA_ypt_set(y,pt,set_id=1)
        2) GluonRatioTable ASCII (x,Q) with SA_ypt(...)
        """
        rg = rgrid.copy()
        yv = rg["y"].to_numpy(float)
        pv = rg["pt"].to_numpy(float)

        # 1) Preferred: official provider from gluon_ratio.py
        if hasattr(self.gluon, "SA_ypt_set"):
            SA = np.asarray(self.gluon.SA_ypt_set(yv, pv, set_id=1), float)
            SA = np.where(np.isfinite(SA) & (SA > 0.0), SA, 1.0)
            rg["SA"] = SA
            return rg

        # 2) Generic ASCII (x,Q) table class
        if hasattr(self.gluon, "SA_ypt"):
            SA = np.array([float(self.gluon.SA_ypt(y, p)) for y, p in zip(yv, pv)], float)
            SA = np.where(np.isfinite(SA) & (SA > 0.0), SA, 1.0)
            rg["SA"] = SA
            return rg

        if verbose:
            print("[attach_SA] No recognized gluon provider; using SA=1.0")
        rg["SA"] = 1.0
        return rg


    # --- α(b) and S_AWS ---
    def alpha_of_b(self, b_val: float) -> float:
        b = self.T_grid["b"].to_numpy()
        T = self.T_grid["T"].to_numpy()
        T0 = float(T[0])
        Tb = float(np.interp(b_val, b, T))
        return Tb / T0 if T0 > 0 else 0.0

    def SAWS_ypt_b(self, y: float, pT: float, b_val: float) -> float:
        # robustly fetch SA regardless of provider type
        if hasattr(self.gluon, "SA_ypt"):
            SA = float(self.gluon.SA_ypt(y, pT))
        elif hasattr(self.gluon, "SA_ypt_set"):
            import numpy as _np
            SA = float(_np.asarray(self.gluon.SA_ypt_set([y], [pT], set_id=1), float)[0])
        else:
            SA = 1.0
        alpha = self.alpha_of_b(float(b_val))
        return 1.0 + self.Nnorm * (SA - 1.0) * alpha


    # --- convenience tables for S_A / S_AWS and their ratio ---
    def SAWS_vs_b(self, y, pt, b_values=None):
        if b_values is None: b_values = np.linspace(0.0, 20.0, 201)
        SA = float(self.gluon.SA_ypt(y, pt))
        rows = []
        for b in np.asarray(b_values, float):
            a = float(self.alpha_of_b(b))
            SAWS = 1.0 + self.Nnorm * (SA - 1.0) * a
            rows.append(dict(b=b, alpha=a, SA=SA, SAWS=SAWS, ratio=SAWS/SA if SA!=0 else np.nan))
        return pd.DataFrame(rows)

    def SA_vs_y_fixed_pt(self, y_edges: Sequence[float], pt: float) -> pd.DataFrame:
        rows=[]
        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            yc = 0.5*(yl+yr)
            rows.append({"y":yc,"y_left":yl,"y_right":yr,"SA":float(self.gluon.SA_ypt(yc, pt))})
        return pd.DataFrame(rows)

    def SA_vs_pt_fixed_y(self, pt_edges: Sequence[float], y: float) -> pd.DataFrame:
        rows=[]
        for pl, pr in zip(pt_edges[:-1], pt_edges[1:]):
            pc = 0.5*(pl+pr)
            rows.append({"pt":pc,"pt_left":pl,"pt_right":pr,"SA":float(self.gluon.SA_ypt(y, pc))})
        return pd.DataFrame(rows)

    def SAWS_over_SA_vs_y_fixed_b_pt(self, y_edges: Sequence[float], pt: float, b: float) -> pd.DataFrame:
        rows=[]
        a = self.alpha_of_b(b)
        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            yc = 0.5*(yl+yr)
            SA = float(self.gluon.SA_ypt(yc, pt))
            SAWS = 1.0 + self.Nnorm*(SA-1.0)*a
            rows.append({"y":yc,"y_left":yl,"y_right":yr,"ratio":(SAWS/SA if SA!=0 else np.nan)})
        return pd.DataFrame(rows)

    def SAWS_over_SA_vs_pt_fixed_b_y(self, pt_edges: Sequence[float], y: float, b: float) -> pd.DataFrame:
        rows=[]
        a = self.alpha_of_b(b)
        for pl, pr in zip(pt_edges[:-1], pt_edges[1:]):
            pc = 0.5*(pl+pr)
            SA = float(self.gluon.SA_ypt(y, pc))
            SAWS = 1.0 + self.Nnorm*(SA-1.0)*a
            rows.append({"pt":pc,"pt_left":pl,"pt_right":pr,"ratio":(SAWS/SA if SA!=0 else np.nan)})
        return pd.DataFrame(rows)

    def K_of(self, y: float, pT: float, b_val: float) -> float:
        SA = float(self.gluon.SA_ypt(y, pT)) if hasattr(self.gluon, "SA_ypt") else \
            (float(self.gluon.SA_ypt_set([y],[pT], set_id=1)[0]) if hasattr(self.gluon,"SA_ypt_set") else 1.0)
        SA = 1.0 if (not np.isfinite(SA) or SA <= 0) else SA
        alpha = self.alpha_of_b(float(b_val))
        return (1.0 + self.Nnorm * (SA - 1.0) * alpha) / SA

    def K_vs_b(self, y: float, pt: float, b_values=None) -> pd.DataFrame:
        df = self.SAWS_vs_b(y, pt, b_values=b_values).copy()
        df.rename(columns={"ratio":"K"}, inplace=True)
        return df[["b","alpha","SA","K"]]

    def K_vs_y_fixed_b_pt(self, y_edges, pt: float, b: float) -> pd.DataFrame:
        df = self.SAWS_over_SA_vs_y_fixed_b_pt(y_edges, pt, b).copy()
        df.rename(columns={"ratio":"K"}, inplace=True)
        return df[["y","y_left","y_right","K"]]

    def K_vs_pt_fixed_b_y(self, pt_edges, y: float, b: float) -> pd.DataFrame:
        df = self.SAWS_over_SA_vs_pt_fixed_b_y(pt_edges, y, b).copy()
        df.rename(columns={"ratio":"K"}, inplace=True)
        return df[["pt","pt_left","pt_right","K"]]

    def rpa_grid_at_b(self, rgrid: pd.DataFrame, b_val: float) -> pd.DataFrame:
        """Return RpA(b; y, pT) on the full (y,pT) grid (then you can bin)."""
        rg = self.attach_SA_to_grid(rgrid, verbose=False)
        alpha = self.alpha_of_b(float(b_val))
        return self._scale_grid_by_alpha(rg, alpha)

    # --- min-bias (no α(b), no K scaling) to overlay as a single band ---
    def minbias_in_window(
        self, rgrid, df_pa, *,
        y_min, y_max, pt_min, pt_max,
        weight_mode: Literal["pa","pp"] = "pa",
        pp_central: Optional[pd.DataFrame] = None
    ):
        cols = ["y","pt"]
        R = rgrid[cols + [c for c in rgrid.columns if c.startswith("r_")]].copy()
        W = RpAAnalysis._make_weight_table(rgrid, df_pa, df_pp=pp_central, mode=weight_mode)
        M = R.merge(W, on=cols, how="inner")
        cut = (M["y"]>=y_min)&(M["y"]<=y_max)&(M["pt"]>=pt_min)&(M["pt"]<=pt_max)
        M = M.loc[cut]
        if M.empty:
            raise ValueError("minbias_in_window: no overlapping (y,pt) points.")

        w = np.asarray(M["w"], float); Wsum = float(np.sum(w))
        rc = float(np.sum(w * np.asarray(M["r_central"], float)) / Wsum)
        mem_cols = [c for c in M.columns if c.startswith("r_mem_")]
        if mem_cols:
            mem_int = [float(np.sum(w * np.asarray(M[c], float)) / Wsum) for c in mem_cols]
            lo, hi = self._hessian_band(mem_int, rc, pairwise=True)
        else:
            lo = float(np.sum(w * np.asarray(M["r_lo"], float)) / Wsum)
            hi = float(np.sum(w * np.asarray(M["r_hi"], float)) / Wsum)
        return dict(r_central=rc, r_lo=lo, r_hi=hi)


    # --- alpha-scaling helper ---
    def plot_minbias_band(self, ax, rgrid, df_pa, *, y_min, y_max, pt_min, pt_max,
                          color="crimson", alpha=0.25, label="min-bias (npdf)", lw=1.8, zorder=1.0):
        """
        Draw a horizontal min-bias nPDF band on `ax` using the Hessian band of R0=σ_pA/σ_pp
        (no α(b) scaling). This uses the exact same weighting as your other plots.
        """
        res = self.minbias_in_window(rgrid, df_pa, y_min=y_min, y_max=y_max, pt_min=pt_min, pt_max=pt_max)
        # Respect current x-limits; if not set yet, use (0,1)
        try:
            x0, x1 = ax.get_xlim()
        except Exception:
            x0, x1 = 0.0, 1.0
        y_c, y_lo, y_hi = float(res["r_central"]), float(res["r_lo"]), float(res["r_hi"])
        ax.fill_between([x0, x1], [y_lo, y_lo], [y_hi, y_hi], color=color, alpha=alpha, zorder=zorder)
        ax.axhline(y_c, color=color, lw=lw, label=label, zorder=zorder+0.1)
        return ax

    def _scale_grid_by_alpha(self, rg_with_SA: pd.DataFrame, alpha: float) -> pd.DataFrame:
        SA = rg_with_SA["SA"].to_numpy()
        SA_safe = np.clip(SA, 1e-12, None)
        K = (1.0 + self.Nnorm * (SA - 1.0) * float(alpha)) / SA_safe

        out = rg_with_SA.copy()
        out["K"] = K  # expose K(b,y,pT)

        for col in ("r_central","r_lo","r_hi"):
            if col in out.columns:
                out[col] = out[col].to_numpy() * K

        # keep the band envelope sane after scaling
        if set(["r_central","r_lo","r_hi"]).issubset(out.columns):
            rc = out["r_central"].to_numpy()
            lo = out["r_lo"].to_numpy()
            hi = out["r_hi"].to_numpy()
            out["r_lo"] = np.minimum.reduce([lo, rc, hi])
            out["r_hi"] = np.maximum.reduce([lo, rc, hi])
        return out


    # --- DEFAULT PATH ROUTING (matched if enabled) ---
    def rpa_vs_b(self, rgrid, df_pa, y_min, y_max, pt_min, pt_max,
                 b_values: Optional[Sequence[float]] = None, sigmaNN_mb: float = 71.0, verbose: bool = True) -> pd.DataFrame:
        if self._matched is not None:
            m = self._matched
            return self.rpa_vs_b_matched(
                m['base_xy'], m['r0'], m['members'], df_pa, b_values=b_values,
                epps_provider=m['epps'], set_ids=m.get('set_ids'),
                y_min=y_min, y_max=y_max, pt_min=pt_min, pt_max=pt_max,
                sigmaNN_mb=sigmaNN_mb, verbose=verbose
            )
        if verbose:
            print(f"[rpa_vs_b] window: y∈[{y_min},{y_max}], pT∈[{pt_min},{pt_max}]")
        rg = self.attach_SA_to_grid(rgrid, verbose=False)
        msel = (rg["y"]>=y_min)&(rg["y"]<=y_max)&(rg["pt"]>=pt_min)&(rg["pt"]<=pt_max)
        rg = rg.loc[msel].copy()
        if rg.empty:
            if verbose: print("  -> no points in window; returning empty DataFrame.")
            return pd.DataFrame(columns=["b","alpha","N_part","r_central","r_lo","r_hi"])
        wtab = df_pa[["y","pt","val"]].rename(columns={"val":"w"})
        rg = rg.merge(wtab, on=["y","pt"], how="left")
        rg["w"] = rg["w"].clip(lower=0).fillna(0.0)
        
        # w_pp = np.where((rg["r_central"] > 0) & np.isfinite(rg["r_central"]),
        #                 rg["w_pa"] / rg["r_central"], 0.0)
        # rg["w"] = np.clip(w_pp, 0, None).fillna(0.0)

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
   
    def rpa_vs_centrality_integrated(
        self, rgrid, df_pa, centrality_edges_pct,
        y_min, y_max, pt_min: float = 0.0, pt_max: float = 20.0,
        sigmaNN_mb: float = 71.0, weight: str = "inelastic", verbose: bool = True,
        *,
        weight_mode: Literal["pa","pp","blend"] = "pa",
        w_floor_frac: Optional[float] = 0.05,
        pt_floor: Optional[float] = 2.0,
        pp_central: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if verbose:
            print(f"[rpa_vs_centrality_integrated] window: y∈[{y_min},{y_max}], pT∈[{pt_min},{pt_max}]  weight_mode={weight_mode}")

        rg = self.attach_SA_to_grid(rgrid, verbose=False)
        m = (rg["y"]>=y_min)&(rg["y"]<=y_max)&(rg["pt"]>=pt_min)&(rg["pt"]<=pt_max)
        if pt_floor is not None:
            m &= (rg["pt"] >= float(pt_floor))
        rgw = rg.loc[m].copy()
        if rgw.empty:
            if verbose: print("  -> no points in window; returning empty DataFrame.")
            return pd.DataFrame(columns=["cent_bin","r_central","r_lo","r_hi"])

        # build weights: σ_pA (default) or σ_pp (stable in backward/low-pT), or a 50–50 blend
        if weight_mode == "pa":
            W = self._W_BUILDER(rgrid, df_pa, mode="pa")
        elif weight_mode == "pp":
            W = self._W_BUILDER(rgrid, df_pa, df_pp=pp_central, mode="pp")
        else:  # "blend"
            W_pa = self._W_BUILDER(rgrid, df_pa, mode="pa").rename(columns={"w":"w_pa"})
            W_pp = self._W_BUILDER(rgrid, df_pa, df_pp=pp_central, mode="pp").rename(columns={"w":"w_pp"})
            W = W_pa.merge(W_pp, on=["y","pt"], how="inner")
            W["w"] = 0.5*(W["w_pa"] + W["w_pp"])
            W = W[["y","pt","w"]]

        rgw = rgw.merge(W, on=["y","pt"], how="left")
        w = rgw["w"].to_numpy(float)
        # tiny-weight floor to avoid pathologies when σ_pA ~ 0
        if (w_floor_frac is not None) and (w_floor_frac > 0.0):
            pos = w[w > 0.0]
            if pos.size:
                wmin = float(np.nanpercentile(pos, 5)) * float(w_floor_frac)
                w = np.where(w > 0.0, np.maximum(w, wmin), 0.0)
                rgw["w"] = w

        alphas = self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight)
        rows = []
        for label, alpha in alphas:
            if not np.isfinite(alpha):
                if verbose: print(f"  [warn] alpha NaN for bin {label}; skipping.")
                continue
            g = self._scale_grid_by_alpha(rgw, float(alpha))
            ww = g["w"].to_numpy(float)
            def _wavg(col):
                return _wavg_safe(g, col)
            rows.append({"cent_bin":label,
                         "r_central": _wavg("r_central"),
                         "r_lo":      _wavg("r_lo"),
                         "r_hi":      _wavg("r_hi")})
            if verbose:
                print(f"  bin {label}: alpha={alpha:.6f}, points={len(g)}, wsum={np.sum(ww):.3e}")
        return pd.DataFrame(rows)


    # --- matched variants (set-matched SA^g_k) ---
    def rpa_vs_centrality_integrated_matched(
        self, base_xy, r0, members, df_pa, centrality_edges_pct,
        epps_provider, set_ids=None,
        y_min=-5, y_max=5, pt_min=0.0, pt_max=20.0,
        sigmaNN_mb=71.0, weight="inelastic", verbose=False
    ):
        base = base_xy.copy()
        msel = ((base["y"]>=y_min)&(base["y"]<=y_max)&(base["pt"]>=pt_min)&(base["pt"]<=pt_max))
        if not np.any(msel):
            return pd.DataFrame(columns=["cent_bin","r_central","r_lo","r_hi"])

        y  = base.loc[msel,"y"].to_numpy(float)
        pt = base.loc[msel,"pt"].to_numpy(float)

        wtab = df_pa.merge(base[msel], on=["y","pt"], how="inner")[["y","pt","val"]]
        w = wtab["val"].to_numpy(float)
        if not np.any(w>0):
            return pd.DataFrame(columns=["cent_bin","r_central","r_lo","r_hi"])

        # Central K on the same (y,pt)
        SA_c  = epps_provider.SA_ypt_set(y, pt, set_id=1)
        use_centralK = bool(self._matched.get("centralK", False))
        r0_sel = r0[msel.to_numpy()]

        # Map member-row -> set_id, and build (+,-) pairs
        Kmem = members.shape[0]
        set_ids = set_ids or list(range(2, Kmem+2))
        if len(set_ids) != Kmem:
            raise ValueError("[matched] set_ids length must equal #member rows")
        pairs = self._get_hessian_pairs(epps_provider, set_ids)

        rows = []
        for label, alpha in self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight):
            if not np.isfinite(alpha):
                if verbose: print(f"  [warn] alpha NaN for bin {label}; skipping.")
                continue
            alpha = float(alpha)

            # central
            Kc = (1.0 + self.Nnorm*(SA_c - 1.0)*alpha) / np.clip(SA_c, 1e-12, None)
            Rc = r0_sel * Kc
            r_c = float(np.sum(Rc*w)/np.sum(w)) if np.sum(w)>0 else np.nan

            # each error member (build dict by set_id)
            vals_by_id = {}
            for j, sid in enumerate(set_ids):
                if use_centralK:
                    Kj = Kc
                else:
                    SAj = epps_provider.SA_ypt_set(y, pt, set_id=int(sid))
                    Kj  = (1.0 + self.Nnorm*(SAj - 1.0)*alpha) / np.clip(SAj, 1e-12, None)
                Rj  = members[j, msel.to_numpy()] * Kj
                vals_by_id[int(sid)] = float(np.sum(Rj*w)/np.sum(w)) if np.sum(w)>0 else np.nan

            lo, hi = self._hessian_from_pairs(vals_by_id, pairs, r_c)
            rows.append({"cent_bin": label, "r_central": r_c, "r_lo": lo, "r_hi": hi})

        return pd.DataFrame(rows)

    def rpa_vs_b_matched(
        self, base_xy, r0, members, df_pa, *,
        epps_provider, set_ids=None, y_min=-5.0, y_max: float = 5.0, pt_min: float = 0.0, pt_max: float = 20.0,
        b_values: Optional[Sequence[float]] = None, sigmaNN_mb=71.0, verbose=False
    ) -> pd.DataFrame:
        base = base_xy.copy()
        msel = ((base["y"]>=y_min)&(base["y"]<=y_max)&(base["pt"]>=pt_min)&(base["pt"]<=pt_max))
        if not np.any(msel):
            return pd.DataFrame(columns=["b","alpha","N_part","r_central","r_lo","r_hi"])

        y  = base.loc[msel,"y"].to_numpy(float)
        pt = base.loc[msel,"pt"].to_numpy(float)

        wtab = df_pa.merge(base[msel], on=["y","pt"], how="inner")[["y","pt","val"]]
        w = wtab["val"].to_numpy(float)
        if not np.any(w>0):
            return pd.DataFrame(columns=["b","alpha","N_part","r_central","r_lo","r_hi"])

        SA_c  = epps_provider.SA_ypt_set(y, pt, set_id=1)
        use_centralK = bool(self._matched.get("centralK", False))
        r0_sel = r0[msel.to_numpy()]

        Kmem = members.shape[0]
        set_ids = set_ids or list(range(2, Kmem+2))
        if len(set_ids) != Kmem:
            raise ValueError("[matched] set_ids length must equal #member rows")
        pairs = self._get_hessian_pairs(epps_provider, set_ids)

        if b_values is None:
            b_values = self.T_grid["b"].to_numpy()
        rows = []
        T_b = self.T_grid[["b","T"]].to_numpy(float)

        for bv in b_values:
            alpha = self.alpha_of_b(float(bv))
            Kc = (1.0 + self.Nnorm*(SA_c - 1.0)*alpha) / np.clip(SA_c, 1e-12, None)
            Rc = r0_sel * Kc
            r_c = float(np.sum(Rc*w)/np.sum(w)) if np.sum(w)>0 else np.nan

            vals_by_id = {}
            for j, sid in enumerate(set_ids):
                if use_centralK:
                    Kj = Kc
                else:
                    SAj = epps_provider.SA_ypt_set(y, pt, set_id=int(sid))
                    Kj  = (1.0 + self.Nnorm*(SAj - 1.0)*alpha) / np.clip(SAj, 1e-12, None)
                Rj  = members[j, msel.to_numpy()] * Kj
                vals_by_id[int(sid)] = float(np.sum(Rj*w)/np.sum(w)) if np.sum(w)>0 else np.nan

            lo, hi = self._hessian_from_pairs(vals_by_id, pairs, r_c)
            Tb = float(np.interp(float(bv), T_b[:,0], T_b[:,1]))
            Np = self.geom.Npart_pA(Tb, sigmaNN_mb=sigmaNN_mb)

            rows.append({"b": float(bv), "alpha": float(alpha), "N_part": float(Np),
                         "r_central": r_c, "r_lo": lo, "r_hi": hi})
        return pd.DataFrame(rows)


    def rpa_vs_y_in_centrality_bins(
        self, rgrid, df_pa, centrality_edges_pct,
        y_width: float = 0.5, pt_min: Optional[float] = 0.0, pt_max: Optional[float] = None,
        sigmaNN_mb: float = 71.0, weight: str = "inelastic", verbose: bool = True,
        *,
        weight_mode: Literal["pa","pp","blend"] = "pa",
        w_floor_frac: Optional[float] = 0.0,
        pt_floor: Optional[float] = None,
        pp_central: Optional[pd.DataFrame] = None
    ) -> List[Tuple[str, pd.DataFrame]]:
        rg = self.attach_SA_to_grid(rgrid, verbose=False)

        # set a safe weight table once and re-use
        if weight_mode == "pa":
            W = self._W_BUILDER(rgrid, df_pa, mode="pa")
        elif weight_mode == "pp":
            W = self._W_BUILDER(rgrid, df_pa, df_pp=pp_central, mode="pp")
        else:
            W_pa = self._W_BUILDER(rgrid, df_pa, mode="pa").rename(columns={"w":"w_pa"})
            W_pp = self._W_BUILDER(rgrid, df_pa, df_pp=pp_central, mode="pp").rename(columns={"w":"w_pp"})
            W = W_pa.merge(W_pp, on=["y","pt"], how="inner")
            W["w"] = 0.5*(W["w_pa"] + W["w_pp"])
            W = W[["y","pt","w"]]

        if pt_min is not None: rg = rg[rg["pt"] >= float(pt_min)]
        if pt_max is not None: rg = rg[rg["pt"] <= float(pt_max)]
        if (pt_floor is not None): rg = rg[rg["pt"] >= float(pt_floor)]
        g0 = rg.merge(W, on=["y","pt"], how="left")

        # apply tiny-weight floor if asked
        if (w_floor_frac is not None) and (w_floor_frac > 0.0) and not g0.empty:
            w = g0["w"].to_numpy(float)
            pos = w[w > 0.0]
            if pos.size:
                wmin = float(np.nanpercentile(pos, 5)) * float(w_floor_frac)
                g0["w"] = np.where(w > 0.0, np.maximum(w, wmin), 0.0)

        alphas = self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight)
        out = []
        for label, alpha in alphas:
            if not np.isfinite(alpha):
                if verbose: print(f"  [warn] alpha NaN for bin {label}; skipping.")
                out.append((label, pd.DataFrame())); continue

            g = self._scale_grid_by_alpha(g0.copy(), float(alpha))
            if g.empty:
                out.append((label, pd.DataFrame(columns=["y","y_left","y_right","r_central","r_lo","r_hi"]))); continue

            y = g["y"].to_numpy(float)
            vmin, vmax = float(np.min(y)), float(np.max(y))
            y_edges = np.arange(math.floor(vmin/y_width)*y_width, math.ceil(vmax/y_width)*y_width + 1e-12, y_width)
            ids = np.digitize(y, y_edges) - 1; g["ybin"] = ids

            rows = []
            for bidx in sorted(np.unique(ids)):
                sub = g[g["ybin"]==bidx]
                if sub.empty: continue
                w = np.clip(sub["w"].to_numpy(float), 0, None)
                s = float(np.sum(w))
                if s == 0.0: continue
                y_left, y_right = y_edges[bidx], y_edges[bidx+1]
                for col in ("r_central","r_lo","r_hi"):
                    sub[col] = np.asarray(sub[col], float)
                rows.append({
                    "y": 0.5*(y_left+y_right), "y_left": float(y_left), "y_right": float(y_right),
                    "r_central": _wavg_safe(sub, "r_central"),
                    "r_lo":      _wavg_safe(sub, "r_lo"),
                    "r_hi":      _wavg_safe(sub, "r_hi"),
                })
            out.append((label, pd.DataFrame(rows)))
            if verbose: print(f"  {label}: y-bins={len(rows)} (Δy={y_width})")
        return out

    def rpa_vs_y_in_centrality_bins_matched(
        self, base_xy, r0, members, df_pa, centrality_edges_pct,
        epps_provider, set_ids=None, y_width: float = 0.5,
        pt_min: float = 0.0, pt_max: float = 20.0,
        sigmaNN_mb: float = 71.0, weight: str = "inelastic", verbose: bool = False
    ):
        base = base_xy.copy()
        msel = (base["pt"] >= pt_min) & (base["pt"] <= pt_max)
        if not np.any(msel):
            return [(f"{centrality_edges_pct[i]}-{centrality_edges_pct[i+1]}%", pd.DataFrame())
                    for i in range(len(centrality_edges_pct)-1)]

        y  = base.loc[msel, "y"].to_numpy(float)
        pt = base.loc[msel, "pt"].to_numpy(float)

        wtab = df_pa.merge(base[msel], on=["y","pt"], how="inner")[["y","pt","val"]]
        w = wtab["val"].to_numpy(float)
        if not np.any(w>0):
            return [(f"{centrality_edges_pct[i]}-{centrality_edges_pct[i+1]}%", pd.DataFrame())
                    for i in range(len(centrality_edges_pct)-1)]

        vmin, vmax  = float(np.min(y)), float(np.max(y))
        y_edges     = np.arange(np.floor(vmin/y_width)*y_width, np.ceil(vmax/y_width)*y_width + 1e-12, y_width)
        ids         = np.digitize(y, y_edges) - 1
        uniq_bins   = sorted(np.unique(ids))
        ## Central set K on the same (y,pt)
        SA_c  = epps_provider.SA_ypt_set(y, pt, set_id=1)
        use_centralK = bool(self._matched.get("centralK", False))
        Kmem = members.shape[0]
        set_ids = set_ids or list(range(2, Kmem+2))
        if len(set_ids) != Kmem:
            raise ValueError("[matched] set_ids length must equal #member rows")
        pairs = self._get_hessian_pairs(epps_provider, set_ids)
        r0_sel = r0[msel.to_numpy()]

        out = []
        for label, alpha in self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight):
            if not np.isfinite(alpha):
                out.append((label, pd.DataFrame())); continue
            alpha = float(alpha)

            Kj_c  = (1.0 + self.Nnorm * (SA_c - 1.0) * alpha) / np.clip(SA_c, 1e-12, None)
            rows = []

            for b in uniq_bins:
                sel = (ids == b)
                if not np.any(sel): continue
                w_sel = w[sel]
                if not np.any(w_sel>0): continue

                r_c = (np.sum((r0_sel[sel] * Kj_c[sel]) * w_sel) / np.sum(w_sel))

                # Members → dict by set_id
                vals_by_id = {}
                for j, sid in enumerate(set_ids):
                    if use_centralK:
                        Kj = Kj_c[sel]
                    else:
                        SAj = epps_provider.SA_ypt_set(y[sel], pt[sel], set_id=int(sid))
                        Kj  = (1.0 + self.Nnorm * (SAj - 1.0) * alpha) / np.clip(SAj, 1e-12, None)
                    Rj  = members[j, msel.to_numpy()][sel] * Kj
                    vals_by_id[int(sid)] = (np.sum(Rj * w_sel) / np.sum(w_sel)) if np.sum(w_sel) > 0 else np.nan

                lo, hi = self._hessian_from_pairs(vals_by_id, pairs, r_c)
                y_left, y_right = y_edges[b], y_edges[b+1]
                rows.append({"y":0.5*(y_left+y_right), "y_left":float(y_left), "y_right":float(y_right),
                             "r_central": r_c, "r_lo": lo, "r_hi": hi})

            out.append((label, pd.DataFrame(rows)))
            if verbose: print(f"  {label}: y-bins={len(rows)} (Δy={y_width})")
        return out


    def rpa_vs_pt_in_centrality_bins(self, rgrid, df_pa, centrality_edges_pct,
                                     y_min, y_max, pt_width: float = 2.5, sigmaNN_mb: float = 71.0,
                                     weight: str = "inelastic", verbose: bool = True, pt_floor: float = 2.0,
                                     *, weight_mode: Literal["pa","pp","blend"]="pa",
                                     w_floor_frac: Optional[float]=0.05,
                                     pp_central: Optional[pd.DataFrame]=None) -> List[Tuple[str, pd.DataFrame]]:
        if self._matched is not None:
            m = self._matched
            return self.rpa_vs_pt_in_centrality_bins_matched(
                m['base_xy'], m['r0'], m['members'], df_pa, centrality_edges_pct,
                m['epps'], m.get('set_ids'),
                y_min=y_min, y_max=y_max, pt_width=pt_width,
                sigmaNN_mb=sigmaNN_mb, weight=weight, verbose=verbose
            )
        rg = self.attach_SA_to_grid(rgrid, verbose=False)
        df_pa_w = df_pa[["y","pt","val"]].rename(columns={"val":"w"})
        rg = rg[(rg["y"]>=y_min)&(rg["y"]<=y_max)].copy()
        rg = rg.merge(df_pa_w, on=["y","pt"], how="left")

        # Optional robust weighting
        if weight_mode == "pa":
            W = self._W_BUILDER(rgrid, df_pa, mode="pa")
        elif weight_mode == "pp":
            W = self._W_BUILDER(rgrid, df_pa, df_pp=pp_central, mode="pp")
        else:
            W_pa = self._W_BUILDER(rgrid, df_pa, mode="pa").rename(columns={"w":"w_pa"})
            W_pp = self._W_BUILDER(rgrid, df_pa, df_pp=pp_central, mode="pp").rename(columns={"w":"w_pp"})
            W = W_pa.merge(W_pp, on=["y","pt"], how="inner"); W["w"] = 0.5*(W["w_pa"]+W["w_pp"]); W = W[["y","pt","w"]]
        rg = rg.drop(columns=["w"], errors="ignore").merge(W, on=["y","pt"], how="left")

        if (w_floor_frac is not None) and (w_floor_frac > 0.0):
            w = rg["w"].to_numpy(float); pos = w[w>0]
            if pos.size:
                wmin = float(np.nanpercentile(pos, 5)) * float(w_floor_frac)
                rg["w"] = np.where(w > 0.0, np.maximum(w, wmin), 0.0)

        if rg.empty:
            return [(f"{centrality_edges_pct[i]}-{centrality_edges_pct[i+1]}%", pd.DataFrame()) for i in range(len(centrality_edges_pct)-1)]
        alphas = self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight)
        out = []
        pt = rg["pt"].to_numpy()
        pmin, pmax = float(np.min(pt)), float(np.max(pt))
        pt_edges = np.arange(math.floor(pmin/pt_width)*pt_width, math.ceil(pmax/pt_width)*pt_width + 1e-12, pt_width)

        # pmax = float(np.max(pt))
        # pt_edges = np.arange(0.0, math.ceil(pmax/pt_width)*pt_width + 1e-12, pt_width)
            
        ids = np.digitize(pt, pt_edges) - 1
        rg["ptbin"] = ids
        for label, alpha in alphas:
            if not np.isfinite(alpha):
                if verbose: print(f"  [warn] alpha NaN for bin {label}; skipping.")
                out.append((label, pd.DataFrame())); continue
            g = self._scale_grid_by_alpha(rg, alpha)
            def _wavg(df, col):
                w = np.clip(df["w"].to_numpy(), 0, None); v = df[col].to_numpy(); s = w.sum()
                return float((v*w).sum()/s) if s>0 else np.nan
            rows = []
            for bidx in sorted(np.unique(ids)):
                sub = g[g["ptbin"]==bidx]
                if sub.empty: continue
                pt_left, pt_right = pt_edges[bidx], pt_edges[bidx+1]
                if pt_right <= float(pt_floor): continue
                rows.append({"pt":0.5*(pt_left+pt_right),"pt_left":pt_left,"pt_right":pt_right,
                             "r_central":_wavg(sub,"r_central"),"r_lo":_wavg(sub,"r_lo"),"r_hi":_wavg(sub,"r_hi")})
            out.append((label, pd.DataFrame(rows)))
            if verbose: print(f"  {label}: pT-bins={len(rows)} in y∈[{y_min},{y_max}] (ΔpT={pt_width})")
        return out

    def rpa_vs_pt_in_centrality_bins_matched(
        self, base_xy, r0, members, df_pa, centrality_edges_pct,
        epps_provider, set_ids=None, y_min: float = -5.0, y_max: float = 5.0,
        pt_width: float = 2.5, sigmaNN_mb: float = 71.0,
        weight: str = "inelastic", verbose: bool = False,
        pt_floor: float = 2.0):
        base = base_xy.copy()
        msel = (base["y"] >= y_min) & (base["y"] <= y_max)
        if not np.any(msel):
            return [(f"{centrality_edges_pct[i]}-{centrality_edges_pct[i+1]}%", pd.DataFrame())
                    for i in range(len(centrality_edges_pct)-1)]

        y  = base.loc[msel, "y"].to_numpy(float)
        pt = base.loc[msel, "pt"].to_numpy(float)

        wtab = df_pa.merge(base[msel], on=["y","pt"], how="inner")[["y","pt","val"]]
        w = wtab["val"].to_numpy(float)
        if not np.any(w>0):
            return [(f"{centrality_edges_pct[i]}-{centrality_edges_pct[i+1]}%", pd.DataFrame())
                    for i in range(len(centrality_edges_pct)-1)]

        pmin, pmax  = float(np.min(pt)), float(np.max(pt))
        pt_edges    = np.arange(np.floor(pmin/pt_width)*pt_width, np.ceil(pmax/pt_width)*pt_width + 1e-12, pt_width)
        ids         = np.digitize(pt, pt_edges) - 1
        uniq_bins   = sorted(np.unique(ids))

        SA_c  = epps_provider.SA_ypt_set(y, pt, set_id=1)
        use_centralK = bool(self._matched.get("centralK", False))
        Kmem = members.shape[0]
        set_ids = set_ids or list(range(2, Kmem+2))
        if len(set_ids) != Kmem:
            raise ValueError("[matched] set_ids length must equal #member rows")
        pairs = self._get_hessian_pairs(epps_provider, set_ids)
        r0_sel = r0[msel.to_numpy()]

        out = []
        for label, alpha in self._alphas_for_bins(centrality_edges_pct, sigmaNN_mb, weight):
            if not np.isfinite(alpha):
                out.append((label, pd.DataFrame())); continue
            alpha = float(alpha)

            Kj_c  = (1.0 + self.Nnorm * (SA_c - 1.0) * alpha) / np.clip(SA_c, 1e-12, None)
            rows = []
            for b in uniq_bins:
                sel = (ids == b)
                if not np.any(sel): continue
                pt_left, pt_right = pt_edges[b], pt_edges[b+1]
                if pt_right <= float(pt_floor):  # optional safeguard for ultra-low pT
                    continue

                w_sel = w[sel]
                if not np.any(w_sel>0): continue

                r_c = (np.sum((r0_sel[sel] * Kj_c[sel]) * w_sel) / np.sum(w_sel))

                vals_by_id = {}
                for j, sid in enumerate(set_ids):
                    if use_centralK:
                        Kj = Kj_c[sel]
                    else:
                        SAj = epps_provider.SA_ypt_set(y[sel], pt[sel], set_id=int(sid))
                        Kj  = (1.0 + self.Nnorm * (SAj - 1.0) * alpha) / np.clip(SAj, 1e-12, None)
                    Rj  = members[j, msel.to_numpy()][sel] * Kj
                    vals_by_id[int(sid)] = (np.sum(Rj * w_sel) / np.sum(w_sel)) if np.sum(w_sel) > 0 else np.nan

                lo, hi = self._hessian_from_pairs(vals_by_id, pairs, r_c)
                rows.append({"pt":0.5*(pt_left+pt_right), "pt_left":float(pt_left), "pt_right":float(pt_right),
                             "r_central": r_c, "r_lo": lo, "r_hi": hi})

            out.append((label, pd.DataFrame(rows)))
            if verbose: print(f"  {label}: pT-bins={len(rows)} in y∈[{y_min},{y_max}] (ΔpT={pt_width})")
        return out

# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------

__all__ = [
    "NPDFSystem", "RpAAnalysis",
    "style_axes", "step_band_xy", "overlay_error_members", "slice_nearest_pt_for_each_y",
    "band_xy", "step_band_from_centers", "step_band_from_left_edges", "centers_to_left_edges",
    "plot_rpa_vs_centrality_hzerr",
    "WoodsSaxonPb", "CentralityModel", "GluonFromGrid", "GluonRatioTable",
    "read_topdrawer", "load_top_file", "discover_by_number",
    "ensure_dir", "ensure_out", "round_grid", "weighted_average", "GridStats",
]
