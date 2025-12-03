# npdf_centrality.py
#
# Centrality-dependent nPDF factor K(b; y, pT) for pA, and binned R_pA with
# Hessian bands.
#
# This module is a cleaned-up, reusable refactor of your notebook logic.
# It handles:
#
# 1. Precomputing S_A(y,pT) for all 49 EPPS sets on the σ grid.
# 2. Building K(b; y,pT) = S_AWS / S_A, averaged over centrality bins.
# 3. Fusing K with σ_pA/σ_pp (central + 48 members) to get R_pA(y,pT)
#    with nPDF centrality dependence, on the common σ grid.
# 4. Binning R_pA (with Hessian bands) vs:
#    - y   (in pT range)
#    - pT  (in y-window)
#    - centrality (in y-window)
#
# The binning helpers are generic: they only require a dict of per-centrality
# DataFrames with columns ['y','pt','r_central','r_mem_***'], plus σ grids
# (df_pp, df_pa) and (optionally) a GluonEPPSProvider for SA@y0 weighting.

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


# ----------------------------------------------------------------------
# 1) S_A cache and K(b; y,pT)
# ----------------------------------------------------------------------

def precompute_SA_all(gluon, base_df: pd.DataFrame, y_shift: float) -> np.ndarray:
    """
    Compute S_A(y,pT) for all 49 EPPS sets once on the (y,pt) grid of base_df.

    Parameters
    ----------
    gluon   : GluonEPPSProvider
    base_df : DataFrame with columns 'y', 'pt'
    y_shift : float, rapidity shift applied when evaluating S_A

    Returns
    -------
    SA_all : np.ndarray of shape (49, N) where N = len(base_df)
    """
    yy = base_df["y"].to_numpy()
    pp = base_df["pt"].to_numpy()
    ys = yy + float(y_shift)

    SA_all = [gluon.SA_ypt_set(ys, pp, set_id=sid) for sid in range(1, 50)]
    return np.stack(SA_all, axis=0)   # (49, N)


def K0_KM_avg_over_bin(gluon,
                       base_df: pd.DataFrame,
                       glauber,
                       c0: float,
                       c1: float,
                       *,
                       nb: int = 5,
                       y_shift: float = 0.0,
                       SA_all: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average K(b; y,pT) = S_AWS(b; y,pT) / S_A(y,pT) over the centrality
    bin [c0,c1] using nb sampling points in inelastic percentile.

    Fast version:
      * Uses precomputed S_A_all (49,N) if provided.
      * Uses analytic S_AWS = 1 + Nnorm * (S_A - 1) * alpha(b).
      * Averages K(b) over the b-grid.

    Parameters
    ----------
    gluon   : GluonEPPSProvider
    base_df : DataFrame with 'y','pt' defining σ grid
    glauber : OpticalGlauber
    c0,c1   : centrality edges in %, e.g. 0,20
    nb      : number of b-sampling points in that bin (in percentile space)
    y_shift : rapidity shift applied to S_A
    SA_all  : optional precomputed S_A(y,pT) array, shape (49, N)

    Returns
    -------
    K0 : np.ndarray, shape (N,)   central K(b;y,pT) averaged over the bin
    KM : np.ndarray, shape (48,N) member K(b;y,pT) averaged over the bin
    """
    yy = base_df["y"].to_numpy()
    pp = base_df["pt"].to_numpy()
    ys = yy + float(y_shift)

    if SA_all is None:
        SA_all_loc = np.stack(
            [gluon.SA_ypt_set(ys, pp, set_id=sid) for sid in range(1, 50)],
            axis=0
        )  # (49, N)
    else:
        SA_all_loc = SA_all

    SA_safe = np.clip(SA_all_loc, 1e-12, None)

    # Normalisation for WS profile
    Nnorm = float(gluon.Nnorm())

    # b-grid in percentile space
    ps    = np.linspace(c0 / 100.0, c1 / 100.0, nb)
    bgrid = np.array([glauber.b_from_percentile(p, kind="pA") for p in ps])

    K0_acc = []
    KM_acc = []

    for b_val in bgrid:
        alpha_b = float(gluon.alpha_of_b(b_val))
        # S_AWS(b; y,pT) for all sets
        SAWS_all = 1.0 + Nnorm * (SA_all_loc - 1.0) * alpha_b   # (49,N)
        # K_all(b; y,pT)
        K_all = SAWS_all / SA_safe                              # (49,N)

        K0_acc.append(K_all[0])        # central set
        KM_acc.append(K_all[1:, :])    # 48 members

    K0 = np.mean(np.stack(K0_acc, axis=0), axis=0)   # (N,)
    KM = np.mean(np.stack(KM_acc, axis=0), axis=0)   # (48,N)

    return K0, KM


# ----------------------------------------------------------------------
# 2) Fuse σ_pA/σ_pp with a multiplicative factor (here K)
# ----------------------------------------------------------------------

def fuse_sigma_and_K(base_df: pd.DataFrame,
                     r0: np.ndarray,
                     M: np.ndarray,
                     K0: np.ndarray,
                     KM: np.ndarray) -> pd.DataFrame:
    """
    Fuse σ_pA/σ_pp (central + 48 members) with K(b; y,pT) (central + 48 members)
    to obtain R_pA + Hessian band and member columns on the same (y,pt) grid.

    Parameters
    ----------
    base_df : DataFrame with 'y','pt' defining the grid (length N)
    r0      : np.ndarray, shape (N,)      central σ_pA/σ_pp
    M       : np.ndarray, shape (48,N)    member σ_pA/σ_pp (Hessian pairs)
    K0      : np.ndarray, shape (N,)      central factor
    KM      : np.ndarray, shape (48,N)    member factors

    Returns
    -------
    out : DataFrame with columns
        'y','pt','r_central','r_lo','r_hi','r_mem_001'..'r_mem_048'
    """
    r0 = np.asarray(r0, float)
    M  = np.asarray(M,  float)
    K0 = np.asarray(K0, float)
    KM = np.asarray(KM, float)

    if r0.shape != K0.shape:
        raise ValueError(f"r0.shape={r0.shape} and K0.shape={K0.shape} mismatch")
    if M.shape != KM.shape:
        raise ValueError(f"M.shape={M.shape} and KM.shape={KM.shape} mismatch")
    if M.shape[0] % 2 != 0:
        raise ValueError("M (and KM) must have even number of members (Hessian pairs)")

    r0b = r0 * K0          # central: (N,)
    Mb  = M  * KM          # members: (48,N)

    D = Mb[0::2, :] - Mb[1::2, :]
    h = 0.5 * np.sqrt(np.sum(D * D, axis=0))

    out = base_df[["y", "pt"]].copy()
    out["r_central"] = r0b
    out["r_lo"]      = r0b - h
    out["r_hi"]      = r0b + h

    for j in range(Mb.shape[0]):
        out[f"r_mem_{j+1:03d}"] = Mb[j]

    return out


# ----------------------------------------------------------------------
# 3) High-level driver: one energy → per-centrality tables
# ----------------------------------------------------------------------

def compute_df49_by_centrality(base_df: pd.DataFrame,
                               r0: np.ndarray,
                               M: np.ndarray,
                               gluon,
                               glauber,
                               cent_bins: List[Tuple[float, float]],
                               *,
                               nb_bsamples: int = 5,
                               y_shift_fraction: float = 0.0,
                               y_shift: Optional[float] = None,
                               SA_all: Optional[np.ndarray] = None
                               ) -> Tuple[Dict[str, pd.DataFrame],
                                          Dict[str, Tuple[np.ndarray, np.ndarray]],
                                          np.ndarray,
                                          float]:
    """
    High-level driver for a single energy.

    Parameters
    ----------
    base_df    : DataFrame with 'y','pt' (grid for σ_pA/σ_pp)
    r0         : np.ndarray, shape (N,)    central σ_pA/σ_pp on that grid
    M          : np.ndarray, shape (48,N)  member σ_pA/σ_pp
    gluon      : GluonEPPSProvider
    glauber    : OpticalGlauber
    cent_bins  : list of (c0,c1) in %, e.g. [(0,20),(20,40),...]
    nb_bsamples : int, nb percentile samples per centrality bin in b
    y_shift_fraction : float, shift in units of Δy (0 → no shift)
    y_shift    : optional absolute rapidity shift; if not None,
                 overrides y_shift_fraction.
    SA_all     : optional precomputed S_A(y,pT) cache (49,N).
                 If None, it will be computed here.

    Returns
    -------
    df49_by_cent : dict[tag] -> DataFrame
                   tag is like "0-20%", "20-40%", ...
                   DataFrame has columns:
                     y, pt, r_central, r_lo, r_hi, r_mem_001..r_mem_048
    K_by_cent    : dict[tag] -> (K0, KM)
                   K0: (N,), KM: (48,N)
    SA_all       : np.ndarray, shape (49,N) actually used
    y_shift_used : float, actual rapidity shift applied
    """
    ys_unique = np.sort(np.unique(base_df["y"].to_numpy()))
    if ys_unique.size < 1:
        raise RuntimeError("base_df must contain at least one y value")

    if y_shift is None:
        if ys_unique.size >= 2:
            dy_sigma = float(np.diff(ys_unique).min())
        else:
            dy_sigma = 0.0
        y_shift_used = float(y_shift_fraction) * dy_sigma
    else:
        y_shift_used = float(y_shift)

    if SA_all is None:
        SA_all = precompute_SA_all(gluon, base_df, y_shift_used)

    df49_by_cent: Dict[str, pd.DataFrame] = {}
    K_by_cent: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for (c0, c1) in cent_bins:
        tag = f"{int(c0)}-{int(c1)}%"
        K0, KM = K0_KM_avg_over_bin(
            gluon,
            base_df,
            glauber,
            c0,
            c1,
            nb=nb_bsamples,
            y_shift=y_shift_used,
            SA_all=SA_all,
        )
        K_by_cent[tag]    = (K0, KM)
        df49_by_cent[tag] = fuse_sigma_and_K(base_df, r0, M, K0, KM)

    return df49_by_cent, K_by_cent, SA_all, y_shift_used


# ----------------------------------------------------------------------
# 4) Centrality weights for MB (w(c) scheme used in your notebook)
# ----------------------------------------------------------------------

def make_centrality_weight_dict(cent_bins: List[Tuple[float, float]],
                                c0: float = 0.25
                                ) -> Dict[str, float]:
    """
    Return dict[tag] -> W_bin for each centrality bin, with tags "0-20%",...

    Uses w(c) ∝ exp(-c/c0) on c∈[0,1], integrated over each bin.

    Parameters
    ----------
    cent_bins : list of (c0,c1) in %, e.g. [(0,20),(20,40),...]
    c0        : float, exponential scale in w(c)

    Returns
    -------
    WCENT : dict mapping tag -> weight (normalized to sum to 1)
    """
    edges_frac = np.array(
        [cent_bins[0][0]] + [b for (_, b) in cent_bins],
        float
    ) / 100.0  # e.g. [0.0,0.2,0.4,0.6,0.8,1.0]

    num   = np.exp(-edges_frac[:-1] / c0) - np.exp(-edges_frac[1:] / c0)
    denom = 1.0 - np.exp(-1.0 / c0)
    w = num / max(denom, 1e-30)

    w = np.clip(w, 0.0, None)
    s = np.sum(w)
    if s > 0.0:
        w /= s
    else:
        w = np.ones_like(w) / len(w)

    tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]
    return {tag: float(wi) for tag, wi in zip(tags, w)}


# ----------------------------------------------------------------------
# 5) σ-based weights for binning (generalized get_weights)
# ----------------------------------------------------------------------

def get_weights(sub: pd.DataFrame,
                df_pp: pd.DataFrame,
                df_pa: pd.DataFrame,
                gluon=None,
                *,
                mode: str = "pp@local",
                y_ref: float = 0.0,
                pt_floor_w: Optional[float] = 1.0) -> np.ndarray:
    """
    Returns normalized weights w_n for the given sub-DF.

    Parameters
    ----------
    sub   : DataFrame slice in a (y, pT) bin; must contain columns 'y', 'pt',
            'r_central', and optional 'wcent'.
    df_pp : σ_pp grid with columns 'y','pt','val'
    df_pa : σ_pA grid with columns 'y','pt','val'
    gluon : optional GluonEPPSProvider (needed for SA@y0)
    mode  : "flat", "pa@y0", "pa@local", "pp@local", or "SA@y0".
    y_ref : reference y for *@y0 modes.
    pt_floor_w : if not None and mode uses σ-weights, for pT < pt_floor_w the
                 weight is replaced by the weight at the first pT ≥ pt_floor_w
                 at the same y.

    Returns
    -------
    wn : np.ndarray, shape (len(sub),), normalized weights.
    """
    if len(sub) == 0:
        return np.zeros(0, float)

    # --- base weights ---
    if mode == "flat":
        w = np.ones(len(sub), float)

    # σ_pA(y_ref, pt) as function of pt only
    elif mode == "pa@y0":
        lut = {}
        for ptv, grp in df_pa.groupby("pt"):
            ys   = grp["y"].to_numpy()
            vals = grp["val"].to_numpy()
            j    = int(np.argmin(np.abs(ys - y_ref)))
            lut[float(ptv)] = float(vals[j])
        w = sub["pt"].map(lambda v: lut.get(float(v), 0.0)).to_numpy()

    # σ_pA(y, pt) at each local point
    elif mode == "pa@local":
        lut = {(float(r.y), float(r.pt)): float(r.val) for r in df_pa.itertuples()}
        w = np.array([lut.get((float(r.y), float(r.pt)), 0.0)
                      for r in sub.itertuples()], float)

    # σ_pp(y, pt) at each local point
    elif mode == "pp@local":
        lut = {(float(r.y), float(r.pt)): float(r.val) for r in df_pp.itertuples()}
        w = np.array([lut.get((float(r.y), float(r.pt)), 0.0)
                      for r in sub.itertuples()], float)

    # nPDF-only weight S_A(y_ref, pt)
    elif mode == "SA@y0":
        if gluon is None:
            raise ValueError("gluon provider required for mode='SA@y0'")
        puniq = np.sort(sub["pt"].unique())
        SAref = gluon.SA_ypt_set(np.full_like(puniq, y_ref), puniq, set_id=1)
        lut   = {float(p): float(v) for p, v in zip(puniq, SAref)}
        w = sub["pt"].map(lambda v: lut.get(float(v), 0.0)).to_numpy()

    else:
        raise ValueError(f"Unknown weight mode '{mode}'")

    # --- include centrality fraction if present (MB stacks) ---
    if "wcent" in sub.columns:
        w = w * sub["wcent"].to_numpy()

    # --- low-pT truncation (not for flat/SA) on the *weights* ---
    if (pt_floor_w is not None) and (mode not in ("flat", "SA@y0")) and len(sub):
        pt_arr = sub["pt"].to_numpy()
        mask_bad = (pt_arr < float(pt_floor_w))
        if np.any(mask_bad):
            src = df_pa if mode.startswith("pa@") else df_pp
            y_vals = sub.loc[mask_bad, "y"].to_numpy()
            for yy in np.unique(y_vals):
                cand = src[(src["y"] == float(yy)) &
                           (src["pt"] >= float(pt_floor_w))].sort_values("pt")
                if len(cand):
                    w[(sub["y"].to_numpy() == yy) & mask_bad] = float(cand["val"].iloc[0])

    # --- normalize ---
    wsum = np.sum(w)
    if wsum > 0.0:
        wn = w / wsum
    else:
        wn = np.ones_like(w) / max(len(w), 1)

    return wn


# ----------------------------------------------------------------------
# 6) Binning helpers: vs y, vs pT, vs centrality
# ----------------------------------------------------------------------

def bin_rpa_vs_y(df49_by_cent: Dict[str, pd.DataFrame],
                 df_pp: pd.DataFrame,
                 df_pa: pd.DataFrame,
                 gluon,
                 *,
                 cent_bins: List[Tuple[float, float]],
                 y_edges: np.ndarray,
                 pt_range_avg: Tuple[float, float],
                 weight_mode: str = "pp@local",
                 y_ref: float = 0.0,
                 pt_floor_w: Optional[float] = 1.0,
                 wcent_dict: Optional[Dict[str, float]] = None,
                 include_mb: bool = True
                 ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Bin R_pA vs y for each centrality bin and (optionally) for MB.

    NOTE: to make R_pA(y) consistent with your good R_pA(pT) panels,
    points with pT < pt_floor_w are dropped inside each y-bin before
    averaging, while the pT integration range is still given by
    pt_range_avg = (pt_min, pt_max).

    Parameters
    ----------
    df49_by_cent : dict[tag] -> DataFrame with 'y','pt','r_central','r_mem_***'
    df_pp, df_pa : σ grids for pp and pA (with 'y','pt','val')
    gluon        : GluonEPPSProvider (for SA@y0 if needed)
    cent_bins    : list of (c0,c1) in % (defines the tags)
    y_edges      : np.ndarray of y-bin edges (length Ny+1)
    pt_range_avg : (pt_min, pt_max) used for the R(y) averaging
    weight_mode  : same choices as get_weights
    y_ref        : reference y for *@y0 modes
    pt_floor_w   : low-pT truncation for the averaging
    wcent_dict   : dict[tag] -> weight for MB stacking (w(c) scheme)
    include_mb   : whether to build MB result

    Returns
    -------
    result : dict[tag] -> dict with keys
             'y_left', 'y_right', 'r_central', 'r_lo', 'r_hi'
             tag includes centrality tags and, if include_mb, "MB".
    """
    tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]
    result: Dict[str, Dict[str, np.ndarray]] = {}

    pt0, pt1 = pt_range_avg

    # --- centrality bins ---
    for tag in tags:
        df49 = df49_by_cent[tag].copy()
        df49["wcent"] = 1.0  # trivial

        y_left, y_right = [], []
        r0_list, rlo_list, rhi_list = [], [], []

        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            sub = df49[(df49["y"] >= yl) & (df49["y"] < yr) &
                       (df49["pt"] >= pt0) & (df49["pt"] <= pt1)].copy()

            # NEW: drop pT below pt_floor_w, like in bin_rpa_vs_pT
            if pt_floor_w is not None:
                sub = sub[sub["pt"] >= float(pt_floor_w)]
            # END NEW

            y_left.append(yl)
            y_right.append(yr)

            if sub.empty:
                r0_list.append(np.nan)
                rlo_list.append(np.nan)
                rhi_list.append(np.nan)
                continue

            wn = get_weights(sub, df_pp, df_pa, gluon,
                             mode=weight_mode, y_ref=y_ref,
                             pt_floor_w=pt_floor_w)

            r0  = float(np.sum(wn * sub["r_central"].to_numpy()))
            mem = np.array(
                [np.sum(wn * sub[c].to_numpy())
                 for c in sub.columns if c.startswith("r_mem_")],
                float
            )
            D   = mem[0::2] - mem[1::2]
            h   = 0.5 * np.sqrt(np.sum(D * D))

            r0_list.append(r0)
            rlo_list.append(r0 - h)
            rhi_list.append(r0 + h)

        result[tag] = dict(
            y_left=np.array(y_left, float),
            y_right=np.array(y_right, float),
            r_central=np.array(r0_list, float),
            r_lo=np.array(rlo_list, float),
            r_hi=np.array(rhi_list, float),
        )

    # --- MB stack ---
    if include_mb:
        if wcent_dict is None:
            raise ValueError("wcent_dict required when include_mb=True")

        rows = []
        for (a, b) in cent_bins:
            ctag = f"{int(a)}-{int(b)}%"
            tmp = df49_by_cent[ctag].copy()
            tmp["wcent"] = float(wcent_dict[ctag])
            rows.append(tmp)
        df49_MB = pd.concat(rows, ignore_index=True)

        y_left, y_right = [], []
        r0_list, rlo_list, rhi_list = [], [], []

        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            sub = df49_MB[(df49_MB["y"] >= yl) & (df49_MB["y"] < yr) &
                          (df49_MB["pt"] >= pt0) & (df49_MB["pt"] <= pt1)].copy()

            # same low-pT drop as centrality bins
            if pt_floor_w is not None:
                sub = sub[sub["pt"] >= float(pt_floor_w)]

            y_left.append(yl)
            y_right.append(yr)

            if sub.empty:
                r0_list.append(np.nan)
                rlo_list.append(np.nan)
                rhi_list.append(np.nan)
                continue

            wn = get_weights(sub, df_pp, df_pa, gluon,
                             mode=weight_mode, y_ref=y_ref,
                             pt_floor_w=pt_floor_w)

            r0  = float(np.sum(wn * sub["r_central"].to_numpy()))
            mem = np.array(
                [np.sum(wn * sub[c].to_numpy())
                 for c in sub.columns if c.startswith("r_mem_")],
                float
            )
            D   = mem[0::2] - mem[1::2]
            h   = 0.5 * np.sqrt(np.sum(D * D))

            r0_list.append(r0)
            rlo_list.append(r0 - h)
            rhi_list.append(r0 + h)

        result["MB"] = dict(
            y_left=np.array(y_left, float),
            y_right=np.array(y_right, float),
            r_central=np.array(r0_list, float),
            r_lo=np.array(rlo_list, float),
            r_hi=np.array(rhi_list, float),
        )

    return result


def bin_rpa_vs_pT(df49_by_cent: Dict[str, pd.DataFrame],
                  df_pp: pd.DataFrame,
                  df_pa: pd.DataFrame,
                  gluon,
                  *,
                  cent_bins: List[Tuple[float, float]],
                  pt_edges: np.ndarray,
                  y_window: Tuple[float, float],
                  weight_mode: str = "pp@local",
                  y_ref: float = 0.0,
                  pt_floor_w: Optional[float] = 1.0,
                  wcent_dict: Optional[Dict[str, float]] = None,
                  include_mb: bool = True
                  ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Bin R_pA vs pT for each centrality bin and (optionally) for MB,
    in a given y-window.

    This matches your notebook:
      - First bin: [pt_edges[0], pt_edges[1]] inclusive on the left.
      - Subsequent bins: (pl, pr] so points on the edge go to the upper bin.
      - pT < pt_floor_w are dropped for the averaging but the bin on the axis
        remains [pl,pr]; this is how you fix the 0–2.5 bin behaviour.

    Parameters
    ----------
    df49_by_cent : dict[tag] -> DataFrame with 'y','pt','r_central','r_mem_***'
    df_pp, df_pa : σ grids for pp and pA
    gluon        : GluonEPPSProvider (for SA@y0 if needed)
    cent_bins    : list of (c0,c1) in % (defines the tags)
    pt_edges     : np.ndarray of pT-bin edges (length Np+1)
    y_window     : (y0, y1) inclusive
    weight_mode  : same choices as get_weights
    y_ref        : reference y for *@y0 modes
    pt_floor_w   : low-pT truncation scale for averaging
    wcent_dict   : dict[tag] -> weight for MB stacking
    include_mb   : whether to build MB result

    Returns
    -------
    result : dict[tag] -> dict with keys
             'pt_left', 'pt_right', 'r_central', 'r_lo', 'r_hi'
    """
    y0, y1 = y_window
    tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]
    result: Dict[str, Dict[str, np.ndarray]] = {}

    def _bin_one_df(df49: pd.DataFrame) -> Dict[str, np.ndarray]:
        p_left, p_right = [], []
        r0_list, rlo_list, rhi_list = [], [], []

        for i, (pl, pr) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
            # First bin: [pl, pr]; subsequent: (pl, pr]
            if i == 0:
                mask_pt = (df49["pt"] >= pl) & (df49["pt"] <= pr)
            else:
                mask_pt = (df49["pt"] > pl) & (df49["pt"] <= pr)

            sub = df49[mask_pt &
                       (df49["y"] >= y0) & (df49["y"] <= y1)].copy()

            # drop points below pt_floor_w, but KEEP the 0–2.5 bin on the axis
            if pt_floor_w is not None:
                sub = sub[sub["pt"] >= float(pt_floor_w)]

            p_left.append(pl)
            p_right.append(pr)

            if sub.empty:
                r0_list.append(np.nan)
                rlo_list.append(np.nan)
                rhi_list.append(np.nan)
                continue

            wn = get_weights(sub, df_pp, df_pa, gluon,
                             mode=weight_mode, y_ref=y_ref,
                             pt_floor_w=pt_floor_w)

            r0  = float(np.sum(wn * sub["r_central"].to_numpy()))
            mem = np.array(
                [np.sum(wn * sub[c].to_numpy())
                 for c in sub.columns if c.startswith("r_mem_")],
                float
            )
            D   = mem[0::2] - mem[1::2]
            h   = 0.5 * np.sqrt(np.sum(D * D))

            r0_list.append(r0)
            rlo_list.append(r0 - h)
            rhi_list.append(r0 + h)

        return dict(
            pt_left=np.array(p_left, float),
            pt_right=np.array(p_right, float),
            r_central=np.array(r0_list, float),
            r_lo=np.array(rlo_list, float),
            r_hi=np.array(rhi_list, float),
        )

    # centrality bins
    for tag in tags:
        df49 = df49_by_cent[tag].copy()
        df49["wcent"] = 1.0
        result[tag] = _bin_one_df(df49)

    # MB
    if include_mb:
        if wcent_dict is None:
            raise ValueError("wcent_dict required when include_mb=True")

        rows = []
        for (a, b) in cent_bins:
            ctag = f"{int(a)}-{int(b)}%"
            tmp = df49_by_cent[ctag].copy()
            tmp["wcent"] = float(wcent_dict[ctag])
            rows.append(tmp)
        df49_MB = pd.concat(rows, ignore_index=True)

        result["MB"] = _bin_one_df(df49_MB)

    return result


def bin_rpa_vs_centrality(df49_by_cent: Dict[str, pd.DataFrame],
                          df_pp: pd.DataFrame,
                          df_pa: pd.DataFrame,
                          gluon,
                          *,
                          cent_bins: List[Tuple[float, float]],
                          y_window: Tuple[float, float],
                          pt_range_avg: Tuple[float, float],
                          weight_mode: str = "pp@local",
                          y_ref: float = 0.0,
                          pt_floor_w: Optional[float] = 1.0,
                          width_weights: Optional[np.ndarray] = None
                          ) -> Dict[str, np.ndarray]:
    """
    Bin R_pA vs centrality for a given y-window and pT-averaging range.

    This reproduces your notebook logic + the same low-pT handling as
    bin_rpa_vs_pT and bin_rpa_vs_y:
      - For each centrality bin, integrate over y-window & pT-range with
        σ-based weights and Hessian members, after dropping pT < pt_floor_w.
      - MB line is computed by combining per-bin member means with bin-width
        weights wcent ∝ (Δc), normalized.

    Parameters
    ----------
    df49_by_cent : dict[tag] -> DataFrame with 'y','pt','r_central','r_mem_***'
    df_pp, df_pa : σ grids
    gluon        : GluonEPPSProvider
    cent_bins    : list of (c0,c1) in %
    y_window     : (y0,y1) inclusive
    pt_range_avg : (pt_min, pt_max) for the averaging
    weight_mode  : same choices as get_weights
    y_ref        : reference y for *@y0 modes
    pt_floor_w   : low-pT truncation for the averaging
    width_weights: optional weights per bin for MB; if None, use
                   w_i ∝ (c1 - c0) as in your notebook.

    Returns
    -------
    result : dict with keys
             'cent_left', 'cent_right', 'r_central', 'r_lo', 'r_hi',
             'mb_r_central', 'mb_r_lo', 'mb_r_hi'
    """
    y0, y1 = y_window
    pt0, pt1 = pt_range_avg

    cent_left = [a for (a, _) in cent_bins]
    cent_right = [b for (_, b) in cent_bins]
    tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    Rc_list, Rlo_list, Rhi_list = [], [], []
    mem_means_by_bin = []

    for tag, (c0, c1) in zip(tags, cent_bins):
        g = df49_by_cent[tag]
        sub = g[(g["y"] >= y0) & (g["y"] <= y1) &
                (g["pt"] >= pt0) & (g["pt"] <= pt1)].copy()

        # NEW: drop pT below pt_floor_w before averaging
        if pt_floor_w is not None:
            sub = sub[sub["pt"] >= float(pt_floor_w)]
        # END NEW

        if sub.empty:
            Rc_list.append(np.nan)
            Rlo_list.append(np.nan)
            Rhi_list.append(np.nan)
            mem_means_by_bin.append(np.full(48, np.nan))
            continue

        wn = get_weights(sub, df_pp, df_pa, gluon,
                         mode=weight_mode, y_ref=y_ref,
                         pt_floor_w=pt_floor_w)

        r0  = float(np.sum(wn * sub["r_central"].to_numpy()))
        mem = np.array(
            [np.sum(wn * sub[c].to_numpy())
             for c in sub.columns if c.startswith("r_mem_")],
            float
        )
        D   = mem[0::2] - mem[1::2]
        h   = 0.5 * np.sqrt(np.sum(D * D))

        Rc_list.append(r0)
        Rlo_list.append(r0 - h)
        Rhi_list.append(r0 + h)
        mem_means_by_bin.append(mem)

    Rc_arr  = np.array(Rc_list, float)
    Rlo_arr = np.array(Rlo_list, float)
    Rhi_arr = np.array(Rhi_list, float)

    # --- MB combination ---
    mem_means_by_bin = np.array(mem_means_by_bin, float)  # (nbin,48)

    if width_weights is None:
        wcent = np.array([(b - a) / 100.0 for (a, b) in cent_bins], float)
        s = np.sum(wcent)
        if s > 0.0:
            wcent /= s
        else:
            wcent = np.ones_like(wcent) / len(wcent)
    else:
        wcent = np.asarray(width_weights, float)
        s = np.sum(wcent)
        if s > 0.0:
            wcent /= s

    with np.errstate(invalid="ignore"):
        mem_mb = np.nansum(wcent[:, None] * mem_means_by_bin, axis=0)

    Dmb = mem_mb[0::2] - mem_mb[1::2]
    hmb = 0.5 * np.sqrt(np.sum(Dmb * Dmb))

    rmb = float(np.nansum(wcent * Rc_arr))
    mb_r_central = rmb
    mb_r_lo      = rmb - hmb
    mb_r_hi      = rmb + hmb

    return dict(
        cent_left=np.array(cent_left, float),
        cent_right=np.array(cent_right, float),
        r_central=Rc_arr,
        r_lo=Rlo_arr,
        r_hi=Rhi_arr,
        mb_r_central=mb_r_central,
        mb_r_lo=mb_r_lo,
        mb_r_hi=mb_r_hi,
    )
