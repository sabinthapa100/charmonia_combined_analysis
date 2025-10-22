"""
combine_module.py  — publication-ready plots with minimal changes
===============================================================

Glue layer to (1) load nPDF, eLoss, Primordial; (2) produce CNM and Total
with asymmetric error propagation; (3) draw Nature-grade figures.

Key physics (unchanged):
- CNM = nPDF × eLoss  (state-blind)
- Total = CNM × Primordial(state)  (state-dependent)
- Asymmetric error propagation in relative space, added in quadrature.

Design:
- Minimal edits to your working logic
- One flexible centrality plotter + back-compatible convenience wrappers
- Legend/notes/ticks inside panels by options (no grid by default)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- external modules (APIs unchanged) ----
from eloss_module import ELossRun, load_eloss_run                     # :contentReference[oaicite:2]{index=2}
from npdf_module import (                                            # :contentReference[oaicite:3]{index=3}
    NPDFSystem, RpAAnalysis, WoodsSaxonPb, CentralityModel,
    GluonFromGrid, GluonRatioTable,
    style_axes as _npdf_style_axes,  # not used for grid
    band_xy as _band_xy,
    plot_rpa_vs_centrality_hzerr as _plot_rpa_vs_centrality_hzerr
)
from primordial_module import ReaderConfig, build_ensemble, make_bins_from_width

# ---------------------- constants / palettes (exported) ----------------------
# CENT_EDGES_ELOSS = [0, 20, 40, 60, 100]
CENT_EDGES_ELOSS = [0, 10, 20, 40, 60, 80, 100]
Y_WINDOWS_3      = [(-1.93, 1.93), (2.03, 3.53), (-4.46, -2.96)]
PT_BINS_2P5      = [(i, i+2.5) for i in np.arange(0.0, 20.0, 2.5)]

# Components: unchanged, distinct
COMP_COLORS = {
    "nPDF":       "#1f77b4",  # blue
    "eLoss":      "#ff7f0e",  # orange
    "CNM":        "#2ca02c",  # green
    "Primordial": "#9467bd",  # purple
    "Total":      "#d62728",  # red
}
# States: distinct from components
STATE_COLORS = {
    "jpsi_1S":  "#8c564b",  # brown
    "psi_2S":   "#e377c2",  # pink
    "chicJ_1P": "#17becf",  # cyan
}
# Pretty LaTeX labels
PRETTY_STATE = {
    "jpsi_1S":  r"$J/\psi(1S)$",
    "psi_2S":   r"$\psi(2S)$",
    "chicJ_1P": r"$\chi_c(1P)$",
}
# Band opacities
BAND_ALPHA = {"CNM": 0.22, "Primordial": 0.22, "Total": 0.28}

# --- tolerant centrality label normalization (one canonical form) ---
_CNUM = re.compile(r"([0-9]+(?:\.[0-9]+)?)")

def _norm_cent_label(s: str) -> str:
    """Return a single canonical string like '0-10%' for any of:
       0-10, '0–10', '0 - 10 %', [0,10], pandas.Interval, etc."""
    if s is None:
        return ""
    txt = str(s).strip().replace("–", "-")
    # Interval-like?
    if hasattr(s, "left") and hasattr(s, "right"):
        lo, hi = float(s.left), float(s.right)
    else:
        nums = _CNUM.findall(txt.replace("%", ""))
        if len(nums) < 2:
            return txt
        lo, hi = float(nums[0]), float(nums[1])
    def _fmt(x):
        return str(int(round(x))) if abs(x - round(x)) < 1e-8 else f"{x:g}"
    return f"{_fmt(lo)}-{_fmt(hi)}%"

def _as_y_bins(y_bins):
    """Accept [(yl,yr)], or a 1D array of edges, or a float width; return [(yl,yr), ...]."""
    import numpy as np
    # float/int → treat as width on [-5,5]
    if isinstance(y_bins, (int, float)):
        w = float(y_bins)
        edges = np.arange(-5.0, 5.0 + 1e-12, w)
        return list(zip(edges[:-1], edges[1:]))

    # list/array of numbers → edges
    y_bins = list(y_bins)
    if y_bins and isinstance(y_bins[0], (int, float)):
        edges = np.asarray(y_bins, float)
        return list(zip(edges[:-1], edges[1:]))

    # already [(yl,yr)]
    return y_bins

# --------------------------- math / propagation ---------------------------
def _rel_band(c, lo, hi):
    c  = np.asarray(c,  float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    eps = 1e-14
    rel_p = np.where(np.abs(c) > eps, (hi - c)/c, 0.0)
    rel_m = np.where(np.abs(c) > eps, (c - lo)/c, 0.0)
    return rel_m, rel_p

def combine_product_asym(Ac, Alo, Ahi, Bc, Blo, Bhi):
    """Multiply two quantities with asymmetric bands. Returns central, lo, hi arrays."""
    Ac, Alo, Ahi = map(lambda x: np.asarray(x, float), (Ac, Alo, Ahi))
    Bc, Blo, Bhi = map(lambda x: np.asarray(x, float), (Bc, Blo, Bhi))
    Arelm, Arelp = _rel_band(Ac, Alo, Ahi)
    Brelm, Brelp = _rel_band(Bc, Blo, Bhi)
    Cc = Ac * Bc
    Crelp = np.sqrt(Arelp**2 + Brelp**2)
    Crelm = np.sqrt(Arelm**2 + Brelm**2)
    Chi = Cc * (1.0 + Crelp)
    Clo = Cc * (1.0 - Crelm)
    return Cc, Clo, Chi

def avg_band_by_weights(x_c, x_lo, x_hi, w):
    """Average central and +/- deviations separately (preserves asymmetry)."""
    x_c  = np.asarray(x_c,  float)
    x_lo = np.asarray(x_lo, float)
    x_hi = np.asarray(x_hi, float)
    w    = np.asarray(w,    float)
    s = w.sum()
    if s <= 0: return (np.nan, np.nan, np.nan)
    cbar = float(np.sum(w * x_c) / s)
    dlo  = float(np.sum(w * (x_c - x_lo)) / s)
    dhi  = float(np.sum(w * (x_hi - x_c)) / s)
    return cbar, cbar - dlo, cbar + dhi

def _nearest_b_rows(df: pd.DataFrame, bL: float, bR: float) -> pd.DataFrame:
    """Use rows inside [bL,bR] if any; otherwise the rows at the closest b to the bin center."""
    if df is None or df.empty or "b" not in df.columns:
        return df.iloc[0:0].copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    sub = df[(df["b"] >= bL) & (df["b"] <= bR)]
    if not sub.empty:
        return sub
    bmid  = 0.5*(float(bL) + float(bR))
    bvals = df["b"].to_numpy(dtype=float)
    bstar = float(bvals[np.argmin(np.abs(bvals - bmid))])
    return df[np.isclose(df["b"].astype(float), bstar)]

# --- utility: accept float OR bins list for y_width
def _normalize_y_width(y_width):
    import numpy as np
    # float-like
    try:
        return float(y_width)
    except Exception:
        pass
    # list/array of (yl, yr) bins OR a single pair
    if isinstance(y_width, (list, tuple, np.ndarray)):
        if len(y_width) == 2 and all(isinstance(v, (int, float)) for v in y_width):
            return float(abs(y_width[1] - y_width[0]))
        first = y_width[0]
        if isinstance(first, (list, tuple, np.ndarray)) and len(first) == 2:
            return float(abs(first[1] - first[0]))
    raise TypeError("y_width must be a float or an iterable of (yl,yr) bins")


# ------------------------------- styling --------------------------------
def _pub_style():
    plt.rcParams.update({
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.linewidth": 1.4,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "lines.linewidth": 2.2,
        "legend.fontsize": 13,
        "figure.autolayout": True,
    })

def _style_axes(ax, xlab, ylab, title: Optional[str]=None, minor_ticks: bool=True):
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if minor_ticks:
        ax.minorticks_on()
    if title:
        ax.set_title(title)

def _legend_apply(ax, handles, labels, legend_mode, legend_loc, panel_index, grid_first_panel=0):
    if legend_mode == "each":
        ax.legend(handles, labels, loc=legend_loc, frameon=False)
    elif legend_mode == "panel-first":
        if panel_index == grid_first_panel:
            ax.legend(handles, labels, loc=legend_loc, frameon=False)
    # legend_mode == "figure" handled by caller

def _annotate_corner(ax, text: str, loc: str = "lower right", fontsize: int = 12, alpha: float = 0.9):
    locs = {
        "lower right": (0.98, 0.04), "lower left": (0.02, 0.04),
        "upper right": (0.98, 0.96), "upper left": (0.02, 0.96),
    }
    xy = locs.get(loc, (0.98, 0.04))
    ax.text(xy[0], xy[1], text, transform=ax.transAxes,
            ha="right" if "right" in loc else "left",
            va="bottom" if "lower" in loc else "top",
            fontsize=fontsize, alpha=alpha)

def _parse_centbin(s: str) -> Tuple[float, float]:
    s = str(s).strip().replace("–", "-").replace("%", "")
    L, R = s.split("-")
    return float(L), float(R)

# --- centrality helpers (minimal + robust) ---
def _parse_cent_label(label: str) -> Tuple[float, float]:
    s = str(label).strip().replace("%", "").replace("–", "-")
    L, R = s.split("-")
    return float(L), float(R)

def _overlap_width(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    L = max(a[0], b[0]); R = min(a[1], b[1])
    return max(0.0, R - L)

# ------------------------------- Combiner -------------------------------
@dataclass
class Combiner:
    tag: str
    e_loss_base: str
    npdf_folder: str
    prim_base: str
    glauber_root: str
    sigmaNN_mb: float
    sqrt_sNN_GeV: float
    kick: str = "pp"
    verbose: bool = False

    # filled in __post_init__
    eloss: ELossRun = None
    ana: RpAAnalysis = None
    npdf_sys: NPDFSystem = None
    rgrid: pd.DataFrame = None
    model: CentralityModel = None
    ctab: pd.DataFrame = None
    ens = None
    runs = None

    def __post_init__(self):
        _pub_style()
        # tiny memoizers: (label, y_width, pt_range, state) -> DataFrame
        self._cache_cnm_y_in_cent: dict = {}
        self._cache_prim_y_in_cent: dict = {}
        # eLoss
        self.eloss = load_eloss_run(Path(self.e_loss_base), particle="JPsi")     # :contentReference[oaicite:4]{index=4}
        # nPDF
        self.ana = RpAAnalysis()                                                 # :contentReference[oaicite:5]{index=5}
        self.npdf_sys = NPDFSystem.from_folder(self.npdf_folder, kick=self.kick, name=f"p+Pb {self.tag} TeV")
        self.rgrid = self.ana.compute_rpa_grid(self.npdf_sys.df_pp, self.npdf_sys.df_pa, self.npdf_sys.df_errors, join="intersect")
        # centrality model
        SA = self._build_SA_provider(self.npdf_sys, self.sqrt_sNN_GeV)
        geom = WoodsSaxonPb()
        self.model = CentralityModel.from_inputs(SA, geom, b_max=12.0, nb=601)
        self.ctab = self.model.centrality_table(CENT_EDGES_ELOSS, sigmaNN_mb=self.sigmaNN_mb)
        try:
            ana0 = next(iter(self.runs.values()))  # any run; all share the same maps
            maps = ana0.centrality
            Ncols = []
            for _, row in self.ctab.iterrows():
                bL, bR = float(row["b_left"]), float(row["b_right"])
                bs = np.linspace(bL, bR, 101)
                nbin = maps.b_to_nbin(bs)
                Ncols.append(float(np.nanmean(nbin)))
            self.ctab["N_coll"] = Ncols
        except Exception:
            # keep going even if N_coll cannot be added
            pass        
        # normalize ctab labels
        self.ctab["cent_bin"] = self.ctab["cent_bin"].map(_norm_cent_label)
        # build a lookup from normalized → original eLoss tag
        self._eloss_tag_map = { _norm_cent_label(t): t for t in self.eloss.available_cent_tags }

        # primordial (HNM), τ-ensemble
        self.ens, self.runs = build_ensemble(self.prim_base, self.glauber_root, tags=("tau1","tau2"),
                                             cfg=ReaderConfig(debug=self.verbose))
        if abs(self.sqrt_sNN_GeV - 5020.0) > 1.0:
            for ana in self.runs.values():
                ana.set_energy_scaling_by_sqrts(self.sqrt_sNN_GeV)

    # -------- internals --------
    def _build_SA_provider(self, sys_npds: NPDFSystem, sqrt_sNN_GeV: float):
        folder = Path("./input/gluon_ratio")
        cand = list((folder/"5TeV").glob("*.*")) + list((folder/"8TeV").glob("*.*"))
        if cand:
            return GluonRatioTable(cand[0], sqrt_sNN_GeV=float(sqrt_sNN_GeV)).load()
        return GluonFromGrid(sys_npds.df_pp, sys_npds.df_pa)

    def _b_interval_for_cent(self, label) -> Tuple[float, float]:
        """
        Robust cent-bin resolver: accepts '0-10', '0–10%', pandas.Interval, etc.
        """
        # direct exact match first
        try:
            m = self.ctab["cent_bin"].astype(str).str.replace("–", "-").str.replace("%", "") == \
                str(label).strip().replace("–", "-").replace("%", "")
            if m.any():
                row = self.ctab.loc[m].iloc[0]
                return float(row["b_left"]), float(row["b_right"])
        except Exception:
            pass

        # normalize to numeric pair and match numerically
        def _to_pair(x):
            if hasattr(x, "left") and hasattr(x, "right"):
                return (float(x.left), float(x.right))
            s = str(x).strip().replace("–", "-").replace("%", "")
            L, R = s.split("-")
            return (float(L), float(R))

        try:
            L0, R0 = _to_pair(label)
            for _, row in self.ctab.iterrows():
                L1, R1 = _to_pair(row["cent_bin"])
                if abs(L1 - L0) < 1e-6 and abs(R1 - R0) < 1e-6:
                    return float(row["b_left"]), float(row["b_right"])
        except Exception:
            pass

        raise KeyError(f"centrality label not found: {label!r}")

    # --- utility: tolerant mapper from any cent label -> eLoss tag you have
    def _cent_label_to_tag(self, cent_label: str) -> str:
        norm = _norm_cent_label(cent_label)
        # fast path: exact normalized tag
        if hasattr(self, "_eloss_tag_map") and norm in self._eloss_tag_map:
            return self._eloss_tag_map[norm]
        # otherwise choose the eLoss bin with the largest overlap
        want = _parse_cent_label(norm)
        best_tag, best_w = None, -1.0
        for t in self.eloss.available_cent_tags:
            w = _overlap_width(want, _parse_cent_label(t))
            if w > best_w:
                best_w, best_tag = w, t
        return best_tag or next(iter(self.eloss.available_cent_tags))

    # --- centrality parsing (reuse your normalizer) ---
    def _cent_pair(self, s: str) -> tuple[float, float]:
        s = str(s).strip().replace("–", "-").replace("%", "")
        a, b = s.split("-")
        return float(a), float(b)

    def _largest_overlap_key(self, target: str, pool: list[str]) -> str:
        """Pick the label from pool that overlaps most with 'target'."""
        L0, R0 = self._cent_label_to_tag(target), None  # use the same parser you already have
        want = self._cent_pair(target)
        best, bestw = pool[0], -1.0
        for p in pool:
            L, R = self._cent_pair(p)
            w = max(0.0, min(want[1], R) - max(want[0], L))
            if w > bestw: best, bestw = p, w
        return best

    def prepare_exp_pt_overlay(self, exppt: pd.DataFrame, force_last_bin_to_80_100: bool = True) -> pd.DataFrame:
        """
        Return a copy of exppt with a new column 'cent_match' that is mapped onto this
        model's centrality labels (e.g. force ALICE 80–90% → 80–100%).
        """
        if exppt is None or exppt.empty:
            return pd.DataFrame()

        out = exppt.copy()
        # canonicalize experimental labels (e.g. "0–10%" → "0-10%")
        out["centrality"] = out["centrality"].astype(str).str.replace("–", "-")

        # force 80-90 → 80-100 if requested (ALICE 8 TeV case)
        if force_last_bin_to_80_100:
            out.loc[out["centrality"].str.fullmatch(r"\s*80-90%?\s*"), "centrality"] = "80-100%"

        # build model label pool
        cent_pool = [str(x) for x in self.ctab["cent_bin"]]

        # best-overlap mapping
        def _map_one(lbl: str) -> str:
            try:
                return self._largest_overlap_key(lbl, cent_pool)
            except Exception:
                return lbl
        out["cent_match"] = out["centrality"].map(_map_one)

        # light tidy: keep only what we need to draw
        keep = ["rapidity","cent_match","x_cen","x_low","x_high","value",
                "stat_up","stat_dn","sys_uncorr_up","sys_uncorr_dn"]
        keep = [c for c in keep if c in out.columns]
        out = out[keep].rename(columns={
            "x_cen":"pt", "x_low":"ptlo", "x_high":"pthi", "value":"val"
        }).reset_index(drop=True)

        # total uncorrelated error (stat ⊕ uncorrelated sys)
        sup = np.nan_to_num(out.get("stat_up", out.get("stat_dn")).astype(float), 0.0)
        uup = np.nan_to_num(out.get("sys_uncorr_up", out.get("sys_uncorr_dn")).astype(float), 0.0)
        out["dtot"] = np.sqrt(sup**2 + uup**2)
        return out


    # def _eloss_mean_centrality(self, cent_label: str, y_range: Tuple[float,float], pt_range: Tuple[float,float]):
    #     tag = cent_label.replace("%","")
    #     val = self.eloss.mean_rpa_over_y_and_pt(tag, y_range, pt_range)
    #     try:
    #         err = self.eloss.mean_rpa_err_over_y_and_pt(tag, y_range, pt_range)
    #         return (val, max(val-err, 0.0), val+err)
    #     except Exception:
    #         return (val, val, val)
    # ----------------- eLoss mean & band for a centrality label -----------------
    def _eloss_mean_centrality(
        self,
        cent_label: str,
        y_range: Tuple[float,float],
        pt_range: Tuple[float,float]
    ) -> Tuple[float,float,float]:
        """
        Return (central, lo, hi) for eLoss averaged over y_range × pt_range at the requested centrality.
        - If cent_label exists in eLoss → use it directly.
        - Else aggregate over overlapping eLoss bins with width-fraction weights.
        """
        # eLoss object can be ELossRun (single) or ELossEnsemble (band)
        E = self.eloss
        avail = set(E.available_cent_tags)

        # exact bin available
        if cent_label in avail:
            c  = E.mean_rpa_over_y_and_pt(cent_label, y_range, pt_range)
            hw = E.mean_rpa_err_over_y_and_pt(cent_label, y_range, pt_range)  # half-width
            return float(c), float(c - hw), float(c + hw)

        # aggregate over overlaps (e.g., requested '0-20' while eLoss has '0-10' & '10-20')
        want = _parse_cent_label(cent_label)
        parts: List[Tuple[float,float,float,float]] = []  # (w, c, lo, hi)

        for tag in E.available_cent_tags:
            have = _parse_cent_label(tag)
            w = _overlap_width(want, have)
            if w <= 0.0:
                continue
            c  = E.mean_rpa_over_y_and_pt(tag, y_range, pt_range)
            hw = E.mean_rpa_err_over_y_and_pt(tag, y_range, pt_range)
            parts.append((w, float(c), float(c - hw), float(c + hw)))

        if not parts:
            # fall back to uniform tiny band if nothing overlaps (shouldn't happen if bins are sensible)
            c = E.mean_rpa_over_y_and_pt(next(iter(E.available_cent_tags)), y_range, pt_range)
            hw = E.mean_rpa_err_over_y_and_pt(next(iter(E.available_cent_tags)), y_range, pt_range)
            return float(c), float(c - hw), float(c + hw)

        W = sum(p[0] for p in parts)
        # Weighted mean for central; aggregate lo/hi as weighted too (conservative but simple).
        c  = sum(w * c for (w, c, lo, hi) in parts) / W
        lo = sum(w * lo for (w, c, lo, hi) in parts) / W
        hi = sum(w * hi for (w, c, lo, hi) in parts) / W
        return float(c), float(lo), float(hi)

    # ----------------- CNM vs centrality (tables) -----------------
    def cnm_vs_centrality(self, y_range: Tuple[float,float], pt_range: Tuple[float,float]) -> pd.DataFrame:
        cent_edges = getattr(self.eloss, "centrality_edges", CENT_EDGES_ELOSS)
        tab_np = self.model.rpa_vs_centrality_integrated(
            self.rgrid, self.npdf_sys.df_pa, cent_edges,
            y_min=y_range[0], y_max=y_range[1],
            pt_min=pt_range[0], pt_max=pt_range[1],
            sigmaNN_mb=self.sigmaNN_mb, weight="inelastic", verbose=False
        ).copy()
        if tab_np.empty:
            return pd.DataFrame(columns=["cent_bin","r_central","r_lo","r_hi","eloss","eloss_lo","eloss_hi","cnm_c","cnm_lo","cnm_hi"])

        e_c, e_lo, e_hi = [], [], []
        for label in tab_np["cent_bin"]:
            c, lo, hi = self._eloss_mean_centrality(label, y_range, pt_range)
            e_c.append(c); e_lo.append(lo); e_hi.append(hi)
        e_c, e_lo, e_hi = map(np.array, (e_c, e_lo, e_hi))

        Cc, Clo, Chi = combine_product_asym(
            tab_np["r_central"].to_numpy(), tab_np["r_lo"].to_numpy(), tab_np["r_hi"].to_numpy(),
            e_c, e_lo, e_hi
        )

        out = tab_np.copy()
        out["eloss"]     = e_c
        out["eloss_lo"]  = e_lo         # <- new
        out["eloss_hi"]  = e_hi         # <- new
        out["cnm_c"]     = Cc
        out["cnm_lo"]    = Clo
        out["cnm_hi"]    = Chi
        return out

    def _ncoll_column(self):
        for k in ("Ncoll","n_coll","N_coll","<Ncoll>","<N_coll>"):
            if k in self.ctab.columns:
                return k
        return None

    def cnm_vs_Ncoll(self, y_window, pt_range):
        base = self.cnm_vs_centrality(y_window, pt_range)
        col = self._ncoll_column()
        if base.empty or col is None:
            return pd.DataFrame()
        return base.merge(self.ctab[["cent_bin", col]], on="cent_bin", how="left").rename(columns={col:"Ncoll"})

    # --- combine_module.py: patch total_vs_Ncoll ---
    def total_vs_Ncoll(self, y_window, pt_range, state):
        rows = []
        cnm  = self.cnm_vs_centrality(y_window, pt_range)
        prim = self.primordial_vs_centrality(pt_range, y_window, state)

        # ✅ normalize labels to one canonical form
        cnm["cent_bin"]  = cnm["cent_bin"].map(_norm_cent_label)
        prim["cent_bin"] = prim["cent_bin"].map(_norm_cent_label)

        for _, row in self.ctab.iterrows():
            cb    = _norm_cent_label(row["cent_bin"])
            # use Glauber N_coll if present, else N_part - 1 (p+Pb identity)
            ncoll = float(row["N_coll"]) if "N_coll" in row and not pd.isna(row["N_coll"]) \
                    else float(row["N_part"]) - 1.0

            c1 = cnm.loc[cnm["cent_bin"] == cb]
            p1 = prim.loc[prim["cent_bin"] == cb]
            if c1.empty or p1.empty:
                continue

            Cc, Clo, Chi = [float(c1[k].values[0]) for k in ("cnm_c","cnm_lo","cnm_hi")]
            Pc, Plo, Phi = [float(p1[k].values[0]) for k in ("c","lo","hi")]
            Tc, Tlo, Thi = combine_product_asym(Cc, Clo, Chi, Pc, Plo, Phi)
            rows.append(dict(N_coll=ncoll, c=Tc, lo=Tlo, hi=Thi, cent_bin=cb))

        # ✅ be graceful if nothing matched
        if not rows:
            return pd.DataFrame(columns=["N_coll","c","lo","hi","cent_bin"])
        return pd.DataFrame(rows).sort_values("N_coll").reset_index(drop=True)


   # --------------- Primordial & Total vs centrality ----------------
    def primordial_vs_centrality(self, pt_range: Tuple[float,float], y_range: Tuple[float,float], state: str) -> pd.DataFrame:
        # Average primordial(y) inside [ymin,ymax] → one number per impact parameter b
        y_bins = make_bins_from_width(y_range[0], y_range[1], 0.5)
        ryb_center, ryb_band = self.ens.central_and_band_vs_y_per_b(
            pt_window=pt_range, y_bins=y_bins, with_feeddown=True, use_nbin=True, flip_y=True
        )
        # --- FIX: avoid pandas FutureWarning by selecting columns before groupby
        C = (ryb_center[["b", state]]
             .groupby("b", sort=True)[state].mean()
             .reset_index(name="c"))
        if f"{state}_lo" in ryb_band.columns and f"{state}_hi" in ryb_band.columns:
            Blo = (ryb_band[["b", f"{state}_lo"]]
                   .groupby("b", sort=True)[f"{state}_lo"].mean()
                   .reset_index(name="lo"))
            Bhi = (ryb_band[["b", f"{state}_hi"]]
                   .groupby("b", sort=True)[f"{state}_hi"].mean()
                   .reset_index(name="hi"))
        else:
            errc = f"{state}_err"
            E = (ryb_center[["b", errc]]
                 .groupby("b", sort=True)[errc].mean()
                 .reset_index(name="err"))
            B = C.merge(E, on="b", how="left")
            Blo = B.assign(lo=B["c"] - B["err"])[["b","lo"]]
            Bhi = B.assign(hi=B["c"] + B["err"])[["b","hi"]]
        M = C.merge(Blo, on="b").merge(Bhi, on="b")

        rows=[]
        for label in self.ctab["cent_bin"]:
            bL, bR = self._b_interval_for_cent(label)
            sub = _nearest_b_rows(M, bL, bR)
            if sub.empty:
                rows.append(dict(cent_bin=label, c=np.nan, lo=np.nan, hi=np.nan))
            else:
                c, lo, hi = avg_band_by_weights(sub["c"], sub["lo"], sub["hi"], np.ones(len(sub)))
                rows.append(dict(cent_bin=label, c=c, lo=lo, hi=hi))
        return pd.DataFrame(rows)

    def total_vs_centrality(self, y_range, pt_range, state: str) -> pd.DataFrame:
        cnm  = self.cnm_vs_centrality(y_range, pt_range)
        prim = self.primordial_vs_centrality(pt_range, y_range, state)
        if cnm.empty or prim.empty:
            return pd.DataFrame(columns=[...])

        # ✅ ensure both sides use identical labels
        cnm["cent_bin"]  = cnm["cent_bin"].map(_norm_cent_label)
        prim["cent_bin"] = prim["cent_bin"].map(_norm_cent_label)

        out = cnm.merge(prim, on="cent_bin", how="inner", suffixes=("_cnm","_prim"))
        Tc, Tlo, Thi = combine_product_asym(
            out["cnm_c"].to_numpy(), out["cnm_lo"].to_numpy(), out["cnm_hi"].to_numpy(),
            out["c"].to_numpy(),     out["lo"].to_numpy(),     out["hi"].to_numpy()
        )
        out["total_c"], out["total_lo"], out["total_hi"] = Tc, Tlo, Thi
        return out

    # ============================= Figures =============================
    # ---- 0) One flexible centrality plotter (choose what to show)
    def figure_result_vs_centrality(
        self,
        y_windows: List[Tuple[float,float]],
        pt_range: Tuple[float,float],
        components: Sequence[str] = ("nPDF", "eLoss", "CNM"),  # allow "Primordial[:state]", "Total[:state]"
        states: Optional[List[str]] = None,
        cnm_as: str = "errorbar",   # "errorbar" (default) or "band"
        save_pdf: Optional[str] = None,
        legend_mode: str = "figure",
        legend_loc: str = "best",
        legend_ncols: int = 3,
        ylim: Optional[Tuple[float,float]] = (0.0, 1.5),
        xticks: Optional[Sequence[float]] = (10, 30, 50, 80),
        ncols: Optional[int] = None,
        annotate_y_note: bool = True,
        note_loc: str = "lower right",
        minor_ticks: bool = True,
    ):
        comp_list = [str(c) for c in components]
        states = states or []

        n = len(y_windows)
        ncols = int(ncols or n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.8*ncols, 4.8*nrows),
                                sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        handles0, labels0 = None, None

        for j, (ywin, ax) in enumerate(zip(y_windows, axes)):
            ymin, ymax = ywin
            cnm_tab = self.cnm_vs_centrality((ymin, ymax), pt_range)
            # style (NO title)
            _style_axes(ax, "centrality [%]", r"$R_{pA}$", title=None, minor_ticks=minor_ticks)

            if cnm_tab.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
                continue

            # nPDF
            if "nPDF" in comp_list:
                _plot_rpa_vs_centrality_hzerr(
                    ax, cnm_tab[["cent_bin","r_central","r_lo","r_hi"]],
                    label="nPDF", color=COMP_COLORS["nPDF"]
                )
            # eLoss
            if "eLoss" in comp_list:
                edges = np.array([_parse_centbin(s) for s in cnm_tab["cent_bin"]], dtype=float)
                x = 0.5*(edges[:,0] + edges[:,1])
                ax.plot(x, cnm_tab["eloss"], "s-", label="eLoss", color=COMP_COLORS["eLoss"])
            # CNM band
            if "CNM" in comp_list:
                edges = np.array([_parse_centbin(s) for s in cnm_tab["cent_bin"]], dtype=float)
                x = 0.5*(edges[:,0] + edges[:,1])
                if cnm_as == "band":
                    # piecewise-constant slab band across centrality bins
                    cent_edges = np.r_[edges[:,0], edges[-1,1]]
                    X = np.repeat(cent_edges, 2)[1:-1]
                    Ylo = np.repeat(cnm_tab["cnm_lo"].to_numpy(float), 2)
                    Yhi = np.repeat(cnm_tab["cnm_hi"].to_numpy(float), 2)
                    ax.fill_between(X, Ylo, Yhi, color=COMP_COLORS["CNM"], alpha=BAND_ALPHA["CNM"], label="CNM")
                else:
                    y  = cnm_tab["cnm_c"].to_numpy()
                    yerr = np.vstack([y - cnm_tab["cnm_lo"].to_numpy(), cnm_tab["cnm_hi"].to_numpy() - y])
                    ax.errorbar(x, y, yerr=yerr, fmt="D", capsize=3, label="CNM", color=COMP_COLORS["CNM"])

            # Primordial/Total for specified states
            for comp in comp_list:
                if comp.startswith("Primordial:") or comp.startswith("Total:"):
                    kind, st = comp.split(":", 1)
                    self._draw_prim_total_centrality(ax, (ymin, ymax), pt_range, st, kind)
            if ("Primordial" in comp_list) or ("Total" in comp_list):
                for st in states:
                    if "Primordial" in comp_list:
                        self._draw_prim_total_centrality(ax, (ymin, ymax), pt_range, st, "Primordial")
                    if "Total" in comp_list:
                        # BUGFIX: pass the correct selector
                        self._draw_prim_total_centrality(ax, (ymin, ymax), pt_range, st, "Total")

            if ylim: ax.set_ylim(*ylim)
            if xticks: ax.set_xticks(xticks)
            if annotate_y_note:
                _annotate_corner(ax, f"{ymin:.1f} < y < {ymax:.1f}", loc=note_loc)

            h, l = ax.get_legend_handles_labels()
            if handles0 is None: handles0, labels0 = h, l
            _legend_apply(ax, h, l, legend_mode, legend_loc, j, grid_first_panel=0)

        for k in range(len(y_windows), len(axes)):
            axes[k].axis("off")
        if legend_mode == "figure" and handles0:
            axes[0].legend(handles0, labels0, loc=legend_loc, ncols=legend_ncols, frameon=False)
        if save_pdf:
            fig.savefig(save_pdf, bbox_inches="tight")
        plt.show()

    # ---- 0) One flexible centrality plotter (choose what to show)
    def figure_result_vs_centrality_by_state(
        self,
        y_windows: Sequence[Tuple[float, float]],
        pt_range: Tuple[float, float],
        states: Sequence[str],
        *,
        components: Sequence[str] = ("nPDF", "eLoss", "CNM", "Primordial", "Total"),
        save_pdf: Optional[str] = None,
        legend_mode: str = "each",
        legend_loc: str = "best",
        legend_ncols: int = 3,
        ylim: Tuple[float, float] = (0.0, 1.5),
        xticks: Sequence[float] = (10, 30, 50, 80),
        ncols: Optional[int] = None,
        minor_ticks: bool = True,
        note_loc: str = "upper right",
    ):
        """
        Columns = states, rows = y_windows.
        Draw slab bands across full centrality bins for nPDF/CNM/Primordial/Total.
        eLoss stays as a line (no errors).
        """
        comps = [str(c) for c in components]
        nrow, ncol = len(y_windows), (ncols or len(states))

        import numpy as np
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            nrow, ncol, figsize=(5.8 * ncol, 4.8 * nrow),
            sharex=True, sharey=True, constrained_layout=True
        )
        axes = np.atleast_1d(axes).reshape(nrow, ncol)

        first_handles, first_labels = None, None

        # helper: connected slab (piecewise-constant) band across bin edges
        def _fill_bin_band(ax, edges, lo, hi, *, color, alpha, label=None):
            E = np.asarray(edges, float)                # [0,20,40,60,100]
            lo = np.asarray(lo, float)                  # len = nbins
            hi = np.asarray(hi, float)
            X   = np.repeat(E, 2)[1:-1]                 # 0,20,20,40,40,60,60,100  (no gaps)
            Ylo = np.repeat(lo, 2)
            Yhi = np.repeat(hi, 2)
            ax.fill_between(X, Ylo, Yhi, color=color, alpha=alpha, label=label, linewidth=0)

        for i, (ymin, ymax) in enumerate(y_windows):
            for j, state in enumerate(states):
                ax = axes[i, j]
                tab = self.total_vs_centrality((ymin, ymax), pt_range, state)

                _style_axes(ax, "centrality [%]", r"$R_{pA}$", title=None, minor_ticks=minor_ticks)
                if tab.empty:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", alpha=0.6)
                    continue

                pairs = np.array([_parse_centbin(s) for s in tab["cent_bin"]], dtype=float)
                cent_edges  = np.r_[pairs[:, 0], pairs[-1, 1]]     # e.g. [0,20,40,60,100]
                cent_center = 0.5 * (pairs[:, 0] + pairs[:, 1])    # for eLoss only

                # nPDF: band (no markers/lines)
                if "nPDF" in comps:
                    _fill_bin_band(
                        ax, cent_edges, tab["r_lo"], tab["r_hi"],
                        color=COMP_COLORS["nPDF"], alpha=BAND_ALPHA.get("CNM", 0.22),
                        label="nPDF"
                    )

                # eLoss: line only
                if "eLoss" in comps and "eloss" in tab.columns:
                    ax.plot(cent_center, tab["eloss"], "s-", label="eLoss", color=COMP_COLORS["eLoss"])

                # CNM: band only
                if "CNM" in comps:
                    _fill_bin_band(
                        ax, cent_edges, tab["cnm_lo"], tab["cnm_hi"],
                        color=COMP_COLORS["CNM"], alpha=BAND_ALPHA["CNM"], label="CNM"
                    )

                # Primordial: band only
                if "Primordial" in comps:
                    _fill_bin_band(
                        ax, cent_edges, tab["lo"], tab["hi"],
                        color=COMP_COLORS["Primordial"], alpha=BAND_ALPHA["Primordial"], label="Primordial"
                    )

                # Total: band only (state color)
                if "Total" in comps:
                    col = STATE_COLORS.get(state, COMP_COLORS["Total"])
                    _fill_bin_band(
                        ax, cent_edges, tab["total_lo"], tab["total_hi"],
                        color=col, alpha=BAND_ALPHA["Total"], label="Total"
                    )

                if ylim:   ax.set_ylim(*ylim)
                if xticks: ax.set_xticks(xticks)

                _annotate_corner(ax, PRETTY_STATE.get(state, state), loc="upper left")
                _annotate_corner(ax, f"{ymin:.1f} < y < {ymax:.1f}", loc=note_loc)

                h, l = ax.get_legend_handles_labels()
                if first_handles is None:
                    first_handles, first_labels = h, l
                _legend_apply(ax, h, l, legend_mode, legend_loc,
                            panel_index=(i * ncol + j), grid_first_panel=0)

        if legend_mode == "figure" and first_handles:
            axes[0, 0].legend(first_handles, first_labels,
                            loc=legend_loc, ncols=legend_ncols, frameon=False)
        if save_pdf:
            plt.savefig(save_pdf, bbox_inches="tight")
        plt.show()



    def _draw_prim_total_centrality(self, ax, y_range, pt_range, state: str, which: str):
        tab = self.total_vs_centrality(y_range, pt_range, state)
        if tab.empty:
            return
        edges = np.array([_parse_centbin(s) for s in tab["cent_bin"]], dtype=float)
        x = 0.5*(edges[:,0] + edges[:,1])
        if which == "Primordial":
            ax.errorbar(
                x, tab["c"],
                yerr=np.vstack([tab["c"]-tab["lo"], tab["hi"]-tab["c"]]),
                fmt="s", capsize=3,
                label=f"Primordial {PRETTY_STATE.get(state,state)}",
                color=COMP_COLORS["Primordial"]
            )
        elif which == "Total":
            ax.errorbar(
                x, tab["total_c"],
                yerr=np.vstack([tab["total_c"]-tab["total_lo"], tab["total_hi"]-tab["total_c"]]),
                fmt="D", capsize=3,
                label=f"CNM X Primordial {PRETTY_STATE.get(state,state)}",
                color=STATE_COLORS.get(state, COMP_COLORS["Total"])
            )
        elif which == "CNM":
            _plot_rpa_vs_centrality_hzerr(
                ax,
                tab[["cent_bin","cnm_c","cnm_lo","cnm_hi"]].rename(
                    columns={"cnm_c":"r_central","cnm_lo":"r_lo","cnm_hi":"r_hi"}),
                label="CNM", color=COMP_COLORS["CNM"], draw_line=False
            )

    # ---- Back-compatible convenience wrappers -------------------------------
    def figure_cnm_vs_centrality(self, y_windows, pt_range,
                                 save_pdf: Optional[str]=None, legend_mode="figure", legend_loc="best",
                                 ylim=(0.0,1.5), xticks=(10,30,50,80), ncols: Optional[int]=None,
                                 minor_ticks: bool=True, note_loc: str="lower right"):
        self.figure_result_vs_centrality(
            y_windows=y_windows, pt_range=pt_range,
            components=("nPDF","eLoss","CNM"),
            save_pdf=save_pdf, legend_mode=legend_mode, legend_loc=legend_loc,
            ylim=ylim, xticks=xticks, ncols=ncols, annotate_y_note=True,
            note_loc=note_loc, minor_ticks=minor_ticks,
        )

    def figure_total_vs_centrality(
        self,
        y_windows,
        pt_range,
        states,
        *,
        components: Sequence[str] = ("nPDF","eLoss","CNM","Primordial","Total"),
        save_pdf_prefix: Optional[str]=None,
        legend_mode="panel-first",
        legend_loc="best",
        ylim=(0.0,1.5),
        xticks=(10,30,50,80),
        ncols: Optional[int]=None,
        minor_ticks: bool=True,
        note_loc: str="lower right"
    ):
        nrow, ncol = len(states), (ncols or len(y_windows))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.8*ncol, 4.8*nrow),
                                sharex=True, sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes).reshape(nrow, ncol)

        for i, state in enumerate(states):
            for j, (ymin, ymax) in enumerate(y_windows):
                ax = axes[i, j]
                tab = self.total_vs_centrality((ymin, ymax), pt_range, state)
                # style (NO title)
                _style_axes(ax, "centrality [%]", r"$R_{pA}$", title=None, minor_ticks=minor_ticks)

                if tab.empty:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
                    continue

                # Optionally show nPDF and eLoss alongside CNM/Primordial/Total
                if "nPDF" in components:
                    _plot_rpa_vs_centrality_hzerr(
                        ax,
                        tab[["cent_bin","r_central","r_lo","r_hi"]] if {"r_central","r_lo","r_hi"}.issubset(tab.columns)
                        else self.cnm_vs_centrality((ymin, ymax), pt_range)[["cent_bin","r_central","r_lo","r_hi"]],
                        label="nPDF", color=COMP_COLORS["nPDF"], draw_line=False
                    )
                if "eLoss" in components:
                    edges = np.array([_parse_centbin(s) for s in tab["cent_bin"]], dtype=float)
                    x = 0.5*(edges[:,0] + edges[:,1])
                    if "eloss" in tab.columns:
                        ax.plot(x, tab["eloss"], "s-", label="eLoss", color=COMP_COLORS["eLoss"])

                if "CNM" in components:
                    _plot_rpa_vs_centrality_hzerr(
                        ax,
                        tab[["cent_bin","cnm_c","cnm_lo","cnm_hi"]].rename(
                            columns={"cnm_c":"r_central","cnm_lo":"r_lo","cnm_hi":"r_hi"}),
                        label="CNM", color=COMP_COLORS["CNM"], draw_line=False
                    )

                if "Primordial" in components:
                    edges = np.array([_parse_centbin(s) for s in tab["cent_bin"]], dtype=float)
                    x = 0.5*(edges[:,0] + edges[:,1])
                    ax.errorbar(
                        x, tab["c"],
                        yerr=np.vstack([tab["c"]-tab["lo"], tab["hi"]-tab["c"]]),
                        fmt="s", capsize=3, label="Primordial", color=COMP_COLORS["Primordial"]
                    )

                if "Total" in components:
                    edges = np.array([_parse_centbin(s) for s in tab["cent_bin"]], dtype=float)
                    x = 0.5*(edges[:,0] + edges[:,1])
                    ax.errorbar(
                        x, tab["total_c"],
                        yerr=np.vstack([tab["total_c"]-tab["total_lo"], tab["total_hi"]-tab["total_c"]]),
                        fmt="D", capsize=3, label="CNM X Primordial",
                        color=STATE_COLORS.get(state, COMP_COLORS["Total"])
                    )

                if ylim: ax.set_ylim(*ylim)
                if xticks: ax.set_xticks(xticks)

                # Notes inside: state (as before) + y-window (new)
                _annotate_corner(ax, PRETTY_STATE.get(state, state), loc="upper left")
                _annotate_corner(ax, f"{ymin:.1f} < y < {ymax:.1f}", loc=note_loc)

                _legend_apply(
                    ax, *ax.get_legend_handles_labels(),
                    legend_mode=legend_mode, legend_loc=legend_loc,
                    panel_index=(i*ncol + j), grid_first_panel=0
                )

        if save_pdf_prefix:
            plt.savefig(f"{save_pdf_prefix}_{state}.pdf", bbox_inches="tight")
        plt.show()

    # ------------------------ y / pT internals & figures ---------------------
    def _npdf_vs_y_in_cent(self, cent_label: str, y_width,   # float OR bins
                        pt_range: tuple[float, float]) -> pd.DataFrame:
        import numpy as np
        width = _normalize_y_width(y_width)
        edges = getattr(self.eloss, "centrality_edges", CENT_EDGES_ELOSS)
        bands = self.model.rpa_vs_y_in_centrality_bins(
            self.rgrid, self.npdf_sys.df_pa, edges,
            y_width=width, pt_min=pt_range[0],
            sigmaNN_mb=self.sigmaNN_mb, weight="inelastic", verbose=False
        )
        d = dict(bands)

        def _norm(s: str) -> tuple[float, float]:
            s = str(s).strip().replace("–", "-").replace("%", "")
            a, b = [float(x) for x in s.split("-")]
            # tiny rounding avoids "0.0" vs "0" mismatches
            return (round(a, 3), round(b, 3))

        dfN = d.get(cent_label)
        if dfN is None:
            want = _norm(cent_label)
            for k, v in d.items():
                if _norm(k) == want:
                    dfN = v
                    break

        if dfN is None or dfN.empty:
            return pd.DataFrame(columns=["y", "c", "lo", "hi"])

        y_left = dfN["y_left"].to_numpy(float)
        dy = (y_left[1] - y_left[0]) if len(y_left) > 1 else float(y_width)
        y_edges = np.r_[y_left, y_left[-1] + dy]
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        return pd.DataFrame({
            "y":  y_centers,
            "c":  dfN["r_central"].to_numpy(float),
            "lo": dfN["r_lo"].to_numpy(float),
            "hi": dfN["r_hi"].to_numpy(float),
        })



    # --- eLoss(y) in one centrality: return y, c, lo, hi (consistent with others)
    def _eloss_vs_y_in_cent(self, cent_label: str,
                            pt_range: tuple[float, float],
                            y_width: float = 0.5) -> pd.DataFrame:
        width = _normalize_y_width(y_width)
        tag = self._cent_label_to_tag(cent_label)
        y_lo, y_hi = self.eloss.cent_bins[tag].y_range
        start = width * math.floor(y_lo / width)
        stop  = width * math.ceil(y_hi / width) + 1e-12
        y_edges = np.arange(start, stop, width)
        df = self.eloss.rpa_vs_y(cent_tag=tag, y_edges=y_edges, pt_range=pt_range)
        if df is None or df.empty:
            return pd.DataFrame(columns=["y","c","lo","hi"])
        out = df.rename(columns={"y_mid": "y", "RpA": "c"})
        cols = [c for c in ("y","c","lo","hi") if c in out.columns]
        return out[cols].reset_index(drop=True)


    # --- nPDF×eLoss → CNM(y) in one centrality; y_width can be float OR list of bins
    def _cnm_vs_y_in_cent(self, cent_label: str, y_width, pt_range: tuple[float,float]) -> pd.DataFrame:
        def _width(v):
            try: return float(v)
            except:
                (a,b) = v if len(v)==2 else v[0]
                return float(abs(b-a))
        w = _width(y_width)

        N = self._npdf_vs_y_in_cent(cent_label, y_width=w, pt_range=pt_range)   # y,c,lo,hi
        E = self._eloss_vs_y_in_cent(cent_label, pt_range=pt_range, y_width=w)  # y,c,lo,hi
        if N.empty or E.empty:
            return pd.DataFrame(columns=["y","c","lo","hi"])

        y = np.union1d(N["y"].to_numpy(float), E["y"].to_numpy(float))
        Nc, Nlo, Nhi = (np.interp(y, N["y"], N["c"]),
                        np.interp(y, N["y"], N["lo"]),
                        np.interp(y, N["y"], N["hi"]))
        Ec, Elo, Ehi = (np.interp(y, E["y"], E["c"]),
                        np.interp(y, E["y"], E["lo"]),
                        np.interp(y, E["y"], E["hi"]))
        Cc, Clo, Chi = combine_product_asym(Nc, Nlo, Nhi, Ec, Elo, Ehi)
        return pd.DataFrame({"y": y, "c": Cc, "lo": Clo, "hi": Chi})


    # --- Primordial(y) in one centrality; tolerate both call orders:
    #     (cent, state, pt_range, y_width=...)   OR   (cent, y_width, pt_range, state)
    def _primordial_vs_y_in_cent(self, cent_label: str, *args, **kwargs) -> pd.DataFrame:
        # Accept (cent, state, pt_range, y_width) OR (cent, y_width, pt_range, state)
        def _width(v):
            try: return float(v)
            except Exception:
                (a, b) = v if len(v) == 2 else v[0]
                return float(abs(b - a))

        if "state" in kwargs and "pt_range" in kwargs:
            state    = kwargs["state"]
            pt_range = tuple(map(float, kwargs["pt_range"]))
            width    = _width(kwargs.get("y_width", 0.5))
        elif len(args) == 3 and isinstance(args[0], str):
            state, pt_range, width = args[0], tuple(map(float, args[1])), _width(args[2])
        elif len(args) == 3:
            width, pt_range, state = _width(args[0]), tuple(map(float, args[1])), args[2]
        else:
            raise TypeError("Usage: (cent_label, state, pt_range, y_width) or (cent_label, y_width, pt_range, state)")

        y_bins = make_bins_from_width(-5, 5, width)
        ryb_center, ryb_band = self.ens.central_and_band_vs_y_per_b(
            pt_window=pt_range, y_bins=y_bins, with_feeddown=True, use_nbin=True, flip_y=True
        )
        C = ryb_center[["b","y",state]].rename(columns={state:"c"}).copy()
        if f"{state}_lo" in ryb_band.columns and f"{state}_hi" in ryb_band.columns:
            B = ryb_band[["b","y",f"{state}_lo",f"{state}_hi"]].rename(
                columns={f"{state}_lo":"lo", f"{state}_hi":"hi"})
        else:
            errc = f"{state}_err"
            B = ryb_center[["b","y",state,errc]].copy()
            B["lo"] = B[state] - B[errc]; B["hi"] = B[state] + B[errc]
            B = B[["b","y","lo","hi"]]
        M = C.merge(B, on=["b","y"], how="inner")
        bL, bR = self._b_interval_for_cent(cent_label)
        sub = _nearest_b_rows(M, bL, bR)
        if sub.empty:
            return pd.DataFrame(columns=["y","c","lo","hi"])
        rows = []
        for yv, g in sub.groupby("y", sort=True):
            w = np.ones(len(g))
            c  = float(np.average(g["c"],           weights=w))
            dlo= float(np.average(g["c"]-g["lo"],   weights=w))
            dhi= float(np.average(g["hi"]-g["c"],   weights=w))
            rows.append(dict(y=float(yv), c=c, lo=c-dlo, hi=c+dhi))
        return pd.DataFrame(rows).sort_values("y").reset_index(drop=True)



    def figure_total_vs_y_per_centrality(self, pt_range, states, y_width: float=0.5,
                                         save_pdf_prefix: Optional[str]=None, legend_mode="panel-first", legend_loc="best",
                                         ylim=(0.0,1.5), ncols: Optional[int]=3, minor_ticks: bool=True, note_loc: str="lower right"):
        labs = list(self.ctab["cent_bin"])
        n = len(labs); ncols = int(min(ncols or 3, n)); nrows = int(np.ceil(n/ncols))

        for state in states:
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.6*ncols, 4.4*nrows), sharex=True, sharey=True, constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for k, (ax, lab) in enumerate(zip(axes, labs)):
                cnm = self._cnm_vs_y_in_cent(lab, y_width=y_width, pt_range=pt_range)
                if not cnm.empty:
                    ax.fill_between(cnm["y"], cnm["lo"], cnm["hi"], alpha=BAND_ALPHA["CNM"], color=COMP_COLORS["CNM"], label="CNM")
                    ax.plot(cnm["y"], cnm["c"], lw=1.8, color=COMP_COLORS["CNM"])
                prim = self._primordial_vs_y_in_cent(lab, state, pt_range, y_width=y_width)
                if not prim.empty:
                    ax.fill_between(prim["y"], prim["lo"], prim["hi"], alpha=BAND_ALPHA["Primordial"], color=COMP_COLORS["Primordial"], label="Primordial")
                    ax.plot(prim["y"], prim["c"], lw=1.6, color=COMP_COLORS["Primordial"])
                if (not cnm.empty) and (not prim.empty):
                    y = prim["y"].to_numpy()
                    Cc  = np.interp(y, cnm["y"], cnm["c"])
                    Clo = np.interp(y, cnm["y"], cnm["lo"])
                    Chi = np.interp(y, cnm["y"], cnm["hi"])
                    Tc, Tlo, Thi = combine_product_asym(Cc, Clo, Chi, prim["c"], prim["lo"], prim["hi"])
                    ax.fill_between(y, Tlo, Thi, alpha=BAND_ALPHA["Total"], color=STATE_COLORS.get(state, COMP_COLORS["Total"]),
                                    label=f"CNM x Primordial")
                    ax.plot(y, Tc, lw=2.0, color=STATE_COLORS.get(state, COMP_COLORS["Total"]))

                _style_axes(ax, "y", r"$R_{pA}$", title=f"{lab}", minor_ticks=minor_ticks)
                if ylim: ax.set_ylim(*ylim)
                _annotate_corner(ax, PRETTY_STATE.get(state, state), loc=note_loc)
                _legend_apply(ax, *ax.get_legend_handles_labels(),
                              legend_mode=legend_mode, legend_loc=legend_loc,
                              panel_index=k, grid_first_panel=0)

            for ax in axes[len(labs):]: ax.axis("off")
            if legend_mode == "figure":
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False)
            if save_pdf_prefix:
                fig.savefig(f"{save_pdf_prefix}_{state}.pdf", bbox_inches="tight")
            plt.show()
            
    # ---- 4) Total vs NColl  
    def total_vs_Ncoll(self, y_window: tuple[float,float], pt_range: tuple[float,float], state: str) -> pd.DataFrame:
        cnm  = self.cnm_vs_centrality(y_window, pt_range)
        prim = self.primordial_vs_centrality(pt_range, y_window, state)
        if cnm.empty or prim.empty:
            return pd.DataFrame(columns=["N_coll","c","lo","hi","cent_bin"])

        # bring both to the same canonical centrality labels
        cnm["cent_bin"]  = cnm["cent_bin"].map(_norm_cent_label)
        prim["cent_bin"] = prim["cent_bin"].map(_norm_cent_label)

        rows = []
        for _, row in self.ctab.iterrows():
            cb = _norm_cent_label(row["cent_bin"])
            # Prefer Glauber <N_coll>; fall back to <N_part>-1 for p+Pb
            if "N_coll" in row and not pd.isna(row["N_coll"]):
                ncoll = float(row["N_coll"])
            elif "N_part" in row:
                ncoll = float(row["N_part"]) - 1.0
            else:
                continue

            c1 = cnm.loc[cnm["cent_bin"]==cb]
            p1 = prim.loc[prim["cent_bin"]==cb]
            if c1.empty or p1.empty:
                continue

            Cc, Clo, Chi = [float(c1[k].values[0]) for k in ("cnm_c","cnm_lo","cnm_hi")]
            Pc, Plo, Phi = [float(p1[k].values[0]) for k in ("c","lo","hi")]
            Tc, Tlo, Thi = combine_product_asym(Cc, Clo, Chi, Pc, Plo, Phi)
            rows.append(dict(N_coll=ncoll, c=Tc, lo=Tlo, hi=Thi, cent_bin=cb))

        if not rows:
            return pd.DataFrame(columns=["N_coll","c","lo","hi","cent_bin"])
        return pd.DataFrame(rows).sort_values("N_coll").reset_index(drop=True)


    def figure_total_vs_pt_per_centrality(
        self, y_windows, pt_bins, states,
        save_pdf_prefix: Optional[str]=None, legend_mode="panel-first", legend_loc="best",
        ylim=(0.0,1.5), xlim=(0.0,20.0), ncols_per_cent: Optional[int]=None,
        minor_ticks: bool=True, note_loc: str="lower right"
    ):
        labs = list(self.ctab["cent_bin"])
        ncent = len(labs)
        ny = len(y_windows)
        order = list(range(ny))  # keep your natural order

        import matplotlib.pyplot as plt
        from npdf_module import band_xy as _band_xy

        fig, axes = plt.subplots(1, ncent * ny, figsize=(4.2*ncent*ny, 4.6),
                                 sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)

        panel = 0
        for ic, lab in enumerate(labs):
            for jj, idx in enumerate(order):
                ax = axes[panel]; panel += 1
                ywin = y_windows[idx]
                # Single legend color is from the first state; curves are identical across states’ CNM
                for state in states[:1]:
                    cnm, prim, tot = self.total_vs_pt_in_centrality(lab, state, ywin, pt_bins)
                    if not cnm.empty:
                        _band_xy(ax, cnm["pt"], cnm["c"], cnm["lo"], cnm["hi"],
                                 label="CNM", color=COMP_COLORS["CNM"])
                    if not prim.empty:
                        _band_xy(ax, prim["pt"], prim["c"], prim["lo"], prim["hi"],
                                 label=f"Primordial {PRETTY_STATE.get(state, state)}",
                                 color=COMP_COLORS["Primordial"])
                    if not tot.empty:
                        _band_xy(ax, tot["pt"], tot["c"], tot["lo"], tot["hi"],
                                 label=f"CNM × Primordial {PRETTY_STATE.get(state, state)}",
                                 color=STATE_COLORS.get(state, COMP_COLORS["Total"]))
                note = f"{ywin[0]:.1f}<y<{ywin[1]:.1f}"
                _style_axes(ax, r"$p_T$ [GeV]", r"$R_{pA}$",
                            title=f"{lab} — {note}", minor_ticks=minor_ticks)
                if xlim: ax.set_xlim(*xlim)
                if ylim: ax.set_ylim(*ylim)

                _legend_apply(ax, *ax.get_legend_handles_labels(),
                              legend_mode=legend_mode, legend_loc=legend_loc,
                              panel_index=(panel-1), grid_first_panel=0)

        if save_pdf_prefix:
            fig.savefig(f"{save_pdf_prefix}_{state}.pdf", bbox_inches="tight")
        plt.show()

    # ---- pT helpers ----------------------------------------------------------

    def _npdf_pt_in_cent(self, cent_label: str, y_window: Tuple[float,float],
                         pt_bins: Sequence[Tuple[float,float]]) -> pd.DataFrame:
        width = pt_bins[0][1] - pt_bins[0][0]
        df = self.ana.rpa_vs_pt_widebins(
            self.rgrid, self.npdf_sys.df_pa,
            y_min=y_window[0], y_max=y_window[1], width=width
        )
        return df[(df["pt_left"] >= 0.0) & (df["pt_left"] <= 20.0)].reset_index(drop=True)

    def _eloss_pt_in_cent(self, cent_label: str, y_window: Tuple[float,float],
                          pt_bins: Sequence[Tuple[float,float]]) -> pd.DataFrame:
        rows = []
        for lo, hi in pt_bins:
            c, loe, hie = self._eloss_mean_centrality(cent_label, y_window, (lo, hi))
            rows.append(dict(pt=0.5*(lo+hi), eloss=c, eloss_lo=loe, eloss_hi=hie))
        return pd.DataFrame(rows)

    def _cnm_pt_in_cent(self, cent_label: str, y_window: Tuple[float,float],
                        pt_bins: Sequence[Tuple[float,float]]) -> pd.DataFrame:
        npdf = self._npdf_pt_in_cent(cent_label, y_window, pt_bins)
        el   = self._eloss_pt_in_cent(cent_label, y_window, pt_bins)
        if npdf.empty or el.empty:
            return pd.DataFrame(columns=["pt","c","lo","hi"])
        width = pt_bins[0][1] - pt_bins[0][0]
        centers = npdf["pt_left"].to_numpy() + 0.5*width
        N_c  = np.interp(el["pt"], centers, npdf["r_central"])
        N_lo = np.interp(el["pt"], centers, npdf["r_lo"])
        N_hi = np.interp(el["pt"], centers, npdf["r_hi"])
        Cc, Clo, Chi = combine_product_asym(N_c, N_lo, N_hi, el["eloss"], el["eloss_lo"], el["eloss_hi"])
        return pd.DataFrame(dict(pt=el["pt"], c=Cc, lo=Clo, hi=Chi)).sort_values("pt").reset_index(drop=True)

    def _primordial_pt_in_cent(self, cent_label: str, state: str, y_window: Tuple[float,float],
                               pt_bins: Sequence[Tuple[float,float]]) -> pd.DataFrame:
        cp_center, cp_band = self.ens.central_and_band_vs_pt_per_b(
            y_window=y_window, pt_bins=pt_bins, with_feeddown=True, use_nbin=True
        )
        C = cp_center[["b","pt",state]].rename(columns={state:"c"}).copy()
        if f"{state}_lo" in cp_band.columns and f"{state}_hi" in cp_band.columns:
            B = cp_band[["b","pt",f"{state}_lo",f"{state}_hi"]].rename(
                columns={f"{state}_lo":"lo", f"{state}_hi":"hi"})
        else:
            errc = f"{state}_err"
            B = cp_center[["b","pt",state,errc]].copy()
            B["lo"] = B[state] - B[errc]; B["hi"] = B[state] + B[errc]
            B = B[["b","pt","lo","hi"]]
        M = C.merge(B, on=["b","pt"], how="inner")
        bL, bR = self._b_interval_for_cent(cent_label)
        sub = _nearest_b_rows(M, bL, bR).copy()
        # if sub.empty:
        #     return pd.DataFrame(columns=["pt","c","lo","hi"])
        if sub.empty:
            # fallback: expand slightly then take nearest available b inside [bL,bR]
            expand = 0.05  # fm; small expansion to catch edges
            sub = M[(M["b"] >= bL - expand) & (M["b"] <= bR + expand)]
            if sub.empty:
                # absolute nearest two points (one on each side) to mimic an average
                idx = np.argsort(np.abs(M["b"] - 0.5*(bL + bR)))[:2]
                sub = M.iloc[idx].copy()

        rows = []
        for pv, g in sub.groupby("pt", sort=True):
            c, lo, hi = avg_band_by_weights(g["c"], g["lo"], g["hi"], np.ones(len(g)))
            rows.append(dict(pt=float(pv), c=c, lo=lo, hi=hi))
        return pd.DataFrame(rows).sort_values("pt").reset_index(drop=True)

    def total_vs_pt_in_centrality(self, cent_label: str, state: str,
                                  y_window: Tuple[float,float], pt_bins: Sequence[Tuple[float,float]]):
        cnm  = self._cnm_pt_in_cent(cent_label, y_window, pt_bins)
        prim = self._primordial_pt_in_cent(cent_label, state, y_window, pt_bins)
        if cnm.empty or prim.empty:
            return cnm, prim, pd.DataFrame(columns=["pt","c","lo","hi"])
        p = prim["pt"].to_numpy()
        Cc  = np.interp(p, cnm["pt"], cnm["c"])
        Clo = np.interp(p, cnm["pt"], cnm["lo"])
        Chi = np.interp(p, cnm["pt"], cnm["hi"])
        Tc, Tlo, Thi = combine_product_asym(Cc, Clo, Chi, prim["c"], prim["lo"], prim["hi"])
        return cnm, prim, pd.DataFrame(dict(pt=p, c=Tc, lo=Tlo, hi=Thi))

    # ---------------- centrality-integrated helpers (for summary figs) -------
    def total_vs_y_integrated_over_centrality(self, y_bins, pt_range, state, cent_weights=None):
        import numpy as np
        rows = []
        y_pairs = _as_y_bins(y_bins)   # <— NEW: normalize edges/width → pairs
        weights = cent_weights or {cb: 1.0 for cb in self.ctab["cent_bin"]}

        for (yl, yr) in y_pairs:
            yc = 0.5*(yl + yr)
            acc_c = acc_lo = acc_hi = used = 0.0
            for cb in self.ctab["cent_bin"]:
                w = float(weights.get(cb, 0.0))
                if w <= 0.0: 
                    continue
                cnm  = self._cnm_vs_y_in_cent(cb, y_width=(yr - yl), pt_range=pt_range)
                prim = self._primordial_vs_y_in_cent(cb, state, pt_range, y_width=(yr - yl))
                if cnm.empty or prim.empty:
                    continue
                Cc  = float(np.interp(yc, cnm["y"],  cnm["c"]))
                Clo = float(np.interp(yc, cnm["y"],  cnm["lo"]))
                Chi = float(np.interp(yc, cnm["y"],  cnm["hi"]))
                Pc  = float(np.interp(yc, prim["y"], prim["c"]))
                Plo = float(np.interp(yc, prim["y"], prim["lo"]))
                Phi = float(np.interp(yc, prim["y"], prim["hi"]))
                Tc, Tlo, Thi = combine_product_asym(Cc, Clo, Chi, Pc, Plo, Phi)
                acc_c  += w*Tc; acc_lo += w*Tlo; acc_hi += w*Thi; used += w
            if used > 0.0:
                rows.append(dict(y=yc, c=acc_c/used, lo=acc_lo/used, hi=acc_hi/used))
        return pd.DataFrame(rows).sort_values("y").reset_index(drop=True)



    def plot_total_vs_y_integrated_all_states(
        self,
        states: Sequence[str],
        y_bins: Sequence[Tuple[float, float]],
        pt_range: Tuple[float, float],
        *,
        ylim: Tuple[float, float] = (0.0, 1.5),
        legend_loc: str = "upper right",
        minor_ticks: bool = True,
        note: Optional[str] = None,
        save: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for st in states:
            df = self.total_vs_y_integrated_over_centrality(y_bins, pt_range, st)
            if df.empty:
                continue
            color = STATE_COLORS.get(st, COMP_COLORS.get("Total", "C3"))
            ax.fill_between(df["y"], df["lo"], df["hi"], alpha=BAND_ALPHA["Total"], color=color)
            ax.plot(df["y"], df["c"], lw=2.2, color=color, label=PRETTY_STATE.get(st, st))
        ax.set_xlabel("y")
        ax.set_ylabel(r"$R_{pA}$")
        if minor_ticks: ax.minorticks_on()
        ax.set_ylim(*ylim)
        ax.legend(loc=legend_loc, frameon=False, ncol=min(3, len(states)))
        if note:
            _annotate_corner(ax, note, loc="lower right")
        plt.tight_layout()
        if save:
            plt.savefig(save, bbox_inches="tight")
        plt.show()

    def plot_vs_pt_by_centrality_with_experiment(
        self,
        *,
        y_window: tuple[float, float],
        pt_bins: list[tuple[float, float]],
        state: str = "jpsi_1S",
        components: tuple[str, ...] = ("Total",),   # any of: "nPDF","eLoss","CNM","Primordial","Total"
        exppt: pd.DataFrame | None = None,          # output of prepare_exp_pt_overlay()
        exp_rapidity_label: str | None = None,      # optional override; else auto from y_window
        exp_label: str = "ALICE",
        xlim=(0, 20), ylim=(0.2, 1.6),
        legend_mode: str = "panel-first",
        legend_loc: str = "lower right",
        save_pdf: str | None = None,
    ):
        comps = {c.lower() for c in components}
        cent_bins = [str(x) for x in self.ctab["cent_bin"]]
        n = len(cent_bins)
        ncols = min(3, n); nrows = int(np.ceil(n/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.8*ncols+0.8, 3.3*nrows),
                                sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        # regex for the ALICE rapidity string from y_window unless user passed one
        if exp_rapidity_label is None:
            a, b = y_window
            # e.g. "-4.46 - -2.96" or "2.03 - 3.53"
            sign = lambda x: "-" if x < 0 else ""
            exp_regex = rf"{a:.2f}\s*-\s*{b:.2f}".replace("+","")
        else:
            exp_regex = re.escape(str(exp_rapidity_label))

        for idx, (ax, cb) in enumerate(zip(axes, cent_bins)):
            # --- THEORY ---
            # draw in a fixed order so overlays are crisp
            if "npdf" in comps:
                dfN = self._npdf_pt_in_cent(cb, y_window, pt_bins)
                if not dfN.empty:
                    width = pt_bins[0][1] - pt_bins[0][0]
                    centers = dfN["pt_left"].to_numpy(float) + 0.5*width
                    y, lo, hi = dfN["r_central"], dfN["r_lo"], dfN["r_hi"]
                    ax.fill_between(centers, lo, hi, alpha=0.22, color="#1f77b4", label="nPDF")
                    ax.plot(centers, y, "-", color="#1f77b4")

            if "eloss" in comps:
                dfE = self._eloss_pt_in_cent(cb, y_window, pt_bins)
                if not dfE.empty:
                    ax.plot(dfE["pt"], dfE["eloss"], "s-", color="#ff7f0e", label="eLoss")

            if "cnm" in comps:
                dfC = self._cnm_pt_in_cent(cb, y_window, pt_bins)
                if not dfC.empty:
                    ax.fill_between(dfC["pt"], dfC["lo"], dfC["hi"], alpha=0.22, color="#2ca02c", label="CNM")
                    ax.plot(dfC["pt"], dfC["c"], "-", color="#2ca02c")

            if "primordial" in comps:
                dfP = self._primordial_pt_in_cent(cb, state, y_window, pt_bins)
                if not dfP.empty:
                    ax.fill_between(dfP["pt"], dfP["lo"], dfP["hi"], alpha=0.22, color="#9467bd",
                                    label=f"Primordial {PRETTY_STATE.get(state,state)}")
                    ax.plot(dfP["pt"], dfP["c"], "-", color="#9467bd")

            if "total" in comps:
                _, _, dfT = self.total_vs_pt_in_centrality(cb, state, y_window, pt_bins)
                if not dfT.empty:
                    col = STATE_COLORS.get(state, "#d62728")
                    ax.fill_between(dfT["pt"], dfT["lo"], dfT["hi"], alpha=0.28, color=col,
                                    label=f"CNM × Primordial")
                    ax.plot(dfT["pt"], dfT["c"], "-", color=col)

            # --- EXPERIMENTAL OVERLAY (matched centrality + rapidity) ---
            if exppt is not None and not exppt.empty:
                g = exppt[
                    (exppt["cent_match"].astype(str) == str(cb)) &
                    (exppt["rapidity"].astype(str).str.contains(exp_regex))
                ].reset_index(drop=True)
                if not g.empty:
                    x  = g["pt"].to_numpy(float)
                    xE = np.vstack([x - g["ptlo"].to_numpy(float), g["pthi"].to_numpy(float) - x])
                    y  = g["val"].to_numpy(float)
                    yE = g["dtot"].to_numpy(float)
                    ax.errorbar(x, y, xerr=xE, yerr=yE, fmt="o", ms=5, mfc="black", mec="black",
                                ecolor="black", color="black", capsize=2.5, lw=1.1,
                                label=f"{exp_label} ({cb})", zorder=10)

            # cosmetics
            ax.set_title(str(cb))
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_xlabel(r"$p_T$ [GeV]"); 
            if idx % ncols == 0: ax.set_ylabel(r"$R_{pA}$")
            ax.minorticks_on()
            ax.text(0.02, 0.96, rf"{y_window[0]:.2f} < $y_{{\rm cms}}$ < {y_window[1]:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10)

            # legend only on the first visible panel
            if legend_mode == "panel-first":
                if idx == 0:
                    ax.legend(frameon=False, loc=legend_loc)
            elif legend_mode == "each":
                ax.legend(frameon=False, loc=legend_loc)

        # hide empty cells
        for ax in axes[len(cent_bins):]:
            ax.set_visible(False)

        if legend_mode == "figure":
            handles, labels = axes[0].get_legend_handles_labels()
            if labels:
                fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

        if save_pdf:
            plt.savefig(save_pdf, bbox_inches="tight", dpi=300)
        plt.show()

    def total_vs_pt_integrated_over_centrality(
        self,
        y_window: Tuple[float, float],
        pt_bins: Sequence[Tuple[float, float]],
        state: str,
        cent_weights: Optional[Dict[str, float]] = None,
        *,
        # --- new, optional plotting knobs (all default = no behavior change) ---
        plot: bool = False,
        ax=None,
        legend_loc: str = "best",
        note_loc: str = "lower right",
        ylim: Optional[Tuple[float, float]] = None,
        xlim: Optional[Tuple[float, float]] = None,
        minor_ticks: bool = True,
        show: bool = True,
        save: Optional[str] = None,
    ):
        """
        Centrality-integrated Total × <state> vs pT.
        Returns a DataFrame with columns [pt, c, lo, hi].

        If plot=True, also draws a publication-ready band+curve:
        - legend placed per `legend_loc` (inside the axes)
        - a small y-window note placed at `note_loc`
        - optional x/y limits via `xlim` / `ylim`
        """
        rows = []
        weights = cent_weights or {cb: 1.0 for cb in self.ctab["cent_bin"]}
        # Use the sum of *used* weights to avoid bias if some bins are empty for this (y, pT)
        for (pl, pr) in pt_bins:
            pc = 0.5 * (pl + pr)
            acc_c = acc_lo = acc_hi = 0.0
            used = 0.0
            for cb in self.ctab["cent_bin"]:
                w = float(weights.get(cb, 0.0))
                if w <= 0.0:
                    continue
                _, _, tot = self.total_vs_pt_in_centrality(cb, state, y_window, [(pl, pr)])
                if tot.empty:
                    continue
                Tc, Tlo, Thi = tot["c"].iloc[0], tot["lo"].iloc[0], tot["hi"].iloc[0]
                acc_c += w * Tc
                acc_lo += w * Tlo
                acc_hi += w * Thi
                used += w
            if used > 0.0:
                rows.append(dict(pt=pc, c=acc_c / used, lo=acc_lo / used, hi=acc_hi / used))

        df = pd.DataFrame(rows)

        # --- Optional, minimal plotting (no change to return type) ---
        if plot and not df.empty:
            import matplotlib.pyplot as plt

            if ax is None:
                fig, ax = plt.subplots(figsize=(6.4, 4.2))
            # band + central curve
            ax.fill_between(
                df["pt"], df["lo"], df["hi"],
                alpha=BAND_ALPHA.get("Total", 0.28),
                color=STATE_COLORS.get(state, COMP_COLORS.get("Total", "C3")),
                label=f"CNM X Primordial"
            )
            ax.plot(
                df["pt"], df["c"],
                lw=2.2,
                color=STATE_COLORS.get(state, COMP_COLORS.get("Total", "C3"))
            )

            # axes style
            ax.set_xlabel(r"$p_T$ [GeV]")
            ax.set_ylabel(r"$R_{pA}$")
            if minor_ticks:
                ax.minorticks_on()
            if xlim:
                ax.set_xlim(*xlim)
            if ylim:
                ax.set_ylim(*ylim)

            # y-window note inside the axes
            y_note = f"{y_window[0]:.2f} < y < {y_window[1]:.2f}"
            _annotate_corner(ax, y_note, loc=note_loc)

            # legend inside the axes
            ax.legend(loc=legend_loc, frameon=False)

            if save:
                plt.tight_layout()
                plt.savefig(save, bbox_inches="tight")
            if show:
                plt.tight_layout()
                plt.show()

        return df

    def plot_total_vs_pt_integrated_subfigs_all_states(
        self,
        states: Sequence[str],
        y_windows: Sequence[Tuple[float, float]],
        pt_bins: Sequence[Tuple[float, float]],
        *,
        ncols: int = 3,
        xlim: Tuple[float, float] = (0.0, 20.0),
        ylim: Tuple[float, float] = (0.0, 1.5),
        legend_loc: str = "best",
        minor_ticks: bool = True,
        save: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt
        n = len(y_windows)
        ncols = min(max(1, ncols), n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.6 * nrows), sharex=True, sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        for i, (ax, ywin) in enumerate(zip(axes, y_windows)):
            for st in states:
                df = self.total_vs_pt_integrated_over_centrality(ywin, pt_bins, st, plot=False)
                if df.empty:
                    continue
                color = STATE_COLORS.get(st, COMP_COLORS.get("Total", "C3"))
                ax.fill_between(df["pt"], df["lo"], df["hi"], alpha=BAND_ALPHA["Total"], color=color)
                ax.plot(df["pt"], df["c"], lw=2.2, color=color, label=PRETTY_STATE.get(st, st))
            ax.set_title(f"{ywin[0]:.1f} < y < {ywin[1]:.1f}")
            ax.set_xlabel(r"$p_T$ [GeV]")
            ax.set_ylabel(r"$R_{pA}$")
            if minor_ticks: ax.minorticks_on()
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.legend(loc=legend_loc, frameon=False, ncol=min(3, len(states)))

        for j in range(len(y_windows), len(axes)):
            axes[j].axis("off")

        if save:
            plt.savefig(save, bbox_inches="tight")
        plt.show()


    # --- Minimal, backwards-compatible experimental overlay utilities ----

    @staticmethod
    # --------------------- experimental overlay helpers (minimal) ---------------------

    def make_exp_df_binned(
        *,
        y_center: Sequence[float],
        y_halfwidth: Sequence[float],
        value: Sequence[float],
        stat: Sequence[float],
        syst: Sequence[float],
        label: Optional[str] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a tidy experimental table for RpA vs y with symmetric stat/syst errors.
        Returns columns:
        y, ylo, yhi, val, dstat, dsyst, dtot, label, source
        """
        yc  = np.asarray(y_center,    float)
        dy  = np.asarray(y_halfwidth, float)
        val = np.asarray(value,       float)
        ds  = np.asarray(stat,        float)
        sy  = np.asarray(syst,        float)
        dt  = np.array([_quadrature(a, b) for a, b in zip(ds, sy)], float)

        df = pd.DataFrame(dict(
            y=yc,
            ylo=yc - dy,
            yhi=yc + dy,
            val=val,
            dstat=ds,
            dsyst=sy,
            dtot=dt,
        ))
        if label is not None:  df["label"]  = str(label)
        if source is not None: df["source"] = str(source)
        return df

    # ---- Minimal experimental overlays (keeps your numerics untouched) ----

    def overlay_experiment_vs_y(ax, df, *, color=None, marker="o", filled=True,
                                capsize=3.0, label=None, legend_loc=None, zorder=10):
        """
        Draw RpA(y) points with horizontal/vertical errors from experimental_data.make_df_y().
        Columns used: y, ylo, yhi, val, dtot; optional label shown if legend_loc is given.
        """
        import numpy as np
        if df is None or df.empty: 
            return ax
        x    = df["y"].to_numpy(float)
        y    = df["val"].to_numpy(float)
        xerr = np.vstack([x - df["ylo"].to_numpy(float), df["yhi"].to_numpy(float) - x])
        yerr = df["dtot"].to_numpy(float)
        mfc  = color if filled else "none"
        mec  = color
        h = ax.errorbar(
            x, y, yerr=yerr, xerr=xerr, fmt=marker, ms=6.5,
            mfc=mfc, mec=mec, ecolor=color, color=color, capsize=capsize, lw=1.6,
            label=(label if legend_loc is not None else None), zorder=zorder
        )
        if legend_loc is not None and label:
            ax.legend(loc=legend_loc, frameon=False)
        return h

    def overlay_experiment_vs_pt(ax, df, *, color=None, marker="o", filled=True,
                                capsize=3.0, label=None, legend_loc=None, zorder=10):
        """
        Draw RpA(pT) points with horizontal/vertical errors from experimental_data.make_df_pt().
        Columns used: pt, ptlo, pthi, val, dtot.
        """
        import numpy as np
        if df is None or df.empty:
            return ax
        x    = df["pt"].to_numpy(float)
        y    = df["val"].to_numpy(float)
        xerr = np.vstack([x - df["ptlo"].to_numpy(float), df["pthi"].to_numpy(float) - x])
        yerr = df["dtot"].to_numpy(float)
        mfc  = color if filled else "none"
        mec  = color
        h = ax.errorbar(
            x, y, yerr=yerr, xerr=xerr, fmt=marker, ms=6.5,
            mfc=mfc, mec=mec, ecolor=color, color=color, capsize=capsize, lw=1.6,
            label=(label if legend_loc is not None else None), zorder=zorder
        )
        if legend_loc is not None and label:
            ax.legend(loc=legend_loc, frameon=False)
        return h

    def total_vs_Npart(self, y_window: tuple[float,float], pt_range: tuple[float,float], state: str):
        """Return Total RpA vs <N_part> using the Glauber centrality table."""
        rows = []
        for _, row in self.ctab.iterrows():
            cb     = str(row["cent_bin"])
            Npart  = float(row["N_part"])  # from Glauber
            # reuse your existing per-centrality samplers
            cnm   = self._cnm_vs_centrality(y_window, pt_range)   # or self._cnm_in_cent(cb, ...)
            prim  = self._primordial_vs_centrality(y_window, pt_range, state)
            # each returns a table across cent bins; select the current bin
            cnm_cb  = cnm[cnm["cent_bin"]==cb]
            prim_cb = prim[prim["cent_bin"]==cb]
            if cnm_cb.empty or prim_cb.empty:
                continue
            Cc, Clo, Chi = [float(cnm_cb[k].values[0]) for k in ("c","lo","hi")]
            Pc, Plo, Phi = [float(prim_cb[k].values[0]) for k in ("c","lo","hi")]
            Tc, Tlo, Thi = combine_product_asym(Cc, Clo, Chi, Pc, Plo, Phi)
            rows.append(dict(N_part=Npart, c=Tc, lo=Tlo, hi=Thi, cent_bin=cb))
        return pd.DataFrame(rows).sort_values("N_part").reset_index(drop=True)
    

