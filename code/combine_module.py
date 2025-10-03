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
CENT_EDGES_ELOSS = [0, 20, 40, 60, 100]
Y_WINDOWS_3      = [(-1.93, 1.93), (1.5, 4.0), (-5.0, -2.5)]
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

def _parse_centbin(s: str) -> Tuple[float,float]:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*%", str(s))
    if not m: return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))


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

    def _b_interval_for_cent(self, label: str) -> Tuple[float,float]:
        row = self.ctab[self.ctab["cent_bin"]==label].iloc[0]
        return float(row["b_left"]), float(row["b_right"])

    def _eloss_mean_centrality(self, cent_label: str, y_range: Tuple[float,float], pt_range: Tuple[float,float]):
        tag = cent_label.replace("%","")
        val = self.eloss.mean_rpa_over_y_and_pt(tag, y_range, pt_range)
        try:
            err = self.eloss.mean_rpa_err_over_y_and_pt(tag, y_range, pt_range)
            return (val, max(val-err, 0.0), val+err)
        except Exception:
            return (val, val, val)

    # ----------------- CNM vs centrality (tables) -----------------
    def cnm_vs_centrality(self, y_range: Tuple[float,float], pt_range: Tuple[float,float]) -> pd.DataFrame:
        tab_np = self.model.rpa_vs_centrality_integrated(
            self.rgrid, self.npdf_sys.df_pa, CENT_EDGES_ELOSS,
            y_min=y_range[0], y_max=y_range[1],
            pt_min=pt_range[0], pt_max=pt_range[1],
            sigmaNN_mb=self.sigmaNN_mb, weight="inelastic", verbose=False
        ).copy()
        if tab_np.empty:
            return pd.DataFrame(columns=["cent_bin","r_central","r_lo","r_hi","eloss","cnm_c","cnm_lo","cnm_hi"])
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
        out["eloss"]  = e_c
        out["cnm_c"]  = Cc
        out["cnm_lo"] = Clo
        out["cnm_hi"] = Chi
        return out

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
            sub = M[(M["b"]>=bL)&(M["b"]<=bR)]
            if sub.empty:
                rows.append(dict(cent_bin=label, c=np.nan, lo=np.nan, hi=np.nan))
            else:
                c, lo, hi = avg_band_by_weights(sub["c"], sub["lo"], sub["hi"], np.ones(len(sub)))
                rows.append(dict(cent_bin=label, c=c, lo=lo, hi=hi))
        return pd.DataFrame(rows)

    def total_vs_centrality(self, y_range: Tuple[float,float], pt_range: Tuple[float,float], state: str) -> pd.DataFrame:
        cnm = self.cnm_vs_centrality(y_range, pt_range)
        prim = self.primordial_vs_centrality(pt_range, y_range, state)
        if cnm.empty or prim.empty:
            return pd.DataFrame(columns=["cent_bin","cnm_c","cnm_lo","cnm_hi","c","lo","hi","total_c","total_lo","total_hi"])
        out = cnm.merge(prim, on="cent_bin", how="inner", suffixes=("_cnm","_prim"))
        Tc, Tlo, Thi = combine_product_asym(
            out["cnm_c"].to_numpy(), out["cnm_lo"].to_numpy(), out["cnm_hi"].to_numpy(),
            out["c"].to_numpy(),     out["lo"].to_numpy(),     out["hi"].to_numpy()
        )
        out["total_c"]  = Tc
        out["total_lo"] = Tlo
        out["total_hi"] = Thi
        return out

    # ============================= Figures =============================
    # ---- 0) One flexible centrality plotter (choose what to show)
    def figure_result_vs_centrality(
        self,
        y_windows: List[Tuple[float,float]],
        pt_range: Tuple[float,float],
        components: Sequence[str] = ("nPDF", "eLoss", "CNM"),  # allow "Primordial[:state]", "Total[:state]"
        states: Optional[List[str]] = None,
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
    def _cnm_vs_y_in_cent(self, cent_label: str, y_width: float, pt_range: Tuple[float,float]) -> pd.DataFrame:
        bands = self.model.rpa_vs_y_in_centrality_bins(
            self.rgrid, self.npdf_sys.df_pa, CENT_EDGES_ELOSS,
            y_width=y_width, pt_min=pt_range[0],
            sigmaNN_mb=self.sigmaNN_mb, weight="inelastic", verbose=False
        )
        dfN = dict(bands).get(cent_label)
        if dfN is None or dfN.empty:
            return pd.DataFrame(columns=["y","c","lo","hi"])
        y_left = dfN["y_left"].to_numpy(float)
        dy = (y_left[1] - y_left[0]) if len(y_left) > 1 else y_width
        y_edges = np.r_[y_left, y_left[-1] + dy]
        y_centers = 0.5*(y_edges[:-1] + y_edges[1:])

        E_vals, E_los, E_his = [], [], []
        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            c, lo, hi = self._eloss_mean_centrality(cent_label, (yl, yr), pt_range)
            E_vals.append(c); E_los.append(lo); E_his.append(hi)

        Cc, Clo, Chi = combine_product_asym(
            dfN["r_central"], dfN["r_lo"], dfN["r_hi"],
            np.array(E_vals), np.array(E_los), np.array(E_his)
        )
        return pd.DataFrame(dict(y=y_centers, c=Cc, lo=Clo, hi=Chi))

    def _primordial_vs_y_in_cent(self, cent_label: str, state: str, pt_range: Tuple[float,float], y_width: float=0.5) -> pd.DataFrame:
        y_bins = make_bins_from_width(-5, 5, y_width)
        ryb_center, ryb_band = self.ens.central_and_band_vs_y_per_b(
            pt_window=pt_range, y_bins=y_bins, with_feeddown=True, use_nbin=True, flip_y=True
        )
        C = ryb_center[["b","y",state]].rename(columns={state:"c"}).copy()
        if f"{state}_lo" in ryb_band.columns and f"{state}_hi" in ryb_band.columns:
            B = ryb_band[["b","y",f"{state}_lo",f"{state}_hi"]].rename(columns={f"{state}_lo":"lo", f"{state}_hi":"hi"})
        else:
            errc = f"{state}_err"
            B = ryb_center[["b","y",state,errc]].copy()
            B["lo"] = B[state] - B[errc]; B["hi"] = B[state] + B[errc]
            B = B[["b","y","lo","hi"]]
        M = C.merge(B, on=["b","y"], how="inner")
        bL,bR = self._b_interval_for_cent(cent_label)
        sub = M[(M["b"]>=bL)&(M["b"]<=bR)].copy()
        if sub.empty: return pd.DataFrame(columns=["y","c","lo","hi"])
        rows=[]
        for yv,g in sub.groupby("y", sort=True):
            c, lo, hi = avg_band_by_weights(g["c"], g["lo"], g["hi"], np.ones(len(g)))
            rows.append(dict(y=float(yv), c=c, lo=lo, hi=hi))
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
            
    # ---- 4) Total vs pT — flexible # of y-windows & layout (no IndexError anymore)
    
    def figure_total_vs_pt_per_centrality(
        self, y_windows, pt_bins, states,
        save_pdf_prefix: Optional[str]=None, legend_mode="panel-first", legend_loc="best",
        ylim=(0.0,1.5), xlim=(0.0,20.0), ncols_per_cent: Optional[int]=None,
        minor_ticks: bool=True, note_loc: str="lower right"
    ):
        labs = list(self.ctab["cent_bin"])
        ncent = len(labs)
        ny = len(y_windows)
        order = list(range(ny))  # natural order; if you want the (mid,back,forward) order, reorder y_windows upstream

        # panels: ncent * ny
        ncols = int(ncols_per_cent or ny) * ncent
        fig, axes = plt.subplots(1, ncent * ny, figsize=(4.2*ncent*ny, 4.6), sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)

        panel = 0
        for ic, lab in enumerate(labs):
            for jj, idx in enumerate(order):
                ax = axes[panel]; panel += 1
                ywin = y_windows[idx]
                # Build cnm/prim/total at this centrality & y-window
                for state in states[:1]:  # legend content is identical if multiple states; label uses this color
                    cnm, prim, tot = self.total_vs_pt_in_centrality(lab, state, ywin, pt_bins)
                    if not cnm.empty:
                        _band_xy(ax, cnm["pt"], cnm["c"], cnm["lo"], cnm["hi"], label="CNM", color=COMP_COLORS["CNM"])
                    if not prim.empty:
                        _band_xy(ax, prim["pt"], prim["c"], prim["lo"], prim["hi"], label=f"Primordial {PRETTY_STATE.get(state, state)}", color=COMP_COLORS["Primordial"])
                    if not tot.empty:
                        _band_xy(ax, tot["pt"], tot["c"], tot["lo"], tot["hi"], label=f"CNM X Primordial {PRETTY_STATE.get(state, state)}",
                                 color=STATE_COLORS.get(state, COMP_COLORS["Total"]))
                note = f"{ywin[0]:.1f}<y<{ywin[1]:.1f}"
                _style_axes(ax, r"$p_T$ [GeV]", r"$R_{pA}$", title=f"{lab} — {note}", minor_ticks=minor_ticks)
                if xlim: ax.set_xlim(*xlim)
                if ylim: ax.set_ylim(*ylim)

                _legend_apply(ax, *ax.get_legend_handles_labels(),
                              legend_mode=legend_mode, legend_loc=legend_loc,
                              panel_index=(panel-1), grid_first_panel=0)

        if save_pdf_prefix:
            fig.savefig(f"{save_pdf_prefix}_{state}.pdf", bbox_inches="tight")
        plt.show()

    # ---- pT figures and helpers --------------------------------------------
    def _npdf_pt_in_cent(self, cent_label: str, y_window: Tuple[float,float], pt_bins: Sequence[Tuple[float,float]]) -> pd.DataFrame:
        width = pt_bins[0][1]-pt_bins[0][0]
        df = self.ana.rpa_vs_pt_widebins(self.rgrid, self.npdf_sys.df_pa, y_min=y_window[0], y_max=y_window[1], width=width)
        return df[(df["pt_left"]>=0.0) & (df["pt_left"]<=20.0)].reset_index(drop=True)

    def _eloss_pt_in_cent(self, cent_label: str, y_window: Tuple[float,float], pt_bins: Sequence[Tuple[float,float]]) -> pd.DataFrame:
        rows=[]
        for lo,hi in pt_bins:
            c, loe, hie = self._eloss_mean_centrality(cent_label, y_window, (lo,hi))
            rows.append(dict(pt=0.5*(lo+hi), eloss=c, eloss_lo=loe, eloss_hi=hie))
        return pd.DataFrame(rows)

    def _cnm_pt_in_cent(self, cent_label: str, y_window: Tuple[float,float], pt_bins: Sequence[Tuple[float,float]]) -> pd.DataFrame:
        npdf = self._npdf_pt_in_cent(cent_label, y_window, pt_bins)
        el   = self._eloss_pt_in_cent(cent_label, y_window, pt_bins)
        if npdf.empty or el.empty:
            return pd.DataFrame(columns=["pt","c","lo","hi"])
        width = pt_bins[0][1]-pt_bins[0][0]
        centers = npdf["pt_left"].to_numpy() + 0.5*width
        N_c  = np.interp(el["pt"], centers, npdf["r_central"])
        N_lo = np.interp(el["pt"], centers, npdf["r_lo"])
        N_hi = np.interp(el["pt"], centers, npdf["r_hi"])
        Cc, Clo, Chi = combine_product_asym(N_c, N_lo, N_hi, el["eloss"], el["eloss_lo"], el["eloss_hi"])
        return pd.DataFrame(dict(pt=el["pt"], c=Cc, lo=Clo, hi=Chi)).sort_values("pt").reset_index(drop=True)

    def _primordial_pt_in_cent(self, cent_label: str, state: str, y_window: Tuple[float,float], pt_bins: Sequence[Tuple[float,float]]) -> pd.DataFrame:
        cp_center, cp_band = self.ens.central_and_band_vs_pt_per_b(
            y_window=y_window, pt_bins=pt_bins, with_feeddown=True, use_nbin=True
        )
        C = cp_center[["b","pt",state]].rename(columns={state:"c"}).copy()
        if f"{state}_lo" in cp_band.columns and f"{state}_hi" in cp_band.columns:
            B = cp_band[["b","pt",f"{state}_lo",f"{state}_hi"]].rename(columns={f"{state}_lo":"lo", f"{state}_hi":"hi"})
        else:
            errc = f"{state}_err"
            B = cp_center[["b","pt",state,errc]].copy()
            B["lo"] = B[state]-B[errc]; B["hi"]=B[state]+B[errc]
            B = B[["b","pt","lo","hi"]]
        M = C.merge(B, on=["b","pt"], how="inner")
        bL,bR = self._b_interval_for_cent(cent_label)
        sub = M[(M["b"]>=bL)&(M["b"]<=bR)].copy()
        if sub.empty: return pd.DataFrame(columns=["pt","c","lo","hi"])
        rows=[]
        for pv,g in sub.groupby("pt", sort=True):
            c, lo, hi = avg_band_by_weights(g["c"], g["lo"], g["hi"], np.ones(len(g)))
            rows.append(dict(pt=float(pv), c=c, lo=lo, hi=hi))
        return pd.DataFrame(rows).sort_values("pt").reset_index(drop=True)

    def total_vs_pt_in_centrality(self, cent_label: str, state: str, y_window: Tuple[float,float], pt_bins: Sequence[Tuple[float,float]]):
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
    def total_vs_y_integrated_over_centrality(self, y_bins: Sequence[Tuple[float,float]], pt_range: Tuple[float,float], state: str,
                                              cent_weights: Optional[Dict[str,float]] = None) -> pd.DataFrame:
        rows=[]
        weights = cent_weights or {cb:1.0 for cb in self.ctab["cent_bin"]}
        W = sum(weights.values()) if weights else 1.0
        for (yl, yr) in y_bins:
            yc = 0.5*(yl+yr)
            acc_c=acc_lo=acc_hi=0.0; used=0.0
            for cb in self.ctab["cent_bin"]:
                w = weights.get(cb, 0.0)
                cnm = self._cnm_vs_y_in_cent(cb, y_width=yr-yl, pt_range=pt_range)
                prim = self._primordial_vs_y_in_cent(cb, state, pt_range, y_width=yr-yl)
                if cnm.empty or prim.empty: 
                    continue
                Cc  = np.interp(yc, cnm["y"], cnm["c"])
                Clo = np.interp(yc, cnm["y"], cnm["lo"])
                Chi = np.interp(yc, cnm["y"], cnm["hi"])
                Pc  = np.interp(yc, prim["y"], prim["c"])
                Plo = np.interp(yc, prim["y"], prim["lo"])
                Phi = np.interp(yc, prim["y"], prim["hi"])
                Tc, Tlo, Thi = combine_product_asym(Cc, Clo, Chi, Pc, Plo, Phi)
                acc_c  += w*Tc; acc_lo += w*Tlo; acc_hi += w*Thi; used += w
            if used>0:
                rows.append(dict(y=yc, c=acc_c/used, lo=acc_lo/used, hi=acc_hi/used))
        return pd.DataFrame(rows)

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

