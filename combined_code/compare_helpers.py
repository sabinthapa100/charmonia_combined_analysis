
# compare_helpers.py — minimal-change, fast overlays for NPWLC vs Pert
# Drop this file next to your notebook (or add its folder to sys.path), then:
#   from compare_helpers import *
#
# Depends on your existing combine_module.py (no changes to it).

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Iterable, List, Sequence, Tuple
from matplotlib.patches import Patch
from combine_module import PRETTY_STATE, STATE_COLORS, BAND_ALPHA, combine_product_asym

MODEL_STYLES = {
    "NPWLC": dict(color="#8a9cff", band_alpha=0.25, ls="-",  z=3, marker="o"),
    "Pert":  dict(color="#ff9c6e", band_alpha=0.25, ls="--", z=2, marker="s", hatch="///"),
}

# ---------------------------------------------------------------------
# Tolerant centrality-bin parser (handles '0-20', '0-20%', even '0-20%%')
# ---------------------------------------------------------------------
import re
_cent_pat = re.compile(r"^\s*([0-9]+(?:\.\d+)?)\s*[-–]\s*([0-9]+(?:\.\d+)?)\s*%*\s*$")

def _band(ax, x, lo, hi, *, color, label="", hatch=None, ls="-", lw=2.2, alpha=0.22, z=2, marker=None):
    if hatch:
        # hatched band: draw unfilled hatch with edgecolor, no facealpha
        ax.fill_between(x, lo, hi, facecolor="none", edgecolor=color, hatch=hatch, linewidth=0.0, zorder=z)
        ax.plot(x, 0.5*(lo+hi), ls, lw=lw, color=color, zorder=z+0.1, marker=marker, label=label)
    else:
        # regular filled band
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, zorder=z)
        ax.plot(x, 0.5*(lo+hi), ls, lw=lw, color=color, zorder=z+0.1, marker=marker, label=label)

def _as_y_bins(y_bins):
    import numpy as np
    if isinstance(y_bins, (int, float)):
        w = float(y_bins)
        edges = np.arange(-5.0, 5.0 + 1e-12, w)
        return list(zip(edges[:-1], edges[1:]))
    y_bins = list(y_bins)
    if y_bins and isinstance(y_bins[0], (int, float)):
        edges = np.asarray(y_bins, float)
        return list(zip(edges[:-1], edges[1:]))
    return y_bins

def _canon_cent(s) -> str:
    """Normalize centrality labels into 'lo-hi%' strings."""
    if hasattr(s, "left") and hasattr(s, "right"):
        lo, hi = float(s.left), float(s.right)
    else:
        s = str(s).replace("–","-").replace("%","").strip()
        m = _cent_pat.match(s)
        if not m: raise ValueError(f"Bad cent_bin '{s}'")
        lo, hi = float(m.group(1)), float(m.group(2))
    return f"{lo:.0f}-{hi:.0f}%"

def _ncoll_for_bins(C, cent_bins):
    """
    Return one N_coll per cent_bin (same length, same order).
    If a label can't be matched, return np.nan (we'll mask it when drawing).
    """
    g = _df_norm_cent(C.ctab.copy(), col="cent_bin")
    if "N_coll" in g.columns:
        g["__N__"] = g["N_coll"].astype(float)
    elif "N_part" in g.columns:
        g["__N__"] = g["N_part"].astype(float) - 1.0
    else:
        raise KeyError("Neither N_coll nor N_part found in ctab")

    m = { _canon_cent(cb): float(nc) for cb, nc in zip(g["cent_bin"], g["__N__"]) }
    # keep same length/order; use NaN if a key is missing
    return np.array([m.get(_canon_cent(cb), np.nan) for cb in cent_bins], float)


def _df_norm_cent(df, col="cent_bin"):
    df = df.copy()
    df[col] = [ _canon_cent(s) for s in df[col] ]
    return df

def _cent_label_note(yw):
    return rf"{yw[0]:.2f} < $y_{{\rm cms}}$ < {yw[1]:.2f}"

def _pick(df, names):  # small convenience
    for n in names:
        if n in df.columns: return n
    return None

def _x_from_ctab(C):
    if "N_coll" in C.ctab.columns:
        return C.ctab["N_coll"].astype(float).to_numpy()
    # p+Pb identity if only N_part is provided
    if "N_part" in C.ctab.columns:
        return (C.ctab["N_part"].astype(float) - 1.0).to_numpy()
    raise KeyError("Neither N_coll nor N_part found in ctab")

def cent_centers(cent_bin_series):
    xs = []
    for s in cent_bin_series:
        # accept pandas.Interval or string labels
        if hasattr(s, "left") and hasattr(s, "right"):
            lo, hi = float(s.left), float(s.right)
        else:
            s = str(s).strip().replace("–", "-").replace("%", "")
            m = _cent_pat.match(s)
            if not m:
                raise ValueError(f"Bad cent_bin '{s}'")
            lo, hi = float(m.group(1)), float(m.group(2))
        xs.append(0.5*(lo + hi))
    xs = np.asarray(xs, float)
    # auto-scale if the bins are in fractions (0..1)
    if xs.size and np.nanmax(xs) <= 1.5:
        xs = xs * 100.0
    return xs

def _legend_if_any(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    if len([x for x in l if x and not x.startswith("_")]) > 0:
        ax.legend(**kw)

# -------------------- internal cache to speed up y-curves ---------------------
# Key off object identity; safe because these are per-session notebooks.
_Y_CACHE_CNM  : Dict[Tuple[int,str,float,Tuple[float,float]], pd.DataFrame] = {}
_Y_CACHE_PRIM : Dict[Tuple[int,str,str,float,Tuple[float,float]], pd.DataFrame] = {}

_YC = {}
def _get_cnm_vs_y(C, cb, y_width, pt_range):
    key = ("cnm", id(C), cb, float(y_width), float(pt_range[0]), float(pt_range[1]))
    df = _YC.get(key)
    if df is None:
        df = C._cnm_vs_y_in_cent(cb, y_width=float(y_width), pt_range=pt_range)
        _YC[key] = df
    return df

def _get_prim_vs_y(C, cb, state, y_width, pt_range):
    key = ("prim", id(C), cb, state, float(y_width), float(pt_range[0]), float(pt_range[1]))
    df = _YC.get(key)
    if df is None:
        df = C._primordial_vs_y_in_cent(cb, state, pt_range, y_width=float(y_width))
        _YC[key] = df
    return df

# ---------------------- fast Total(y) in one centrality -----------------------
def total_vs_y_in_centrality_fast(C, cb, state, y_bins, pt_range):
    if not y_bins:
        return pd.DataFrame(columns=["y","c","lo","hi"])
    width = float(np.unique([round(b[1]-b[0],6) for b in y_bins])[0])
    yc = np.asarray([0.5*(a+b) for (a,b) in y_bins], float)

    cnm  = C._cnm_vs_y_in_cent(cb, width, pt_range)
    prim = C._primordial_vs_y_in_cent(cb, state, pt_range, width)
    if cnm.empty or prim.empty:
        return pd.DataFrame(columns=["y","c","lo","hi"])

    def sample(df):
        y  = df["y"].to_numpy(float)
        c  = np.interp(yc, y, df["c"].to_numpy(float))
        lo = np.interp(yc, y, df["lo"].to_numpy(float))
        hi = np.interp(yc, y, df["hi"].to_numpy(float))
        return c, lo, hi

    Cc,Clo,Chi = sample(cnm); Pc,Plo,Phi = sample(prim)
    Tc,Tlo,Thi = combine_product_asym(Cc,Clo,Chi, Pc,Plo,Phi)
    return pd.DataFrame({"y": yc, "c": Tc, "lo": Tlo, "hi": Thi})

def _component_vs_Ncoll(C, comp: str, y_window, pt_range, state):
    """
    Return DataFrame with columns: ['N_coll','c','lo','hi'] for the requested component.
    Component keys (case-insensitive): 'npdf','eloss','cnm','primordial','total'.
    """
    comp = comp.lower()

    # helpers that return cent_bin + c,lo,hi
    def _cnm_table():
        t = C.cnm_vs_centrality(y_window, pt_range).copy()
        print(t)
        if t.empty: return t
        t = _df_norm_cent(t)
        # normalize column names
        c  = _pick(t, ["cnm_c","r_central","r_center","r_c","npdf_c"]) or "cnm_c"
        lo = _pick(t, ["cnm_lo","r_lo","npdf_lo","lo"]) or "cnm_lo"
        hi = _pick(t, ["cnm_hi","r_hi","npdf_hi","hi"]) or "cnm_hi"
        out = t[["cent_bin", c, lo, hi]].rename(columns={c:"c", lo:"lo", hi:"hi"})
        return out

    def _npdf_table():
        t = C.cnm_vs_centrality(y_window, pt_range).copy()
        print(t)
        if t.empty: return t
        t = _df_norm_cent(t)
        c  = _pick(t, ["r_central","r_center","r_c","npdf_c"])
        lo = _pick(t, ["r_lo","npdf_lo","lo"])
        hi = _pick(t, ["r_hi","npdf_hi","hi"])
        if not (c and lo and hi): return pd.DataFrame(columns=["cent_bin","c","lo","hi"])
        return t[["cent_bin", c, lo, hi]].rename(columns={c:"c", lo:"lo", hi:"hi"})

    def _eloss_table():
        t = C.cnm_vs_centrality(y_window, pt_range).copy()
        print(t)
        if t.empty or ("eloss" not in t.columns): 
            return pd.DataFrame(columns=["cent_bin","c","lo","hi"])
        t = _df_norm_cent(t)
        c  = "eloss"
        lo = _pick(t, ["eloss_lo","lo"]) or c
        hi = _pick(t, ["eloss_hi","hi"]) or c
        return t[["cent_bin", c, lo, hi]].rename(columns={c:"c", lo:"lo", hi:"hi"})

    def _prim_table():
        t = C.primordial_vs_centrality(pt_range, y_window, state).copy()
        if t.empty: return t
        t = _df_norm_cent(t)
        return t[["cent_bin","c","lo","hi"]]

    def _total_table():
        if hasattr(C, "total_vs_Ncoll"):
            t = C.total_vs_Ncoll(y_window, pt_range, state).copy()
            if t.empty: return t
            # tolerate either (cent_bin + c/lo/hi [+ N_coll]) or the names used by total_vs_centrality
            t = _df_norm_cent(t)
            c  = _pick(t, ["c","total_c"])
            lo = _pick(t, ["lo","total_lo"])
            hi = _pick(t, ["hi","total_hi"])
            keep = ["cent_bin", c, lo, hi] + (["N_coll"] if "N_coll" in t.columns else [])
            t = t[keep].rename(columns={c:"c", lo:"lo", hi:"hi"})
            return t
        # fallback: CNM × Primordial
        cnm = _cnm_table()
        prim = _prim_table()
        if cnm.empty or prim.empty:
            return pd.DataFrame(columns=["cent_bin","c","lo","hi"])
        M = cnm.merge(prim, on="cent_bin", suffixes=("_cnm","_prim"))
        Tc, Tlo, Thi = combine_product_asym(M["c_cnm"], M["lo_cnm"], M["hi_cnm"],
                                            M["c_prim"], M["lo_prim"], M["hi_prim"])
        return pd.DataFrame({"cent_bin": M["cent_bin"], "c": Tc, "lo": Tlo, "hi": Thi})

    # dispatch
    if comp == "cnm":        tab = _cnm_table()
    elif comp == "npdf":     tab = _npdf_table()
    elif comp == "eloss":    tab = _eloss_table()
    elif comp == "primordial": tab = _prim_table()
    elif comp == "total":    tab = _total_table()
    else:
        return pd.DataFrame(columns=["N_coll","c","lo","hi"])

    if tab.empty: 
        return pd.DataFrame(columns=["N_coll","c","lo","hi"])

    # attach N_coll per row using THIS model's Glauber mapping
    if "N_coll" in tab.columns:
        x = tab["N_coll"].astype(float).to_numpy()
    else:
        x = _ncoll_for_bins(C, tab["cent_bin"])
    out = pd.DataFrame({"N_coll": x,
                        "c": tab["c"].astype(float).to_numpy(),
                        "lo": tab["lo"].astype(float).to_numpy(),
                        "hi": tab["hi"].astype(float).to_numpy()})
    return out.sort_values("N_coll").reset_index(drop=True)

def _draw_band(ax, x, lo, hi, *, color, alpha=0.25, hatch=None, z=3, label=None,
               span_to=None, handles=None):
    x  = np.asarray(x, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    mask = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    x, lo, hi = x[mask], lo[mask], hi[mask]
    if x.size == 0:
        return
    if span_to is not None:
        x0, x1 = span_to
        x  = np.r_[x0,  x,  x1]
        lo = np.r_[lo[0], lo, lo[-1]]
        hi = np.r_[hi[0], hi, hi[-1]]
    poly = ax.fill_between(x, lo, hi,
                           color=color if not hatch else "none",
                           alpha=alpha if not hatch else 1.0,
                           zorder=z)
    if hatch:
        poly.set_edgecolor(color); poly.set_hatch(hatch); poly.set_linewidth(0.0)
    if handles is not None and label:
        handles.append(Patch(facecolor=("none" if hatch else color),
                             edgecolor=color, hatch=hatch,
                             alpha=(1.0 if hatch else alpha),
                             label=label))


# ----------------- figure: Total(y) per centrality, NPWLC vs Pert -------------
def compare_total_vs_y_per_centrality(models, pt_range, y_bins, state, *,
                                      model_styles, ylim=(0.2,1.6), xlim=(-5,5),
                                      legend_loc="best", ncols=2,
                                      save_pdf_prefix=None, debug=False):
    cent_bins = list(next(iter(models.values())).ctab["cent_bin"])
    n = len(cent_bins); nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.6*nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, cb in zip(axes, cent_bins):
        ax.set_xlabel(r"$y$"); ax.set_ylabel(r"$R_{pA}$")
        ax.minorticks_on(); ax.set_xlim(*xlim); ax.set_ylim(*ylim)

        for label, C in models.items():
            df = total_vs_y_in_centrality_fast(C, cb, state, y_bins, pt_range)
            if df.empty:
                if debug: print(f"[debug] empty Total(y) for {label} @ {cb}")
                continue
            st = model_styles[label]
            poly = ax.fill_between(df["y"], df["lo"], df["hi"], color=st["color"], alpha=st.get("band_alpha",0.25))
            ax.plot(df["y"], df["c"], st.get("ls","-"), lw=2.2, color=st["color"], marker=st.get("marker","o"),
                    label=f"{label} Total")

        _note(ax, state.replace("_", " "), loc="upper left")
        _note(ax, str(cb), loc="upper right")
        _note(ax, rf"{pt_range[0]:.1f} < $p_T$ < {pt_range[1]:.1f} GeV", loc="lower left")
        h,l = ax.get_legend_handles_labels()
        if l: ax.legend(frameon=False, loc=legend_loc)

    for j in range(n, nrows*ncols): axes[j].set_visible(False)
    if save_pdf_prefix:
        fig.savefig(f"{save_pdf_prefix}_{state}_{pt_range[0]:.1f}_{pt_range[1]:.1f}.pdf",
                    bbox_inches="tight", dpi=300)
    plt.show()

# ------------------- centrality-integrated helpers (robust) -------------------
def compare_total_vs_y_integrated(models, y_bins, pt_range, state, *,
                                  model_styles, ylim=(0.0,1.5), legend_loc="best",
                                  save_pdf=None, note=None):
    y_pairs = _as_y_bins(y_bins)  # <— NEW
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, C in models.items():
        df = C.total_vs_y_integrated_over_centrality(y_pairs, pt_range, state)
        if df.empty: 
            continue
        st = model_styles[label]
        ax.fill_between(df["y"], df["lo"], df["hi"], alpha=st["band_alpha"], color=st["color"])
        ax.plot(df["y"], df["c"], st["ls"], lw=2.2, color=st["color"], marker=st["marker"], label=f"{label} Total")
    ax.set_xlabel("y"); ax.set_ylabel(r"$R_{pA}$"); ax.minorticks_on(); ax.set_ylim(*ylim)
    _note(ax, PRETTY_STATE.get(state, state), loc="upper left")
    if note: _note(ax, note, loc="lower right")
    _legend_if_any(ax, frameon=False, loc=legend_loc)
    if save_pdf: plt.savefig(save_pdf, bbox_inches="tight", dpi=300)
    plt.show()

def compare_total_vs_pt_integrated(models: Dict[str,object], y_window: Tuple[float,float],
                                   pt_bins: Sequence[Tuple[float,float]], state: str, *,
                                   model_styles: Dict[str,dict],
                                   ylim=(0.0,1.6), xlim=(0.0,20.0),
                                   legend_loc="best", save_pdf: str | None = None,
                                   note: str | None = None):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, C in models.items():
        df = C.total_vs_pt_integrated_over_centrality(y_window, pt_bins, state, plot=False)
        if df.empty: continue
        st = model_styles[label]
        ax.fill_between(df["pt"], df["lo"], df["hi"], alpha=st["band_alpha"], color=st["color"])
        ax.plot(df["pt"], df["c"], st["ls"], lw=2.2, color=st["color"], marker=st["marker"], label=f"{label} Total")
    ax.set_xlabel(r"$p_T$ [GeV]"); ax.set_ylabel(r"$R_{pA}$"); ax.minorticks_on()
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    _note(ax, PRETTY_STATE.get(state, state), loc="upper left")
    if note: _note(ax, note, loc="lower right")
    # ax.legend(frameon=False, loc=legend_loc)
    _legend_if_any(ax, frameon=False, loc=legend_loc)
    if save_pdf: plt.savefig(save_pdf, bbox_inches="tight", dpi=300)
    plt.show()

# ---------------------- double ratio (centrality-integrated) ------------------
def theory_double_ratio_vs_pt(C, y_window: Tuple[float,float], pt_bins) -> pd.DataFrame:
    """DR = (psi2S/Jpsi) in pPb divided by the same in pp; reuses your combiner outputs."""
    def _rel(lo,c,hi):
        c = np.asarray(c, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float); eps=1e-14
        return np.where(np.abs(c)>eps,(c-lo)/c,0.0), np.where(np.abs(c)>eps,(hi-c)/c,0.0)
    j1 = C.total_vs_pt_integrated_over_centrality(y_window, pt_bins, "jpsi_1S", plot=False)
    p2 = C.total_vs_pt_integrated_over_centrality(y_window, pt_bins, "psi_2S",  plot=False)
    R  = p2["c"].to_numpy() / j1["c"].to_numpy()
    e2m,e2p = _rel(p2["lo"],p2["c"],p2["hi"]); e1m,e1p = _rel(j1["lo"],j1["c"],j1["hi"])
    lo = R*(1.0 - np.sqrt(e2m**2 + e1p**2)); hi = R*(1.0 + np.sqrt(e2p**2 + e1m**2))
    return pd.DataFrame({"pt": j1["pt"], "c": R, "lo": lo, "hi": hi})

def compare_double_ratio_vs_pt(models: Dict[str,object], y_window: Tuple[float,float],
                               pt_bins, *, model_styles: Dict[str,dict],
                               exp_df: pd.DataFrame | None = None, exp_state_name: str = "psi_2S_over_jpsi_1S",
                               ylim=(0.3, 1.5), legend_loc="best", save_pdf: str | None = None):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, C in models.items():
        df = theory_double_ratio_vs_pt(C, y_window, pt_bins)
        st = model_styles[label]
        ax.fill_between(df["pt"], df["lo"], df["hi"], alpha=st["band_alpha"], color=st["color"], label=f"{label} Theory")
        ax.plot(df["pt"], df["c"], st["ls"], lw=2.0, color=st["color"], marker=st["marker"])
    if exp_df is not None and not exp_df.empty:
        x = exp_df["pt"].to_numpy(float)
        xerr = np.vstack([x - exp_df["ptlo"].to_numpy(float), exp_df["pthi"].to_numpy(float) - x])
        y = exp_df["val"].to_numpy(float); yerr = exp_df["dtot"].to_numpy(float)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", mfc="black", mec="black",
                    ecolor="black", color="black", capsize=2.5, lw=1.1, ms=5, label="ALICE", zorder=5)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(r"$[\psi(2S)/J/\psi]_{p\mathrm{Pb}}/[\psi(2S)/J/\psi]_{pp}$")
    ax.set_ylim(*ylim); ax.minorticks_on(); 
    # ax.legend(frameon=False, loc=legend_loc)
    _legend_if_any(ax, frameon=False, loc=legend_loc)
    _note(ax, rf"{y_window[0]:.1f} < $y$ < {y_window[1]:.1f}", loc="upper right")
    if save_pdf: plt.savefig(save_pdf, bbox_inches="tight", dpi=300)
    plt.show()

# ------------------------------ tiny text note -------------------------------
def _note(ax, text, loc="upper left", pad=0.035, fs=10):
    ha = "left"  if "left"  in loc else "right"
    va = "top"   if "upper" in loc else "bottom"
    x  = pad if "left" in loc else 1 - pad
    y  = 1 - pad if "upper" in loc else pad
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs)

def components_vs_y_per_centrality(
    C,                       # Combiner
    state,                   # e.g. "jpsi_1S"
    pt_range,                # (pt_min, pt_max)
    y_width,                 # float OR (yl,yr) OR [(yl,yr), ...]
    components=("nPDF","eLoss","CNM","Primordial","Total"),
    save_pdf_prefix=None,
    note=None,
    ylim=(0.0,1.6),
    xlim=(-5,5),
    debug=False,
):
    import numpy as np, matplotlib.pyplot as plt, pandas as pd
    from combine_module import BAND_ALPHA, STATE_COLORS

    # accept float OR (yl,yr) OR list of (yl,yr)
    def _width(v):
        try:
            return float(v)
        except Exception:
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x,(int,float)) for x in v):
                return float(abs(v[1]-v[0]))
            if isinstance(v, (list, tuple)) and len(v) and len(v[0]) == 2:
                a,b = v[0]
                return float(abs(b-a))
            raise TypeError("y_width must be a float or an iterable of (yl,yr) bins")

    width = _width(y_width)
    want  = {c.lower() for c in components}

    cent_bins = list(C.ctab["cent_bin"])
    n = len(cent_bins); ncols = 2; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.6*nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, cb in zip(axes, cent_bins):
        ax.set_xlabel(r"$y$")
        ax.set_ylabel(r"$R_{pA}$")
        ax.minorticks_on()
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

        # --- nPDF, eLoss, CNM (state-blind) ---
        N = C._npdf_vs_y_in_cent(cb, y_width=width, pt_range=pt_range)      # y,c,lo,hi
        E = C._eloss_vs_y_in_cent(cb, pt_range=pt_range, y_width=width)     # y,c,lo,hi

        if not N.empty and ("npdf" in want):
            ax.fill_between(N["y"], N["lo"], N["hi"], color="#1f77b4", alpha=0.22, label="nPDF")
            ax.plot(N["y"], N["c"], "-", color="#1f77b4")

        if not E.empty and ("eloss" in want):
            ax.plot(E["y"], E["c"], "s-", color="#ff7f0e", label="eLoss")

        # you already have a CNM(y) helper that aligns/interpolates; use it
        Cnm = C._cnm_vs_y_in_cent(cb, y_width=width, pt_range=pt_range)
        if not Cnm.empty and ("cnm" in want):
            ax.fill_between(Cnm["y"], Cnm["lo"], Cnm["hi"], color="#2ca02c", alpha=BAND_ALPHA["CNM"], label="CNM")
            ax.plot(Cnm["y"], Cnm["c"], "-", color="#2ca02c")

        # --- Primordial (stateful) ---
        P = C._primordial_vs_y_in_cent(cb, state, pt_range, width)  # NOTE: positional width
        if not P.empty and ("primordial" in want):
            ax.fill_between(P["y"], P["lo"], P["hi"], color="#9467bd", alpha=BAND_ALPHA["Primordial"],
                            label=f"Primordial {state}")
            ax.plot(P["y"], P["c"], "-", color="#9467bd")

        # --- Total = CNM × Primordial (band only + central line) ---
        if ("total" in want) and (not Cnm.empty) and (not P.empty):
            # interpolate CNM onto Primordial y-grid, then multiply with asymmetric propagation
            y = P["y"].to_numpy(float)
            Cc  = np.interp(y, Cnm["y"], Cnm["c"])
            Clo = np.interp(y, Cnm["y"], Cnm["lo"])
            Chi = np.interp(y, Cnm["y"], Cnm["hi"])
            Tc, Tlo, Thi = combine_product_asym(Cc, Clo, Chi, P["c"], P["lo"], P["hi"])
            col = STATE_COLORS.get(state, "#d62728")
            ax.fill_between(y, Tlo, Thi, color=col, alpha=BAND_ALPHA["Total"], label="Total")
            ax.plot(y, Tc, "-", color=col)

        # panel notes
        ax.text(0.98, 0.96, str(cb), transform=ax.transAxes, ha="right", va="top")
        if note:
            ax.text(0.02, 0.04, note, transform=ax.transAxes, ha="left", va="bottom")
        h,l = ax.get_legend_handles_labels()
        if l: ax.legend(frameon=False, loc="best")

    for j in range(n, nrows*ncols):
        axes[j].set_visible(False)
    if save_pdf_prefix:
        fig.savefig(f"{save_pdf_prefix}_{state}.pdf", bbox_inches="tight")
    plt.show()


# --------------- centrality: Total vs N_coll (overlay models) -----------------
def compare_total_vs_Ncoll(
    models: Dict[str,object],
    y_window: Tuple[float,float],
    pt_range: Tuple[float,float],
    state: str = "jpsi_1S",
    *,
    what: str | Sequence[str] = "Total",
    model_styles: Dict[str,dict] = MODEL_STYLES,
    ylim: Tuple[float,float] = (0.2, 1.2),
    xlim: Tuple[float,float] | None = None,
    span_full_x: bool = False,              # extend band flush to xlim edges
    draw_central: bool = False,             # keep False for “band only”
    legend_loc: str = "best",
    legend_title: str | None = None,
    note: str | None = None,
    exp_df: pd.DataFrame | None = None,     # optional ALICE overlay
    exp_by: str = "rapidity",
    save_pdf: str | None = None
):
    wants = [what] if isinstance(what, str) else list(what)
    wants = [w.lower() for w in wants]   # normalize

    # --- draw CNM only once (model-agnostic) ---
    want_cnm = ("cnm" in wants)
    wants_wo_cnm = [w for w in wants if w != "cnm"]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    handles: list = []  # band legend boxes

    # pick a consistent x window if requested
    if xlim is None:
        first = next(iter(models.values()))
        base = first.ctab
        if "N_coll" in base.columns:
            xlim = (float(base["N_coll"].min()), float(base["N_coll"].max()))
        else:
            xlim = (float(base["N_part"].min() - 1.0), float(base["N_part"].max() - 1.0))

    # ---- CNM once (use the first model’s mapping) ----
    if want_cnm:
        label0, C0 = next(iter(models.items()))
        df_cnm = _component_vs_Ncoll(C0, "cnm", y_window, pt_range, state).copy()
        if not df_cnm.empty:
            cnm_color = globals().get("COMP_COLORS", {}).get("CNM", "#2ca02c")  # green default
            span = xlim if span_full_x else None
            _draw_band(ax, df_cnm["N_coll"], df_cnm["lo"], df_cnm["hi"],
                       color=cnm_color, alpha=BAND_ALPHA.get("CNM", 0.22),
                       hatch=None, z=3, label="CNM", span_to=span, handles=handles)

    # ---- model-dependent components (e.g., Total) ----
    for label, C in models.items():
        st = {"color": STATE_COLORS.get(label, None), "band_alpha": BAND_ALPHA, **model_styles.get(label, {})}
        hatch = st.get("hatch")

        for comp in wants_wo_cnm:
            df = _component_vs_Ncoll(C, comp, y_window, pt_range, state).copy()
            if df.empty:
                continue
            x, lo, hi, c = df["N_coll"], df["lo"], df["hi"], df["c"]
            span = xlim if span_full_x else None

            _draw_band(ax, x, lo, hi,
                       color=st.get("color", None),
                       alpha=st.get("band_alpha", 0.25),
                       hatch=hatch, z=st.get("z", 3),
                       label=f"{label} {comp.capitalize()}",
                       span_to=span, handles=handles)

            if draw_central:
                ax.plot(x, c, st.get("ls","-"), lw=2.0, color=st.get("color", None),
                        marker=st.get("marker","o"), zorder=st.get("z", 3) + 0.1)

    # --- optional ALICE overlay (fallback if plot_series isn’t in scope) ---
    if exp_df is not None and not exp_df.empty:
        if "plot_series" not in globals():
            def plot_series(ax, df, *, x="ncoll", label=None,
                            fmt="o", yerr_mode="stat_plus_uncorr", show_xerr=False):
                import numpy as np
                xv = df[x].astype(float).to_numpy()
                # y: prefer 'val' then fallbacks
                for col in ("val", "value", "RpA", "rpa", "y"):
                    if col in df.columns:
                        yv = df[col].astype(float).to_numpy()
                        break
                else:
                    yv = df.iloc[:, 0].astype(float).to_numpy()
                # yerr
                if yerr_mode == "stat_plus_uncorr":
                    if "dtot" in df.columns:
                        ye = df["dtot"].astype(float).to_numpy()
                    else:
                        def pick(up, dn):
                            if up in df.columns:  return np.nan_to_num(df[up].astype(float), 0.0).to_numpy()
                            if dn in df.columns:  return np.nan_to_num(df[dn].astype(float), 0.0).to_numpy()
                            return 0.0
                        stat = pick("stat_up", "stat_dn")
                        unc  = pick("sys_uncorr_up", "sys_uncorr_dn")
                        ye = np.sqrt(np.asarray(stat, float)**2 + np.asarray(unc, float)**2)
                else:
                    ye = None
                # xerr (optional)
                if show_xerr and {"x_low","x_high"} <= set(df.columns):
                    xe = np.vstack([
                        xv - df["x_low"].astype(float).to_numpy(),
                        df["x_high"].astype(float).to_numpy() - xv
                    ])
                else:
                    xe = None
                ax.errorbar(xv, yv, yerr=ye, xerr=xe, fmt=fmt, ms=5,
                            mfc="black", mec="black", ecolor="black", color="black",
                            capsize=2.5, lw=1.1, label=label, zorder=10)

        for key, g in exp_df.groupby(exp_by):
            plot_series(ax, g, x="ncoll", label=str(key),
                        fmt="o", yerr_mode="stat_plus_uncorr", show_xerr=False)

    ax.set_xlabel(r"$\langle N_{\mathrm{coll}}\rangle$")
    ax.set_ylabel(r"$R_{pA}$")
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.minorticks_on()
    _note(ax, PRETTY_STATE.get(state, state), loc="upper left")
    if note: _note(ax, note, loc="lower right")

    # legend: use our band patches + keep any exp handles
    h_line, l_line = ax.get_legend_handles_labels()
    keepH, keepL = [], []
    for h, l in zip(h_line, l_line):
        if any(k in l.lower() for k in ["npwlc","pert","total","cnm","eloss","npdf","primordial"]):
            continue
        keepH.append(h); keepL.append(l)
    if handles:
        keepH = handles + keepH
        keepL = [h.get_label() for h in handles] + keepL

    ax.legend(keepH, keepL, frameon=True, framealpha=0.92, loc=legend_loc, title=legend_title)

    if save_pdf:
        plt.savefig(save_pdf, bbox_inches="tight", dpi=300)
    plt.show()

import re, numpy as np

_cent_pat = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*%*\s*$")

def _cent_centers(cent_bin_series):
    xs = []
    for s in cent_bin_series:
        if hasattr(s, "left") and hasattr(s, "right"):
            lo, hi = float(s.left), float(s.right)
        else:
            s = str(s).strip().replace("–", "-").replace("%", "")
            m = _cent_pat.match(s)
            if not m: 
                raise ValueError(f"Bad cent_bin '{s}'")
            lo, hi = float(m.group(1)), float(m.group(2))
        xs.append(0.5*(lo + hi))
    xs = np.asarray(xs, float)
    if xs.size and np.nanmax(xs) <= 1.5:  # auto-scale fractional bins
        xs = xs * 100.0
    return xs

def compare_vs_centrality_by_state(models, y_window, pt_range, states, *,
                                   what="Total", show_cnm=True, cnm_as="band",
                                   ylim=(0,1.6), legend_loc="lower center",
                                   ncols=None, debug=False, save_pdf=None):
    """
    what: str or Sequence[str]; any of {"nPDF","eLoss","CNM","Primordial","Total"}.
    - nPDF/eLoss/CNM drawn once (using the first model).
    - Primordial/Total are overlaid across entries in `models` (e.g. NPWLC vs Pert).
    """
    import numpy as np, matplotlib.pyplot as plt
    wants = [what] if isinstance(what, str) else list(what)
    n = len(states); ncols = int(ncols or n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0*ncols, 4.8*nrows),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    first_label, C0 = next(iter(models.items()))

    for ax, state in zip(axes, states):
        ax.set_xlabel("centrality [%]"); ax.set_ylabel(r"$R_{pA}$")
        ax.minorticks_on(); ax.set_ylim(*ylim)

        # components common to all models
        if {"nPDF","eLoss","CNM"} & set(wants):
            cnm_tab = C0.cnm_vs_centrality(y_window, pt_range)
            if cnm_tab.empty:
                if debug: print("[debug] empty CNM centrality table"); continue
            x = cent_centers(cnm_tab["cent_bin"])
            if "nPDF" in wants:
                y  = cnm_tab["r_central"].to_numpy(float)
                ylo, yhi = cnm_tab["r_lo"].to_numpy(float), cnm_tab["r_hi"].to_numpy(float)
                ax.fill_between(x, ylo, yhi, color="#1f77b4", alpha=0.22, label="nPDF")
                ax.plot(x, y, "-", color="#1f77b4")
            if "eLoss" in wants:
                ax.plot(x, cnm_tab["eloss"], "s-", color="#ff7f0e", label="eLoss")
            if "CNM" in wants:
                if cnm_as == "band":
                    ax.fill_between(x, cnm_tab["cnm_lo"], cnm_tab["cnm_hi"], color="#2ca02c", alpha=0.22, label="CNM")
                    ax.plot(x, cnm_tab["cnm_c"], "-", color="#2ca02c")
                else:
                    y = cnm_tab["cnm_c"].to_numpy(float)
                    yerr = np.vstack([y - cnm_tab["cnm_lo"], cnm_tab["cnm_hi"] - y])
                    ax.errorbar(x, y, yerr=yerr, fmt="D", capsize=3, color="#2ca02c", label="CNM")

        # overlays across models for Primordial / Total
        for label, C in models.items():
            if "Primordial" in wants:
                tab = C.primordial_vs_centrality(pt_range, y_window, state)
                if not tab.empty:
                    x = cent_centers(tab["cent_bin"])
                    lo, c, hi = tab["lo"], tab["c"], tab["hi"]
                    st = MODEL_STYLES[label]
                    poly = ax.fill_between(x, lo, hi, color=st["color"], alpha=st["band_alpha"], zorder=st.get("z",2))
                    if st.get("hatch"):     # hatch for Pert, solid for NPWLC
                        poly.set_facecolor("none"); poly.set_edgecolor(st["color"]); poly.set_hatch(st["hatch"])
                    ax.plot(x, c, st.get("ls","-"), lw=2.2, color=st["color"], marker=st.get("marker","o"),
                            label=f"{label} Primordial")
            if "Total" in wants or "CNM×Primordial" in wants:
                tab = C.total_vs_centrality(y_window, pt_range, state)
                if not tab.empty:
                    x = cent_centers(tab["cent_bin"])
                    lo, c, hi = tab["total_lo"], tab["total_c"], tab["total_hi"]
                    st = MODEL_STYLES[label]
                    poly = ax.fill_between(x, lo, hi, color=st["color"], alpha=st["band_alpha"], zorder=st.get("z",2))
                    if st.get("hatch"):
                        poly.set_facecolor("none"); poly.set_edgecolor(st["color"]); poly.set_hatch(st["hatch"])
                    ax.plot(x, c, st.get("ls","-"), lw=2.2, color=st["color"], marker=st.get("marker","o"),
                            label=f"{label} Total")

        _note(ax, PRETTY_STATE.get(state, state), loc="upper left")
        ax.legend(frameon=False, loc=legend_loc)

    for j in range(len(states), nrows*ncols): axes[j].set_visible(False)
    if save_pdf: plt.savefig(save_pdf, bbox_inches="tight", dpi=300)
    plt.show()


def compare_vs_Ncoll(models: Dict[str, object], *,
                     y_window: Tuple[float, float],
                     pt_range: Tuple[float, float],
                     state: str = "jpsi_1S",
                     what: str = "Total",
                     model_styles: Dict[str, dict] | None = None,
                     ylim: Tuple[float, float] = (0.2, 1.2),
                     legend_loc: str = "best",
                     save_pdf: str | None = None):
    """Overlay RpA vs <N_coll> for multiple Combiner models.
       what ∈ {"CNM","Total"} (Primordial doesn’t depend on N_coll directly)."""
    import numpy as np
    import matplotlib.pyplot as plt

    model_styles = model_styles or {k: {} for k in models.keys()}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    # Use the first model's Glauber x-axis for consistency
    first = next(iter(models.values()))
    xref = first.ctab.copy()
    if "N_coll" in xref.columns:
        x = xref["N_coll"].to_numpy(float)
    else:
        x = xref["N_part"].to_numpy(float) - 1.0

    for label, C in models.items():
        if what.lower() == "cnm":
            tab = C.cnm_vs_centrality(y_window, pt_range)
        elif what.lower() == "total":
            # Prefer dedicated helper if present
            if hasattr(C, "total_vs_Ncoll"):
                tab = C.total_vs_Ncoll(y_window, pt_range, state)
                # it already carries N_coll; trust it if so
                if "N_coll" in tab.columns:
                    xvals = tab["N_coll"].to_numpy(float)
                else:
                    xvals = x
                y_c, y_lo, y_hi = tab["c"], tab["lo"], tab["hi"]
                st = model_styles.get(label, {})
                yerr = np.vstack([y_c - y_lo, y_hi - y_c])
                ax.errorbar(xvals, y_c, yerr=yerr, fmt=st.get("marker", "o"),
                            linestyle=st.get("ls", "-"), color=st.get("color", None),
                            label=f"{label} {what}", capsize=0, zorder=st.get("z", 3))
                continue
            else:
                # fallback: combine CNM × Primordial per bin
                cnm  = C.cnm_vs_centrality(y_window, pt_range)
                prim = C.primordial_vs_centrality(pt_range, y_window, state)

                # align by cent_bin
                M = cnm.merge(prim, on="cent_bin", suffixes=("_cnm", "_prim"))
                Tc, Tlo, Thi = combine_product_asym(
                    M["c_cnm"].to_numpy(float),  M["lo_cnm"].to_numpy(float),  M["hi_cnm"].to_numpy(float),
                    M["c_prim"].to_numpy(float), M["lo_prim"].to_numpy(float), M["hi_prim"].to_numpy(float),
                )
                y_c, y_lo, y_hi = Tc, Tlo, Thi
        else:
            raise ValueError("what must be 'CNM' or 'Total'")

        # y from CNM table
        y_c  = tab["c"].to_numpy(float) if what.lower()=="cnm" else np.asarray(y_c, float)
        y_lo = tab["lo"].to_numpy(float) if what.lower()=="cnm" else np.asarray(y_lo, float)
        y_hi = tab["hi"].to_numpy(float) if what.lower()=="cnm" else np.asarray(y_hi, float)
        st = model_styles.get(label, {})
        yerr = np.vstack([y_c - y_lo, y_hi - y_c])
        ax.errorbar(x, y_c, yerr=yerr, fmt=st.get("marker", "o"),
                    linestyle=st.get("ls", "-"), color=st.get("color", None),
                    label=f"{label} {what}", capsize=0, zorder=st.get("z", 3))

    ax.set_xlabel(r"$\langle N_{\mathrm{coll}}\rangle$")
    ax.set_ylabel(r"$R_{pA}$")
    ax.minorticks_on()
    ax.set_ylim(*ylim)
    _legend_if_any(ax, frameon=False, loc=legend_loc)

    if save_pdf:
        plt.savefig(save_pdf, bbox_inches="tight", dpi=300)
    plt.show()


