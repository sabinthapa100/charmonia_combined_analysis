# cross_sections.py
# σ_pp wrappers + Glauber-aware σ_pA/σ_AB → RpA/RAA, with fast caching & bin tools

from __future__ import annotations
from dataclasses import replace
from typing import Iterable, Literal, Tuple, Optional
import numpy as np, math

from quenching import (QuenchParams, xA0_from_L, pA_cross_section, AB_cross_section)

# ===================== σ_pp shape and caching ===============================

def sigma_pp(P, roots_GeV: float, y: float, pt: float) -> float:
    """ Thin wrapper around your Particle.d2sigma_pp. """
    return float(P.d2sigma_pp(float(y), float(pt), float(roots_GeV)))

class SigmaPPTable:
    """
    Simple rect-grid cache for σ_pp(y,pt) with bilinear in y and 1D linear in pt.
    """
    def __init__(self, P, roots_GeV: float, y_grid: np.ndarray, pt_grid: np.ndarray):
        self.P, self.roots = P, float(roots_GeV)
        self.y = np.asarray(y_grid, float)
        self.pt = np.asarray(pt_grid, float)
        Z = np.empty((self.y.size, self.pt.size), float)
        for i, yy in enumerate(self.y):
            for j, pp in enumerate(self.pt):
                Z[i, j] = float(P.d2sigma_pp(float(yy), float(pp), self.roots))
        self._grid = (self.y, self.pt, Z)

    def __call__(self, y: float, pt_arr: np.ndarray) -> np.ndarray:
        yv = float(y)
        y0 = np.clip(np.searchsorted(self.y, yv) - 1, 0, self.y.size - 2)
        t = 0.0 if self.y[y0+1] == self.y[y0] else (yv - self.y[y0])/(self.y[y0+1] - self.y[y0])
        z0 = np.interp(pt_arr, self.pt, self._grid[2][y0],   left=self._grid[2][y0,0],   right=self._grid[2][y0,-1])
        z1 = np.interp(pt_arr, self.pt, self._grid[2][y0+1], left=self._grid[2][y0+1,0], right=self._grid[2][y0+1,-1])
        return (1.0 - t)*z0 + t*z1

def _dsig_pp_shape(P, roots_GeV: float, table: Optional[SigmaPPTable] = None):
    if table is None:
        def f(y, pt_arr):
            return np.asarray([P.d2sigma_pp(float(y), float(p), float(roots_GeV)) for p in pt_arr], float)
        return f
    else:
        def f(y, pt_arr):
            return table(float(y), np.asarray(pt_arr, float))
        return f

# ===================== Glauber L_eff helpers =================================

def _avg_T(gl, kind: Literal["pA","AA"], bmin: float, bmax: float) -> float:
    """ Inelastic-weighted <T> in [bmin,bmax], robust if custom L_eff not provided. """
    T_b = gl.TpA_b if kind=="pA" else gl.TAA_b
    b   = gl.b_grid
    m = (b >= bmin) & (b <= bmax)
    if not np.any(m): return 0.0
    sigma_fm2 = float(gl.sigma_nn_mb)*0.1
    w = b[m]*(1.0 - np.exp(-sigma_fm2*T_b[m]))
    num = float(np.trapezoid(w*T_b[m], b[m]))
    den = float(np.trapezoid(w,        b[m]))
    return num/max(den, 1e-12)

def L_eff_bin(gl, kind: Literal["pA","AA"], cmin: float, cmax: float, rho0_fm3: float = 0.17,
              method_pA: Literal["binomial","optical"]="binomial") -> float:
    """
    Prefer your Glauber’s own L_eff if available (pA). Returns L_eff [fm].
    """
    if kind == "pA" and hasattr(gl, "leff_bin_pA"):
        return float(gl.leff_bin_pA(cmin, cmax, rho0_fm3=rho0_fm3, Lp_fm=1.5, method=method_pA))
    # generic fallback using CDFs available on your gl object
    bmin = float(np.interp(cmin, gl.cum_pA if kind=="pA" else gl.cum_AA, gl.b_grid))
    bmax = float(np.interp(cmax, gl.cum_pA if kind=="pA" else gl.cum_AA, gl.b_grid))
    Tbar = _avg_T(gl, kind, bmin, bmax)          # fm^-2
    return Tbar/max(rho0_fm3, 1e-12)             # fm

def L_eff_minbias(gl, kind: Literal["pA","AA"], rho0_fm3: float = 0.17) -> float:
    if kind == "pA" and hasattr(gl, "leff_minbias_pA"):
        return float(gl.leff_minbias_pA(rho0_fm3=rho0_fm3))
    # fallback
    T_b = gl.TpA_b if kind == "pA" else gl.TAA_b
    b   = gl.b_grid
    sigma_fm2 = float(gl.sigma_nn_mb) * 0.1
    w = b * (1.0 - np.exp(-sigma_fm2 * T_b))
    Tbar = float(np.trapezoid(w * T_b, b) / np.trapezoid(w, b))
    return Tbar / max(rho0_fm3, 1e-12)
    
# ===================== σ_pA / σ_AB and nuclear ratios ========================
from math import exp

def _x2_minus(roots, mT, y): return (mT/float(roots)) * exp(-float(y))  # target-like
def _x2_plus (roots, mT, y): return (mT/float(roots)) * exp(+float(y))  # projectile-like

def sigma_pA(P, roots_GeV: float, qpar: QuenchParams, y: float, pt: float, table=None) -> float:
    mT  = float(P.mT(pt)); y_mx = float(P.y_max(roots_GeV, mT))
    # A is the nucleus (target side): use x2(y) with explicit coherence cap
    xA = min(xA0_from_L(qpar.LA_fm), _x2_minus(roots_GeV, mT, y))
    dspp = _dsig_pp_shape(P, roots_GeV, table=table)
    return float(pA_cross_section(y, pt, mT, xA, 0.0, y_mx, dspp, qpar))

def sigma_AB(P, roots_GeV: float, qpar: QuenchParams, y: float, pt: float, table=None) -> float:
    mT  = float(P.mT(pt)); y_mx = float(P.y_max(roots_GeV, mT))
    # A at -y, B at +y:
    xA = min(xA0_from_L(qpar.LA_fm), _x2_minus(roots_GeV, mT, y))
    xB = min(xA0_from_L(qpar.LB_fm), _x2_plus (roots_GeV, mT, y))
    dspp = _dsig_pp_shape(P, roots_GeV, table=table)
    return float(AB_cross_section(y, pt, mT, xA, xB, y_mx, dspp, qpar))

# def sigma_pA(P, roots_GeV: float, qpar: QuenchParams, y: float, pt: float,
#              table: Optional[SigmaPPTable] = None) -> float:
#     mT  = float(P.mT(pt))
#     y_mx = float(P.y_max(roots_GeV, mT))
#     xA = P.xA(y, pt, roots_GeV, qpar.LA_fm)
#     xB = 0.0  # unused in pA integral
#     dspp = _dsig_pp_shape(P, roots_GeV, table=table)
#     return float(pA_cross_section(y, pt, mT, xA, xB, y_mx, dspp, qpar))

# def sigma_AB(P, roots_GeV: float, qpar: QuenchParams, y: float, pt: float,
#              table: Optional[SigmaPPTable] = None) -> float:
#     mT  = float(P.mT(pt))
#     y_mx = float(P.y_max(roots_GeV, mT))
#     xA = P.xA(-y, pt, roots_GeV, qpar.LA_fm)
#     xB = P.xA(+y, pt, roots_GeV, qpar.LB_fm)
#     dspp = _dsig_pp_shape(P, roots_GeV, table=table)
#     return float(AB_cross_section(y, pt, mT, xA, xB, y_mx, dspp, qpar))

def RpA(P, roots_GeV: float, qpar: QuenchParams, y: float, pt: float,
        table: Optional[SigmaPPTable] = None) -> float:
    den = sigma_pp(P, roots_GeV, y, pt)
    if den <= 0.0: return 0.0
    return sigma_pA(P, roots_GeV, qpar, y, pt, table)/den

def RAA(P, roots_GeV: float, qpar: QuenchParams, y: float, pt: float,
        table: Optional[SigmaPPTable] = None) -> float:
    den = sigma_pp(P, roots_GeV, y, pt)
    return 0.0 if den <= 0.0 else sigma_AB(P, roots_GeV, qpar, y, pt, table)/den

def map_sigma(P, roots_GeV, qpar, y_grid, pt_grid, kind="pA", table=None):
    Y = np.asarray(y_grid, float); PT = np.asarray(pt_grid, float)
    Z = np.empty((Y.size, PT.size), float)
    mT_of = lambda pt: float(P.mT(pt))
    for i,y in enumerate(Y):
        for j,pt in enumerate(PT):
            mT = mT_of(pt)
            y_mx = float(P.y_max(roots_GeV, mT))
            if kind=="pp":
                Z[i,j] = sigma_pp(P, roots_GeV, y, pt)
            elif kind=="pA":
                xA = float(P.xA(y, pt, roots_GeV,qpar.LA_fm))
                Z[i,j] = pA_cross_section(y, pt, mT, xA, 0.0, y_mx, _dsig_pp_shape(P, roots_GeV, table), qpar)
            else: # "AA"
                xA = float(P.xA(y, pt, roots_GeV,qpar.LA_fm))
                xB = float(P.xB(y, pt, roots_GeV, qpar.LB_fm))
                Z[i,j] = AB_cross_section(y, pt, mT, xA, xB, y_mx, _dsig_pp_shape(P, roots_GeV, table), qpar)
    return Z

# ===================== averaging & scans (matching your ELossRun) ============

def _gl_nodes(a: float, b: float, n: int):
    x,w = np.polynomial.legendre.leggauss(int(n))
    xm, xc = 0.5*(b-a), 0.5*(b+a)
    return (xc + xm*x, xm*w)

def _pt_weight(P, roots, qpar: QuenchParams, pt: float, table: Optional[SigmaPPTable]) -> float:
    # same convention as your eloss pipeline: w(pt) ∝ σ_pA(y=0,pt)*pt (clamped)
    try:
        val = sigma_pA(P, roots, qpar, 0.0, float(pt), table)
    except Exception:
        val = 0.0
    return max(1e-300, val*float(pt))
# drop this near the other bin-averagers in cross_sections.py

def sigma_avg_in_bin(P, roots, qpar, y_range, pt_range, kind="pA", Ny=24, Npt=48, table=None):
    ys, wy = _gl_nodes(y_range[0], y_range[1], Ny)
    pts,wp = _gl_nodes(pt_range[0], pt_range[1], Npt)
    acc = 0.0
    for y, wy_ in zip(ys, wy):
        for p, wp_ in zip(pts, wp):
            if kind=="pp":
                s = sigma_pp(P, roots, y, p)
            elif kind=="pA":
                s = sigma_pA(P, roots, qpar, y, p, table)
            else:
                s = sigma_AB(P, roots, qpar, y, p, table)
            acc += wy_*wp_*s
    return acc / ((y_range[1]-y_range[0])*(pt_range[1]-pt_range[0]))

def average_R(
    P, roots_GeV: float, qpar: QuenchParams,
    y_range: Tuple[float,float], pt_range: Tuple[float,float],
    Ny: int = 32, Npt: int = 64, kind: Literal["pA","AA"] = "pA",
    table: Optional[SigmaPPTable] = None
) -> float:
    y1,y2 = y_range; p1,p2 = pt_range
    ys, wy = _gl_nodes(y1,y2,Ny)
    pts,wp = _gl_nodes(p1,p2,Npt)
    denom = 0.0
    for p,wp_ in zip(pts,wp):
        denom += _pt_weight(P, roots_GeV, qpar, p, table)*wp_
    if denom <= 0.0: return 0.0
    acc = 0.0
    for y,wy_ in zip(ys,wy):
        num_pt = 0.0
        for p,wp_ in zip(pts,wp):
            R = RpA(P, roots_GeV, qpar, y, p, table) if kind=="pA" \
                else RAA(P, roots_GeV, qpar, y, p, table)
            num_pt += R*_pt_weight(P, roots_GeV, qpar, p, table)*wp_
        acc += (num_pt/denom)*wy_
    return acc/(y2-y1)

def RpA_vs_centrality(
    P, roots_GeV: float, qpar_base: QuenchParams, gl,
    cent_edges: Iterable[float], y_range=(-5.0,5.0), pt_range=(0.1,20.0),
    Ny=32, Npt=64, method_pA: Literal["binomial","optical"]="binomial",
    table: Optional[SigmaPPTable] = None
):
    edges = list(cent_edges)
    rows = []
    for cmin,cmax in zip(edges[:-1], edges[1:]):
        LA = L_eff_bin(gl, "pA", cmin, cmax, method_pA=method_pA)   # [fm]
        qp = replace(qpar_base, LA_fm=float(LA), LB_fm=float(LA))
        val = average_R(P, roots_GeV, qp, y_range, pt_range, Ny, Npt, kind="pA", table=table)
        rows.append(dict(cmin=cmin, cmax=cmax, cent_mid=0.5*(cmin+cmax)*100.0,
                         xerr=0.5*(cmax-cmin)*100.0, RpA=val, L_eff=LA))
    return rows

def RAA_vs_centrality(
    P, roots_GeV: float, qpar_base: QuenchParams, gl,
    cent_edges: Iterable[float], y_range=(-5.0,5.0), pt_range=(0.1,20.0),
    Ny=32, Npt=64, table: Optional[SigmaPPTable] = None
):
    edges = list(cent_edges)
    rows = []
    for cmin,cmax in zip(edges[:-1], edges[1:]):
        LA = L_eff_bin(gl, "AA", cmin, cmax)   # optical fallback
        qp = replace(qpar_base, LA_fm=float(LA), LB_fm=float(LA))
        val = average_R(P, roots_GeV, qp, y_range, pt_range, Ny, Npt, kind="AA", table=table)
        rows.append(dict(cmin=cmin, cmax=cmax, cent_mid=0.5*(cmin+cmax)*100.0,
                         xerr=0.5*(cmax-cmin)*100.0, RAA=val, L_eff=LA))
    return rows

def RpA_band_vs_centrality(
    P, roots_GeV: float, gl, qpar_template: QuenchParams,
    q0_list=(0.05,0.07,0.09), cent_edges=(0.0,0.2,0.4,0.6,0.8,1.0),
    y_range=(-5.0,5.0), pt_range=(0.0,30.0), Ny=24, Npt=56,
    method_pA: Literal["binomial","optical"]="binomial", table: Optional[SigmaPPTable] = None
):
    rows=[]
    for cmin,cmax in zip(cent_edges[:-1], cent_edges[1:]):
        LA = L_eff_bin(gl, "pA", cmin, cmax, method_pA=method_pA)
        vals=[]
        for q0 in q0_list:
            qp = replace(qpar_template, qhat0=float(q0), LA_fm=float(LA), LB_fm=float(LA))
            vals.append(average_R(P, roots_GeV, qp, y_range, pt_range, Ny, Npt, kind="pA", table=table))
        rows.append(dict(cent_mid=0.5*(cmin+cmax)*100.0, xerr=0.5*(cmax-cmin)*100.0,
                         RpA=float(np.mean(vals)), lo=float(np.min(vals)), hi=float(np.max(vals)),
                         L_eff=LA))
    return rows

def map_R(P, roots_GeV: float, qpar: QuenchParams, y_grid: np.ndarray, pt_grid: np.ndarray,
          kind: Literal["pA","AA"]="pA", table: Optional[SigmaPPTable] = None) -> np.ndarray:
    """ Return a (len(y_grid), len(pt_grid)) array of RpA/RAA (no binning). """
    Y = np.asarray(y_grid, float); PT = np.asarray(pt_grid, float)
    Z = np.empty((Y.size, PT.size), float)
    for i, y in enumerate(Y):
        for j, pt in enumerate(PT):
            Z[i,j] = RpA(P, roots_GeV, qpar, y, pt, table) if kind=="pA" \
                     else RAA(P, roots_GeV, qpar, y, pt, table)
    return Z

def RpA_band_vs_y(P, roots, gl, qpar_template, cent_bin, y_edges, pt_range, q0_list=(0.05,0.07,0.09), method_pA="binomial", table=None):
    L,R = cent_bin
    LA = L_eff_bin(gl, "pA", L, R, method_pA=method_pA)
    mids = 0.5*(y_edges[1:]+y_edges[:-1]); rows=[]
    for yl,yr in zip(y_edges[:-1], y_edges[1:]):
        vals=[]
        for q0 in q0_list:
            qp = replace(qpar_template, qhat0=float(q0), LA_fm=float(LA), LB_fm=float(LA))
            vals.append(average_R(P, roots, qp, (yl,yr), pt_range, Ny=24, Npt=56, kind="pA", table=table))
        rows.append(dict(y_left=yl, y_right=yr, y_mid=0.5*(yl+yr), RpA=float(np.mean(vals)), lo=float(np.min(vals)), hi=float(np.max(vals))))
    return rows

def RpA_band_vs_pt(P, roots, gl, qpar_template, cent_bin, y_range, pt_edges, q0_list=(0.05,0.07,0.09), method_pA="binomial", table=None):
    L,R = cent_bin
    LA = L_eff_bin(gl, "pA", L, R, method_pA=method_pA)
    rows=[]
    for pl,pr in zip(pt_edges[:-1], pt_edges[1:]):
        vals=[]
        for q0 in q0_list:
            qp = replace(qpar_template, qhat0=float(q0), LA_fm=float(LA), LB_fm=float(LA))
            vals.append(average_R(P, roots, qp, y_range, (pl,pr), Ny=24, Npt=56, kind="pA", table=table))
        rows.append(dict(pt_left=pl, pt_right=pr, pt_mid=0.5*(pl+pr), RpA=float(np.mean(vals)), lo=float(np.min(vals)), hi=float(np.max(vals))))
    return rows


# ===================== Plotting helpers (compact) ============================

def plot_sigma_surfaces(P, roots_GeV, qpar, y_edges, pt_edges, table=None):
    import matplotlib.pyplot as plt
    y = 0.5*(y_edges[:-1] + y_edges[1:])
    p = 0.5*(pt_edges[:-1] + pt_edges[1:])
    Zpp = map_sigma(P, roots_GeV, qpar, y, p, kind="pp", table=table)
    ZpA = map_sigma(P, roots_GeV, qpar, y, p, kind="pA", table=table)
    fig,axs = plt.subplots(1,2,figsize=(10,4))
    im0 = axs[0].imshow(Zpp.T, origin="lower", extent=[y_edges[0],y_edges[-1],pt_edges[0],pt_edges[-1]], aspect="auto")
    axs[0].set_title(r"$\sigma_{pp}(y,p_T)$"); axs[0].set_xlabel("y"); axs[0].set_ylabel(r"$p_T$ [GeV]"); fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(ZpA.T, origin="lower", extent=[y_edges[0],y_edges[-1],pt_edges[0],pt_edges[-1]], aspect="auto")
    axs[1].set_title(r"$\sigma_{pA}(y,p_T)$"); axs[1].set_xlabel("y"); axs[1].set_ylabel(r"$p_T$ [GeV]"); fig.colorbar(im1, ax=axs[1])
    fig.tight_layout(); return fig, axs, (Zpp, ZpA)

def plot_r_vs_y_in_bins(P, roots_GeV: float, qpar_base: QuenchParams, gl,
                        cent_edges=(0.0,0.2,0.4,0.6,0.8,1.0), pt_range=(0.1,20.0),
                        y_edges=None, which: Literal["pA","AA"]="pA", ncols=3,
                        method_pA: Literal["binomial","optical"]="binomial",
                        table: Optional[SigmaPPTable] = None):
    import matplotlib.pyplot as plt, numpy as np
    if y_edges is None: y_edges = np.linspace(-5,5,41)
    tags = [f"{int(100*L)}-{int(100*R)}%" for L,R in zip(cent_edges[:-1],cent_edges[1:])] + ["MinBias"]
    n = len(tags); nrows = (n + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.2*nrows), squeeze=False); axes = axes.ravel()

    for i,(L,R) in enumerate(list(zip(cent_edges[:-1],cent_edges[1:])) + [(None,None)]):
        if L is None:
            LA = L_eff_minbias(gl, "pA" if which=="pA" else "AA")
        else:
            LA = L_eff_bin(gl, "pA" if which=="pA" else "AA", L, R, method_pA=method_pA)
        qp = replace(qpar_base, LA_fm=float(LA), LB_fm=float(LA))
        mids = 0.5*(y_edges[1:]+y_edges[:-1]); vals=[]
        for yl,yr in zip(y_edges[:-1], y_edges[1:]):
            vals.append(average_R(P, roots_GeV, qp, (yl,yr), pt_range, Ny=24, Npt=56,
                                  kind="pA" if which=="pA" else "AA", table=table))
        ax = axes[i]
        ax.step(y_edges[1:], vals, where="post")
        ax.set_ylim(0.4, max(1.2, 1.05*max(vals)))
        ax.set_xlim(min(y_edges), max(y_edges))
        ax.set_xlabel("y"); ax.set_ylabel("R"+which.upper())
        ax.set_title(tags[i])
        ax.grid(True, ls="--", alpha=0.35)
    for j in range(i+1, axes.size): axes[j].set_visible(False)
    fig.tight_layout(); return fig, axes[:n]

def plot_r_vs_pt_in_bins(P, roots_GeV: float, qpar_base: QuenchParams, gl,
                         cent_edges=(0.0,0.2,0.4,0.6,0.8,1.0), y_range=(-5,5),
                         y_label: Optional[str] = None,  # <-- 1. ADDED THIS ARGUMENT
                         pt_edges=None, which: Literal["pA","AA"]="pA", ncols=3,
                         method_pA: Literal["binomial","optical"]="optical",
                         table: Optional[SigmaPPTable] = None):
    import matplotlib.pyplot as plt, numpy as np
    if pt_edges is None: pt_edges = np.linspace(0.0, 20.0, 21)
    tags = [f"{int(100*L)}-{int(100*R)}%" for L,R in zip(cent_edges[:-1],cent_edges[1:])] + ["MinBias"]
    n = len(tags); nrows = (n + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.2*nrows), squeeze=False); axes = axes.ravel()

    # --- We can create the y-label string from the tuple if it's not provided ---
    # This makes the function more robust.
    if y_label is None:
        y_label = f"{y_range[0]} < y < {y_range[1]}"
    # --- End of optional addition ---

    for i,(L,R) in enumerate(list(zip(cent_edges[:-1],cent_edges[1:])) + [(None,None)]):
        if L is None:
            LA = L_eff_minbias(gl, "pA" if which=="pA" else "AA")
        else:
            LA = L_eff_bin(gl, "pA" if which=="pA" else "AA", L, R, method_pA=method_pA)
        qp = replace(qpar_base, LA_fm=float(LA), LB_fm=float(LA))
        mids = 0.5*(pt_edges[1:]+pt_edges[:-1]); vals=[]
        for pl,pr in zip(pt_edges[:-1], pt_edges[1:]):
            vals.append(average_R(P, roots_GeV, qp, y_range, (pl,pr), Ny=24, Npt=56,
                                  kind="pA" if which=="pA" else "AA", table=table))
        ax = axes[i]
        ax.step(pt_edges[1:], vals, where="post")
        ax.set_ylim(0.4, max(1.2, 1.05*max(vals)))
        ax.set_xlim(min(pt_edges), max(pt_edges))
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel("R"+which.upper())
        
        # --- 2. REPLACED ax.set_title ---
        # ax.set_title(tags[i]) # <-- This line is removed
        
        # Add centrality text inside the plot at top-right
        ax.text(0.95, 0.95, tags[i], 
                ha='right', va='top', 
                transform=ax.transAxes, 
                fontsize=12)
        
        # --- 3. ADDED y_label text ---
        if y_label:
            ax.text(0.95, 0.88, y_label, 
                    ha='right', va='top', 
                    transform=ax.transAxes, 
                    fontsize=10)
        # --- End of changes ---
            
       # ax.grid(True, ls="--", alpha=0.35) # <-- Restored gridlines
        
    for j in range(i+1, axes.size): axes[j].set_visible(False)
    fig.tight_layout(); return fig, axes[:n]
