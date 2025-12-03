# eloss_cronin_centrality.py
"""
Coherent energy loss + pT broadening (Cronin) with centrality dependence.

This module is the notebook logic refactored into reusable functions.

Core physics:
    * R_pA_eloss(P, roots_GeV, qp, y, pT)
    * R_pA_broad(P, roots_GeV, qp, y, pT)
    * R_pA_factored = R_pA_eloss * R_pA_broad

Binned averages (no band):
    * R_binned_2D(...)
    * rpa_binned_vs_y(...)
    * rpa_binned_vs_pT(...)
    * rpa_vs_centrality(...)

Bands (two-point scans in q0 and p0):
    * rpa_band_vs_y_eloss / rpa_band_vs_y_broad / rpa_band_vs_y
    * rpa_band_vs_pT_eloss / rpa_band_vs_pT_broad / rpa_band_vs_pT
    * rpa_band_vs_centrality

Plot helpers (publication style, optional):
    * plot_RpA_vs_y_components_per_centrality(...)
    * plot_RpA_vs_pT_components_per_centrality(...)
    * plot_RpA_vs_centrality_components_band(...)

Min-bias centrality weights:
    * By default we use an exponential scheme w(c) ∝ exp(-c/c0) in c∈[0,1]
      integrated over each centrality bin ("exp" mode).
    * You can switch to optical-Glauber weights ("optical" mode).
    * Or supply your own custom weights ("custom" mode).

All interfaces are designed so you can:
  - call from a notebook with a single energy (e.g. 5.02 TeV),
  - later overlay additional energies on top,
  - later multiply with nPDF factors from npdf_centrality to get
    npdf, eloss, broad, eloss×broad, and npdf×eloss×broad (CNM total).

Author: Sabin + ChatGPT refactor.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Iterable, Literal, Sequence, Tuple, List

import numpy as np

# torch import is required for quenching_fast
import torch

# --- local physics modules ---
from particle import Particle, PPSpectrumParams
from glauber import OpticalGlauber
import quenching_fast as QF

# ------------------------------
# Global numerical knobs / floors
# ------------------------------
F1_FLOOR = 1e-16
F2_FLOOR = 1e-12   # for kinematic edge protection
ZC_EPS   = 1e-12

XMIN_SAFE = 1e-5
XMAX_SAFE = 0.99


# ----------------------------------------------------------------------
# Centrality weights for MB (w(c) scheme, same as in npdf_centrality)
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


def _get_mb_weight_array(
    cent_bins: List[Tuple[float, float]],
    glauber: OpticalGlauber,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
) -> np.ndarray:
    """
    Helper: return normalized MB weights array w_i for each centrality bin
    (same order as cent_bins).

    Modes
    -----
    "exp"     : use make_centrality_weight_dict(cent_bins, c0=mb_c0)
    "optical" : use QF._optical_bin_weight_pA(glauber, a, b)
    "custom"  : use mb_weights_custom[tag] for tag="a-b%"
    """
    labels = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    if mb_weight_mode == "exp":
        w_dict = make_centrality_weight_dict(cent_bins, c0=mb_c0)
        w_arr = np.array([float(w_dict[lab]) for lab in labels], float)

    elif mb_weight_mode == "optical":
        w_arr = np.array(
            [QF._optical_bin_weight_pA(glauber, a, b) for (a, b) in cent_bins],
            float
        )

    elif mb_weight_mode == "custom":
        if mb_weights_custom is None:
            raise ValueError(
                "mb_weights_custom must be provided when mb_weight_mode='custom'"
            )
        w_arr = np.array(
            [float(mb_weights_custom.get(lab, 0.0)) for lab in labels],
            float
        )

    else:
        raise ValueError(f"Unknown mb_weight_mode='{mb_weight_mode}'")

    s = float(w_arr.sum())
    if s > 0.0:
        w_arr /= s
    else:
        w_arr = np.ones_like(w_arr) / max(len(w_arr), 1)
    return w_arr


# ------------------------------------------------
# Small device helper (CPU / GPU, robust)
# ------------------------------------------------
def _qp_device(qp) -> torch.device:
    """
    Infer torch.device from QuenchParams.qp.device when possible.
    Fall back to GPU if available, else CPU.
    """
    dev_str = getattr(qp, "device", None)
    if dev_str is None:
        dev_str = "cuda" if (QF._HAS_TORCH and torch.cuda.is_available()) else "cpu"
    if dev_str == "cuda" and not torch.cuda.is_available():
        dev_str = "cpu"
    return torch.device(dev_str)


# ------------------------------------------------
# Optional helper: scale p0 in pp spectrum
# ------------------------------------------------
def particle_with_scaled_p0(P: Particle, scale: float) -> Particle:
    """
    Return a new Particle with pp.p0 → scale * p0 (m,n unchanged).
    Used to define the Cronin (broadening) band.

    All other attributes (family, state, mass) are copied as is.
    """
    pp = P.pp
    new_pp = PPSpectrumParams(p0=pp.p0 * scale, m=pp.m, n=pp.n)
    return Particle(
        family=P.family,
        state=P.state,
        mass_override_GeV=P.mass_override_GeV,
        pp_params=new_pp,
    )


# ------------------------------------------------
# pp parametrisation: F1(pT) * F2(y,pT)
# ------------------------------------------------
def F1_t(P: Particle, pT_t: torch.Tensor) -> torch.Tensor:
    """
    F1(p_T) = (p0^2 / (p0^2 + p_T^2))^m
    """
    p0, m, _ = P.pp.p0, P.pp.m, P.pp.n
    p0_sq = float(p0) * float(p0)
    return (p0_sq / (p0_sq + pT_t * pT_t)) ** m


def F2_t(
    P: Particle,
    y_t: torch.Tensor,
    pT_t: torch.Tensor,
    roots_GeV: float,
) -> torch.Tensor:
    """
    F2(y,p_T) = [1 - 2 M_T cosh(y) / sqrt(s)]^n, clamped ≥ 0.
    """
    _, _, n = P.pp.p0, P.pp.m, P.pp.n
    M = float(P.M_GeV)
    roots = float(roots_GeV)
    pT_sq = pT_t * pT_t
    mT    = torch.sqrt(pT_sq + M * M)
    arg   = 1.0 - (2.0 * mT / roots) * torch.cosh(y_t)
    arg_clamped = torch.clamp(arg, min=1e-30)
    return arg_clamped ** n


def F2_t_pt(
    P: Particle,
    y_val: float,
    pT_t: torch.Tensor,
    roots_GeV: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    F2(y_val, p_T) for a tensor of p_T values.
    """
    if device is None:
        device = pT_t.device
    y_t = torch.full_like(pT_t, float(y_val), device=device)
    return F2_t(P, y_t, pT_t, roots_GeV)


# ------------------------------------------------
# x_A (coherence) with safety clamp
# ------------------------------------------------
def xA_scalar(P: Particle, roots_GeV: float, qp, y: float, pT: float) -> float:
    """
    x_A = min( x0(L_A), x_2 ) with
      x_2 = (m_T / sqrt(s)) e^{-y},
      x0(L_A) = ħc / (2 m_p L_A).

    Then clamped to [XMIN_SAFE, XMAX_SAFE].
    """
    M  = float(P.M_GeV)
    mT = math.sqrt(M * M + float(pT) ** 2)
    x2 = (mT / float(roots_GeV)) * math.exp(-float(y))
    x0 = QF.xA0_from_L(qp.LA_fm)
    xA = min(x0, x2)
    xA = max(XMIN_SAFE, min(XMAX_SAFE, xA))
    return float(xA)


# ------------------------------------------------
# Coherent energy-loss factor R_pA^loss
# ------------------------------------------------
def R_pA_eloss(
    P: Particle,
    roots_GeV: float,
    qp,
    y: float,
    pT: float,
    Ny: int | None = None,
) -> float:
    r"""
    Coherent energy-loss factor

      R_pA^loss(y,p_T;L_eff)
      = p0 + ∫ dδy  P̂_A(δy; x_A,L)
                   · [F2(y+δy,p_T) / F2(y,p_T)].

    where p0 = 1 - ∫ dδy P̂_A(δy) is the discrete
    no-loss probability (Arleo–Peigné).
    """
    if not QF._HAS_TORCH:
        raise RuntimeError("R_pA_eloss: torch (double precision) required.")

    device = _qp_device(qp)
    M   = float(P.M_GeV)
    pT0 = float(pT)
    mT  = math.sqrt(M * M + pT0 * pT0)

    # kinematic δy_max
    y_max_pt = QF.y_max(roots_GeV, mT)
    dym      = QF.dymax(+y, y_max_pt)
    if dym <= QF.DY_EPS:
        return 1.0

    if Ny is None:
        Ny = QF._Ny_from_dymax(dym)

    xA_val = xA_scalar(P, roots_GeV, qp, y, pT0)

    with torch.no_grad():
        xA    = torch.tensor([xA_val], dtype=torch.float64, device=device)
        y0_t  = torch.tensor([y],     dtype=torch.float64, device=device)
        pT0_t = torch.tensor([pT0],   dtype=torch.float64, device=device)

        F2_den_t = F2_t(P, y0_t, pT0_t, roots_GeV)[0]
        if F2_den_t <= F2_FLOOR:
            return 1.0

        mapping = getattr(qp, "mapping", "exp")

        if mapping == "exp":
            # u = ln δy ∈ [umin, umax], δy = e^u
            umin = -30.0
            umax = math.log(max(dym, 1e-300))
            u, wu = QF._gl_nodes_torch(umin, umax, Ny, device)

            dy = torch.exp(u)
            z  = torch.expm1(dy).clamp_min(QF.Z_FLOOR)

            ph = QF.PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT0)
            if (ph <= 0).all():
                return 1.0

            yshift = y + dy
            F2_num = F2_t(P, yshift, pT0_t.expand_as(yshift), roots_GeV)
            ratio  = F2_num / F2_den_t

            jac  = torch.exp(u)                # dδy/du
            val  = torch.sum(wu * jac * ph * ratio)
            Zc   = torch.sum(wu * jac * ph)

        else:
            # "linear" mapping: direct δy GL
            dy, wy = QF._gl_nodes_torch(0.0, float(dym), Ny, device)
            z  = torch.expm1(dy).clamp_min(QF.Z_FLOOR)

            ph = QF.PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT0)
            if (ph <= 0).all():
                return 1.0

            yshift = y + dy
            F2_num = F2_t(P, yshift, pT0_t.expand_as(yshift), roots_GeV)
            ratio  = F2_num / F2_den_t

            val = torch.sum(wy * ph * ratio)
            Zc  = torch.sum(wy * ph)

        Zc = torch.clamp(Zc, min=0.0, max=1.0)
        if float(Zc.item()) < ZC_EPS:
            return 1.0

        p0 = torch.clamp(1.0 - Zc, 0.0, 1.0)   # discrete no-loss prob.
        R_loss = p0 + val
        return float(R_loss.item())


# ------------------------------------------------
# pT broadening factor R_pA^broad
# ------------------------------------------------
def R_pA_broad(
    P: Particle,
    roots_GeV: float,
    qp,
    y: float,
    pT: float,
    Nphi: int = 256,
) -> float:
    r"""
    R_pA^broad(y,p_T;L_eff) =
      ∫_0^{2π} dφ/(2π) [F1(|p_T - δp_T(φ)|)/F1(p_T)]
                         · [F2(y,|p_T - δp_T(φ)|)/F2(y,p_T)].
    """
    if not QF._HAS_TORCH:
        raise RuntimeError("R_pA_broad: torch missing.")
    device = _qp_device(qp)

    with torch.no_grad():
        xA_val = xA_scalar(P, roots_GeV, qp, y, pT)
        xA = torch.tensor([xA_val], dtype=torch.float64, device=device)

        dpta = QF._dpt_from_xL_t(qp, xA, qp.LA_fm,
                                 hard=qp.use_hard_cronin)[0]
        if torch.abs(dpta) < 1e-8:
            return 1.0

        dpta = torch.clamp(dpta, min=-5.0, max=5.0)

        phi, wphi, cphi, sphi = QF._phi_nodes_gl_torch(Nphi, device)
        pshift = QF._shift_pT_pA(pT, dpta, cphi, sphi)

        pT0_t = torch.tensor([pT], dtype=torch.float64, device=device)

        # F1 ratio
        F1_den = F1_t(P, pT0_t)[0]
        if F1_den <= F1_FLOOR:
            return 1.0
        F1_num = F1_t(P, pshift)
        R1 = F1_num / F1_den

        # F2 ratio
        F2_den = F2_t_pt(P, y_val=y, pT_t=pT0_t,
                         roots_GeV=roots_GeV, device=device)[0]
        if F2_den <= F2_FLOOR:
            return 1.0
        F2_num = F2_t_pt(P, y_val=y, pT_t=pshift,
                         roots_GeV=roots_GeV, device=device)
        R2 = F2_num / F2_den

        R_phi   = R1 * R2
        R_broad = torch.sum(R_phi * wphi)
        return float(R_broad.item())


def R_pA_factored(
    P: Particle,
    roots_GeV: float,
    qp,
    y: float,
    pT: float,
    Ny_eloss: int | None = None,
    Nphi_broad: int = 64,
) -> float:
    """
    Factorised Arleo–Peigné approximation:
      R_pA ≃ R_pA^loss · R_pA^broad
    """
    Rloss  = R_pA_eloss(P, roots_GeV, qp, y, pT, Ny=Ny_eloss)
    Rbroad = R_pA_broad(P, roots_GeV, qp, y, pT, Nphi=Nphi_broad)
    return Rloss * Rbroad


# ------------------------------------------------
# σ_pp weight from table or parametrisation
# ------------------------------------------------
def _sigma_pp_weight(P, roots_GeV: float, table_or_none, y: float, pT: float) -> float:
    """
    σ_pp(y,pT;√s) used as weight.
    """
    if QF._HAS_TORCH and isinstance(table_or_none, QF.TorchSigmaPPTable):
        dev = table_or_none.device
        with torch.no_grad():
            y_t = torch.tensor([y],  dtype=torch.float64, device=dev)
            p_t = torch.tensor([pT], dtype=torch.float64, device=dev)
            return float(table_or_none(y_t, p_t)[0, 0].item())
    else:
        return float(P.d2sigma_pp(float(y), float(pT), float(roots_GeV)))


# ----------------------------------------------------------------
# Generic 2D bin average
# ----------------------------------------------------------------
def R_binned_2D(
    R_func,                     # R_func(y,pT) -> float
    P, roots_GeV: float,
    y_range, pt_range,
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    table_for_pp=None,
    weight_ref_y: float | str = "local",
) -> float:
    """
    Generic bin average of R(y,pT) over y_range × pt_range.

      For weight_kind="pp":
        <R> = (∫ dy dpT σ_pp(y_w,pT)·pT R) / (∫ dy dpT σ_pp(y_w,pT)·pT)

      For weight_kind="flat":
        <R> = (∫ dy dpT R) / (∫ dy dpT 1)

      y_w = y if weight_ref_y == "local", else y_ref (float).
    """
    yl, yr = y_range
    pl, pr = pt_range

    y_nodes, y_w = QF._gl_nodes_np(yl, yr, Ny_bin)
    p_nodes, p_w = QF._gl_nodes_np(pl, pr, Npt_bin)

    if isinstance(weight_ref_y, str) and weight_ref_y.lower() == "local":
        y_ref = None
    else:
        y_ref = float(weight_ref_y)

    acc_num, acc_den = 0.0, 0.0

    for yi, wy in zip(y_nodes, y_w):
        y_for_w = float(yi) if y_ref is None else y_ref
        for pj, wp in zip(p_nodes, p_w):
            pj_f = float(pj)
            R    = float(R_func(float(yi), pj_f))

            if weight_kind == "pp":
                wgt = _sigma_pp_weight(P, roots_GeV, table_for_pp,
                                       y_for_w, pj_f)
                wgt *= max(pj_f, 1e-8)   # σ_pp·pT
            else:
                wgt = 1.0

            acc_num += wy * wp * R * wgt
            acc_den += wy * wp * wgt

    if acc_den <= 0:
        return acc_num
    return float(acc_num / acc_den)


# ------------------------------------------------
# RpA(y): binned in y, integrated over pT
# ------------------------------------------------
def rpa_binned_vs_y(
    P, roots_GeV: float, qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    components: Sequence[Literal["loss", "broad", "total"]] = ("total",),
    Ny_bin: int = 12, Npt_bin: int = 24,
    table_for_pp=None,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    RpA vs y (binned in y, integrated over pT) for each centrality bin + MB.

    MB can use:
      * mb_weight_mode="exp"     → exponential w(c) scheme (default)
      * mb_weight_mode="optical" → optical Glauber weights
      * mb_weight_mode="custom"  → user-provided mb_weights_custom[tag]
    """
    y_edges = np.asarray(y_edges, float)
    assert y_edges.ndim == 1 and y_edges.size >= 2
    Ny_bins = y_edges.size - 1
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    comps  = list(components)
    labels = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    L_by = glauber.leff_bins_pA(cent_bins, method="optical")
    Leff_dict = {lab: float(L_by[lab]) for lab in labels}

    # MB weights (array over cent_bins)
    w_arr_mb = _get_mb_weight_array(
        cent_bins, glauber,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    w_dict = {lab: w_arr_mb[i] for i, lab in enumerate(labels)}

    R_comp = {comp: {lab: np.zeros(Ny_bins, float) for lab in labels}
              for comp in comps}

    for i in range(Ny_bins):
        y_range = (float(y_edges[i]), float(y_edges[i+1]))

        for lab in labels:
            L   = Leff_dict[lab]
            qpL = replace(qp_base, LA_fm=float(L))

            def R_loss(y, pT, qpL=qpL):
                return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=None)

            def R_broad(y, pT, qpL=qpL):
                return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=64)

            for comp in comps:
                if comp == "loss":
                    R_func = R_loss
                elif comp == "broad":
                    R_func = R_broad
                elif comp == "total":
                    def R_func(y, pT, qpL=qpL):
                        return R_loss(y, pT, qpL=qpL) * R_broad(y, pT, qpL=qpL)
                else:
                    raise ValueError(f"Unknown component: {comp}")

                R_bar = R_binned_2D(
                    R_func, P, roots_GeV,
                    y_range, pt_range,
                    Ny_bin=Ny_bin, Npt_bin=Npt_bin,
                    weight_kind=weight_kind,
                    table_for_pp=table_for_pp,
                    weight_ref_y=weight_ref_y,
                )
                R_comp[comp][lab][i] = R_bar

    # min-bias
    for comp in comps:
        R_mat = np.vstack([R_comp[comp][lab] for lab in labels])  # (Ncent, Ny)
        w_arr = np.array([w_dict[lab] for lab in labels])
        R_MB  = np.average(R_mat, axis=0, weights=w_arr)
        R_comp[comp]["MB"] = R_MB

    return y_centers, R_comp, labels


# ------------------------------------------------
# RpA(pT): binned in pT, integrated over y
# ------------------------------------------------
def rpa_binned_vs_pT(
    P, roots_GeV: float, qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    components: Sequence[Literal["loss", "broad", "total"]] = ("total",),
    Ny_bin: int = 12, Npt_bin: int = 24,
    table_for_pp=None,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Same as rpa_binned_vs_y but swapping y ↔ pT roles.

    Returns
    -------
      pT_centers : array
      R_comp     : dict[component][tag] -> array
      labels     : list of centrality labels (no MB).
    """
    pT_edges = np.asarray(pT_edges, float)
    assert pT_edges.ndim == 1 and pT_edges.size >= 2
    Np = pT_edges.size - 1
    pT_centers = 0.5 * (pT_edges[:-1] + pT_edges[1:])

    comps  = list(components)
    labels = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    L_by = glauber.leff_bins_pA(cent_bins, method="optical")
    Leff_dict = {lab: float(L_by[lab]) for lab in labels}

    w_arr_mb = _get_mb_weight_array(
        cent_bins, glauber,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    w_dict = {lab: w_arr_mb[i] for i, lab in enumerate(labels)}

    R_comp = {comp: {lab: np.zeros(Np, float) for lab in labels}
              for comp in comps}

    for i in range(Np):
        pt_range = (float(pT_edges[i]), float(pT_edges[i+1]))

        for lab in labels:
            L   = Leff_dict[lab]
            qpL = replace(qp_base, LA_fm=float(L))

            def R_loss(y, pT, qpL=qpL):
                return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=None)

            def R_broad(y, pT, qpL=qpL):
                return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=64)

            for comp in comps:
                if comp == "loss":
                    R_func = R_loss
                elif comp == "broad":
                    R_func = R_broad
                elif comp == "total":
                    def R_func(y, pT, qpL=qpL):
                        return R_loss(y, pT, qpL=qpL) * R_broad(y, pT, qpL=qpL)
                else:
                    raise ValueError(f"Unknown component: {comp}")

                R_bar = R_binned_2D(
                    R_func, P, roots_GeV,
                    y_range, pt_range,
                    Ny_bin=Ny_bin, Npt_bin=Npt_bin,
                    weight_kind=weight_kind,
                    table_for_pp=table_for_pp,
                    weight_ref_y=weight_ref_y,
                )
                R_comp[comp][lab][i] = R_bar

    for comp in comps:
        R_mat = np.vstack([R_comp[comp][lab] for lab in labels])  # (Ncent, Np)
        w_arr = np.array([w_dict[lab] for lab in labels])
        R_MB  = np.average(R_mat, axis=0, weights=w_arr)
        R_comp[comp]["MB"] = R_MB

    return pT_centers, R_comp, labels


# ------------------------------------------------
# Centrality dependence: RpA(cent)
# ------------------------------------------------
def rpa_vs_centrality(
    P, roots_GeV: float, qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_range, pt_range,
    component: Literal["loss", "broad", "total"] = "total",
    Ny_bin: int = 16, Npt_bin: int = 32,
    table_for_pp=None,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Centrality dependence of RpA:

      <R>_bin(a–b%) = ⟨R(y,pT)⟩_{y_range × pt_range} in that bin.

    L_eff(a–b%) from OpticalGlauber; weights are either flat or σ_pp(y_w,pT)·pT
    (y_w = y or y_w = y_ref depending on weight_ref_y).

    MB is computed from per-bin values using the same centrality weights
    as other functions (mb_weight_mode).
    """
    assert component in ("loss", "broad", "total")
    labels = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]

    L_by = glauber.leff_bins_pA(cent_bins, method="optical")
    Leff_dict = {lab: float(L_by[lab]) for lab in labels}

    R_vals = []

    for lab, (a, b) in zip(labels, cent_bins):
        L   = Leff_dict[lab]
        qpL = replace(qp_base, LA_fm=L)

        def R_loss(y, pT, qpL=qpL):
            return R_pA_eloss(P, roots_GeV, qpL, y, pT, Ny=None)

        def R_broad(y, pT, qpL=qpL):
            return R_pA_broad(P, roots_GeV, qpL, y, pT, Nphi=64)

        if component == "loss":
            R_func = R_loss
        elif component == "broad":
            R_func = R_broad
        else:
            def R_func(y, pT, qpL=qpL):
                return R_loss(y, pT, qpL=qpL) * R_broad(y, pT, qpL=qpL)

        R_bin = R_binned_2D(
            R_func, P, roots_GeV,
            y_range, pt_range,
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            table_for_pp=table_for_pp,
            weight_ref_y=weight_ref_y,
        )
        R_vals.append(R_bin)

    R_vals = np.array(R_vals)

    # MB over centralities using chosen weights
    w_bins = _get_mb_weight_array(
        cent_bins, glauber,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    R_MB = float(np.average(R_vals, weights=w_bins))

    return labels, R_vals, R_MB


# ------------------------------------------------
# Two-point bands + combination
# ------------------------------------------------
def _two_point_band(R_lo: np.ndarray, R_hi: np.ndarray):
    """
    Given arrays R(q_min) and R(q_max), return
      Rc, Rlow, Rhigh  with symmetric error.
    """
    Rc = 0.5 * (R_lo + R_hi)
    dR = 0.5 * np.abs(R_hi - R_lo)
    return Rc, Rc - dR, Rc + dR


def combine_factorized_bands_1d(
    RL_c, RL_lo, RL_hi,
    RB_c, RB_lo, RB_hi,
):
    """
    Combine loss & broad bands into total, assuming factorisation:

      R_tot = R_L * R_B
      (δR_tot / R_tot)^2 = (δR_L / R_L)^2 + (δR_B / R_B)^2.

    Inputs:
      RL_* and RB_* : dict[tag] -> array (or scalar)

    Returns:
      RT_c, RT_lo, RT_hi : dict[tag] -> array (or scalar)
    """
    RT_c, RT_lo, RT_hi = {}, {}, {}
    for lab in RL_c.keys():
        Lc  = np.asarray(RL_c[lab])
        Llo = np.asarray(RL_lo[lab])
        Lhi = np.asarray(RL_hi[lab])

        Bc  = np.asarray(RB_c[lab])
        Blo = np.asarray(RB_lo[lab])
        Bhi = np.asarray(RB_hi[lab])

        dL = 0.5 * np.abs(Lhi - Llo)
        dB = 0.5 * np.abs(Bhi - Blo)

        Lc_safe = np.where(np.abs(Lc) > 1e-12, Lc, 1.0)
        Bc_safe = np.where(np.abs(Bc) > 1e-12, Bc, 1.0)

        Rc   = Lc * Bc
        rel2 = (dL / Lc_safe) ** 2 + (dB / Bc_safe) ** 2
        dR   = Rc * np.sqrt(rel2)

        RT_c[lab], RT_lo[lab], RT_hi[lab] = Rc, Rc - dR, Rc + dR

    return RT_c, RT_lo, RT_hi


# ------------------------------------------------
# Bands vs y
# ------------------------------------------------
def rpa_band_vs_y_eloss(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    q0_pair=(0.05, 0.09),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Binned R_pA^loss(y) band from q0_pair.

    Returns
    -------
      y_cent
      RL_c[lab], RL_lo[lab], RL_hi[lab]   (lab includes "MB")
      labels   (centrality labels)
    """
    q0_lo, q0_hi = q0_pair
    qp_lo = replace(qp_base, qhat0=float(q0_lo))
    qp_hi = replace(qp_base, qhat0=float(q0_hi))

    y_cent_lo, R_lo, labels = rpa_binned_vs_y(
        P, roots_GeV, qp_lo,
        glauber, cent_bins,
        y_edges, pt_range,
        components=("loss",),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    y_cent_hi, R_hi, _ = rpa_binned_vs_y(
        P, roots_GeV, qp_hi,
        glauber, cent_bins,
        y_edges, pt_range,
        components=("loss",),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    assert np.allclose(y_cent_lo, y_cent_hi)
    y_cent = y_cent_lo

    RL_c, RL_lo, RL_hi = {}, {}, {}
    for lab in R_lo["loss"].keys():   # cent bins + "MB"
        Rc, Rl, Rh = _two_point_band(R_lo["loss"][lab],
                                     R_hi["loss"][lab])
        RL_c[lab], RL_lo[lab], RL_hi[lab] = Rc, Rl, Rh

    return y_cent, RL_c, RL_lo, RL_hi, labels


def rpa_band_vs_y_broad(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Binned R_pA^broad(y) band from scaling p0 in the pp spectrum.

    Returns
    -------
      y_cent
      RB_c[lab], RB_lo[lab], RB_hi[lab]
      labels
    """
    P_lo = particle_with_scaled_p0(P, p0_scale_pair[0])
    P_hi = particle_with_scaled_p0(P, p0_scale_pair[1])

    y_cent_lo, R_lo, labels = rpa_binned_vs_y(
        P_lo, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        components=("broad",),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    y_cent_hi, R_hi, _ = rpa_binned_vs_y(
        P_hi, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        components=("broad",),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=table_for_pp,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    assert np.allclose(y_cent_lo, y_cent_hi)
    y_cent = y_cent_lo

    RB_c, RB_lo, RB_hi = {}, {}, {}
    for lab in R_lo["broad"].keys():
        Rc, Rl, Rh = _two_point_band(R_lo["broad"][lab],
                                     R_hi["broad"][lab])
        RB_c[lab], RB_lo[lab], RB_hi[lab] = Rc, Rl, Rh

    return y_cent, RB_c, RB_lo, RB_hi, labels


def rpa_band_vs_y(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    components=("loss", "broad", "total"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    table_for_pp=None,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Full RpA band vs y:

      • eloss band from q0_pair
      • broad band from p0_scale_pair
      • total band from factorised combination in quadrature.
    """
    # ---- eLoss band (q0) ----
    y_cent, RL_c, RL_lo, RL_hi, labels = rpa_band_vs_y_eloss(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        q0_pair=q0_pair,
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )

    # ---- broad band (p0) ----
    y_cent2, RB_c, RB_lo, RB_hi, _ = rpa_band_vs_y_broad(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        p0_scale_pair=p0_scale_pair,
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=table_for_pp,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    assert np.allclose(y_cent, y_cent2)

    # ---- total band from loss ⊗ broad (quadrature) ----
    RT_c, RT_lo, RT_hi = combine_factorized_bands_1d(
        RL_c, RL_lo, RL_hi,
        RB_c, RB_lo, RB_hi,
    )

    bands: dict[str, tuple[dict, dict, dict]] = {}
    if "loss" in components:
        bands["loss"] = (RL_c, RL_lo, RL_hi)
    if "broad" in components:
        bands["broad"] = (RB_c, RB_lo, RB_hi)
    if "total" in components:
        bands["total"] = (RT_c, RT_lo, RT_hi)

    return y_cent, bands, labels


# ------------------------------------------------
# Bands vs pT
# ------------------------------------------------
def rpa_band_vs_pT_eloss(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    q0_pair=(0.05, 0.09),
    component="loss",
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    assert component in ("loss", "total")

    q0_lo, q0_hi = q0_pair
    qp_lo = replace(qp_base, qhat0=float(q0_lo))
    qp_hi = replace(qp_base, qhat0=float(q0_hi))

    pT_cent_lo, R_lo, labels = rpa_binned_vs_pT(
        P, roots_GeV, qp_lo,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=None,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    pT_cent_hi, R_hi, _ = rpa_binned_vs_pT(
        P, roots_GeV, qp_hi,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=None,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    assert np.allclose(pT_cent_lo, pT_cent_hi)
    pT_cent = pT_cent_lo

    RL_c, RL_lo, RL_hi = {}, {}, {}
    for lab in R_lo[component].keys():   # cent bins + "MB"
        Rc, Rl, Rh = _two_point_band(R_lo[component][lab],
                                     R_hi[component][lab])
        RL_c[lab], RL_lo[lab], RL_hi[lab] = Rc, Rl, Rh

    return pT_cent, RL_c, RL_lo, RL_hi, labels


def rpa_band_vs_pT_broad(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    p0_scale_pair=(0.9, 1.1),
    component="broad",
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    assert component in ("broad", "total")

    P_lo = particle_with_scaled_p0(P, p0_scale_pair[0])
    P_hi = particle_with_scaled_p0(P, p0_scale_pair[1])

    pT_cent_lo, R_lo, labels = rpa_binned_vs_pT(
        P_lo, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=None,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    pT_cent_hi, R_hi, _ = rpa_binned_vs_pT(
        P_hi, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        components=(component,),
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        table_for_pp=None,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    assert np.allclose(pT_cent_lo, pT_cent_hi)
    pT_cent = pT_cent_lo

    RB_c, RB_lo, RB_hi = {}, {}, {}
    for lab in R_lo[component].keys():   # cent bins + "MB"
        Rc, Rl, Rh = _two_point_band(R_lo[component][lab],
                                     R_hi[component][lab])
        RB_c[lab], RB_lo[lab], RB_hi[lab] = Rc, Rl, Rh

    return pT_cent, RB_c, RB_lo, RB_hi, labels


def rpa_band_vs_pT(
    P, roots_GeV: float,
    qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    components=("loss", "broad", "total"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Full RpA band vs pT (y-integrated).

      • eloss band from q0_pair
      • broad band from p0_scale_pair
      • total band from factorised combination in quadrature.
    """
    # loss band
    pT_cent, RL_c, RL_lo, RL_hi, labels = rpa_band_vs_pT_eloss(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        q0_pair=q0_pair,
        component="loss",
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )

    # broad band
    pT_cent2, RB_c, RB_lo, RB_hi, _ = rpa_band_vs_pT_broad(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        p0_scale_pair=p0_scale_pair,
        component="broad",
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )
    assert np.allclose(pT_cent, pT_cent2)

    # total
    RT_c, RT_lo, RT_hi = combine_factorized_bands_1d(
        RL_c, RL_lo, RL_hi,
        RB_c, RB_lo, RB_hi,
    )

    bands: dict[str, tuple[dict, dict, dict]] = {}
    if "loss" in components:
        bands["loss"] = (RL_c, RL_lo, RL_hi)
    if "broad" in components:
        bands["broad"] = (RB_c, RB_lo, RB_hi)
    if "total" in components:
        bands["total"] = (RT_c, RT_lo, RT_hi)

    return pT_cent, bands, labels


# ------------------------------------------------
# Bands vs centrality
# ------------------------------------------------
def rpa_band_vs_centrality(
    P, roots_GeV: float, qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_range, pt_range,
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 16, Npt_bin: int = 32,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Error bands vs centrality:

      • q0_pair       → energy-loss band
      • p0_scale_pair → pp(p0) scale band for Cronin (broadening)
      • total band    → quadrature combination of eloss + broad.

    We do NOT treat the (0,100) "MB bin" as a separate optical bin here;
    MB is computed from chosen centrality weights over the genuine bins.
    """
    # Drop any explicit 0-100 bin from the core averaging:
    core_bins = [b for b in cent_bins if not (b[0] == 0 and b[1] == 100)]
    labels    = [f"{int(a)}-{int(b)}%" for (a, b) in core_bins]

    # ---------- loss band (q0 scan) ----------
    q0_lo, q0_hi = q0_pair
    RL_lo, RL_hi, RL_c = {}, {}, {}

    for q0, store in [(q0_lo, RL_lo), (q0_hi, RL_hi)]:
        qp_q = replace(qp_base, qhat0=float(q0))
        _, Rvals_q, _ = rpa_vs_centrality(
            P, roots_GeV, qp_q, glauber, core_bins,
            y_range, pt_range,
            component="loss",
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            mb_weight_mode=mb_weight_mode,
            mb_c0=mb_c0,
            mb_weights_custom=mb_weights_custom,
        )
        for lab, val in zip(labels, Rvals_q):
            store[lab] = val

    for lab in labels:
        Rc = 0.5 * (RL_lo[lab] + RL_hi[lab])
        dR = 0.5 * abs(RL_hi[lab] - RL_lo[lab])
        RL_c[lab]  = Rc
        RL_lo[lab] = Rc - dR
        RL_hi[lab] = Rc + dR

    # ---------- broad band (p0 scan) ----------
    RB_lo, RB_hi, RB_c = {}, {}, {}

    for p0_scale, store in [(p0_scale_pair[0], RB_lo),
                            (p0_scale_pair[1], RB_hi)]:
        P_scaled = particle_with_scaled_p0(P, p0_scale)
        _, Rvals_q, _ = rpa_vs_centrality(
            P_scaled, roots_GeV, qp_base, glauber, core_bins,
            y_range, pt_range,
            component="broad",
            Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind,
            weight_ref_y=weight_ref_y,
            mb_weight_mode=mb_weight_mode,
            mb_c0=mb_c0,
            mb_weights_custom=mb_weights_custom,
        )
        for lab, val in zip(labels, Rvals_q):
            store[lab] = val

    for lab in labels:
        Rc = 0.5 * (RB_lo[lab] + RB_hi[lab])
        dR = 0.5 * abs(RB_hi[lab] - RB_lo[lab])
        RB_c[lab]  = Rc
        RB_lo[lab] = Rc - dR
        RB_hi[lab] = Rc + dR

    # ---------- combine loss + broad in quadrature ----------
    RT_c, RT_lo, RT_hi = {}, {}, {}
    for lab in labels:
        Lc, Llo, Lhi = RL_c[lab], RL_lo[lab], RL_hi[lab]
        Bc, Blo, Bhi = RB_c[lab], RB_lo[lab], RB_hi[lab]

        dL = 0.5 * abs(Lhi - Llo)
        dB = 0.5 * abs(Bhi - Blo)

        Lc_safe = Lc if abs(Lc) > 1e-10 else 1.0
        Bc_safe = Bc if abs(Bc) > 1e-10 else 1.0

        Rc   = Lc * Bc
        rel2 = (dL / Lc_safe) ** 2 + (dB / Bc_safe) ** 2
        dR   = Rc * math.sqrt(rel2)

        RT_c[lab]  = Rc
        RT_lo[lab] = Rc - dR
        RT_hi[lab] = Rc + dR

    # ---------- MB values (centrality weights over core bins only) ----------
    w_bins = _get_mb_weight_array(
        core_bins, glauber,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )

    def _mb_from_dict(Dc, Dlo, Dhi):
        arr_c  = np.array([Dc[lab]  for lab in labels])
        arr_lo = np.array([Dlo[lab] for lab in labels])
        arr_hi = np.array([Dhi[lab] for lab in labels])

        Rc  = float(np.average(arr_c,  weights=w_bins))
        Rlo = float(np.average(arr_lo, weights=w_bins))
        Rhi = float(np.average(arr_hi, weights=w_bins))
        return Rc, Rlo, Rhi

    RMB_loss  = _mb_from_dict(RL_c, RL_lo, RL_hi)
    RMB_broad = _mb_from_dict(RB_c, RB_lo, RB_hi)

    # total MB via factorised combination in quadrature
    RcL_MB, RloL_MB, RhiL_MB = RMB_loss
    RcB_MB, RloB_MB, RhiB_MB = RMB_broad
    dL_MB = 0.5 * abs(RhiL_MB - RloL_MB)
    dB_MB = 0.5 * abs(RhiB_MB - RloB_MB)
    Rc_MB = RcL_MB * RcB_MB
    rel2_MB = (dL_MB / max(abs(RcL_MB), 1e-12)) ** 2 + (dB_MB / max(abs(RcB_MB), 1e-12)) ** 2
    dR_MB = Rc_MB * math.sqrt(rel2_MB)
    RMB_tot = (Rc_MB, Rc_MB - dR_MB, Rc_MB + dR_MB)

    return (labels,
            RL_c, RL_lo, RL_hi,
            RB_c, RB_lo, RB_hi,
            RT_c, RT_lo, RT_hi,
            RMB_loss, RMB_broad, RMB_tot)


# ============================================================
# Plot helpers (optional)
# ============================================================
import matplotlib.pyplot as plt


def _step_from_centers(x_cent, vals):
    """
    Given bin centers x_cent and values vals (same length, uniform spacing),
    build (x_edges, y_step) so that

        plt.step(x_edges, y_step, where="post")

    gives a flat segment per bin.

    Assumes constant bin width.
    """
    x_cent = np.asarray(x_cent, float)
    vals   = np.asarray(vals, float)
    assert x_cent.size == vals.size

    if x_cent.size > 1:
        dx = np.diff(x_cent)
        dx0 = dx[0]
        if not np.allclose(dx, dx0):
            raise ValueError("x_cent not uniformly spaced – can't stepify safely.")
    else:
        # single bin; choose arbitrary width 1
        dx0 = 1.0

    x_edges = np.concatenate(([x_cent[0] - 0.5 * dx0],
                              x_cent + 0.5 * dx0))
    y_step  = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step


def centrality_step_arrays(cent_bins, vals):
    """
    Build step-plot arrays from centrality bins [(0,20), ...] and
    values array of length len(cent_bins).

    Returns
    -------
    x_edges : array, shape (Nbins+1,)
    y_step  : array, shape (Nbins+1,)
    """
    vals = np.asarray(vals, float)
    assert len(vals) == len(cent_bins)

    edges = [cent_bins[0][0]] + [b for (_, b) in cent_bins]
    x_edges = np.array(edges, float)          # e.g. [0,20,40,60,80,100]
    y_step  = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step


def plot_RpA_vs_y_components_per_centrality(
    P, roots_GeV, qp_base,
    glauber: OpticalGlauber, cent_bins,
    y_edges, pt_range,
    show_components=("loss", "broad", "total"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    include_MB: bool = True,
    ncols: int = 3,
    step: bool = True,
    suptitle: str | None = None,
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Make a grid of subplots, one per centrality bin (+ optional MB),
    with different components (loss, broad, total) shown as
    different colours + legend entries.

    Layout:
        panels = centralities (and optional MB),
        curves  = components in show_components.
    """
    # Get only the components we actually need
    comp_list = tuple(sorted(set(show_components)))
    y_cent, bands, labels = rpa_band_vs_y(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        y_edges, pt_range,
        components=comp_list,
        q0_pair=q0_pair,
        p0_scale_pair=p0_scale_pair,
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        table_for_pp=None,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )

    # Which centrality tags to show as panels
    cent_tags = [f"{a}-{b}%" for (a, b) in cent_bins if (a, b) != (0, 100)]
    if include_MB:
        # add MB if present in bands
        any_comp = next(iter(bands.values()))
        Rc_dict_any = any_comp[0]
        if "MB" in Rc_dict_any:
            cent_tags.append("MB")

    # Colours & labels per component (consistent across panels)
    comp_colors = {
        "loss":  "C0",
        "broad": "C1",
        "total": "C3",
    }
    comp_labels = {
        "loss":  r"eloss",
        "broad": r"$p_T$ broad.",
        "total": r"eloss $\times$ $p_T$ broad.",
    }

    # Common pT-range note
    note = rf"$p_T\in[{pt_range[0]:.1f},{pt_range[1]:.1f}]$ GeV"

    # Figure / axes
    n_panels = len(cent_tags)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 3.2 * nrows),
        dpi=130,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for iax, (ax, tag) in enumerate(zip(axes, cent_tags)):
        for comp in show_components:
            Rc_dict, Rlo_dict, Rhi_dict = bands[comp]
            Rc  = np.asarray(Rc_dict[tag])
            Rlo = np.asarray(Rlo_dict[tag])
            Rhi = np.asarray(Rhi_dict[tag])

            col   = comp_colors.get(comp, "k")
            label = comp_labels.get(comp, comp) if iax == 0 else None

            if step:
                x_edges, y_c  = _step_from_centers(y_cent, Rc)
                _,       y_lo = _step_from_centers(y_cent, Rlo)
                _,       y_hi = _step_from_centers(y_cent, Rhi)

                ax.step(x_edges, y_c, where="post",
                        color=col, lw=1.6, label=label)
                ax.fill_between(
                    x_edges, y_lo, y_hi,
                    step="post", color=col, alpha=0.25, linewidth=0.0
                )
            else:
                ax.plot(y_cent, Rc, color=col, lw=1.6, label=label)
                ax.fill_between(
                    y_cent, Rlo, Rhi,
                    color=col, alpha=0.25, linewidth=0.0
                )

        # Horizontal R=1 line
        ax.axhline(1.0, color="k", ls=":", lw=0.8)

        # Panel title = centrality / MB
        if tag == "MB":
            ax.set_title("MB", fontsize=9)
        else:
            ax.set_title(tag, fontsize=9)

        # y-label only on leftmost panels
        if iax % ncols == 0:
            ax.set_ylabel(r"$R_{pA}(y)$")

        ax.set_xlabel(r"$y$")
        ax.grid(False)

        # Note inside each panel
        ax.text(
            0.03, 0.97, note,
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

        # Only first axis gets the legend
        if iax == 0:
            ax.legend(frameon=False, fontsize=7, loc="lower left")

    # Remove any unused axes (if n_panels < nrows*ncols)
    for j in range(n_panels, len(axes)):
        fig.delaxes(axes[j])

    if suptitle is not None:
        fig.suptitle(suptitle, y=0.99)
    plt.xlim(y_edges[0], y_edges[-1])
    fig.tight_layout(rect=[0, 0, 1, 0.96] if suptitle else None)
    return fig, axes


def plot_RpA_vs_pT_components_per_centrality(
    P, roots_GeV, qp_base,
    glauber: OpticalGlauber, cent_bins,
    pT_edges, y_range,
    show_components=("loss", "broad", "total"),
    q0_pair=(0.05, 0.09),
    p0_scale_pair=(0.9, 1.1),
    Ny_bin: int = 12, Npt_bin: int = 24,
    weight_kind: Literal["pp", "flat"] = "pp",
    weight_ref_y: float | str = "local",
    include_MB: bool = True,
    ncols: int = 3,
    step: bool = True,
    suptitle: str | None = None,
    ylabel: str = r"$R_{pA}(p_T)$",
    mb_weight_mode: Literal["exp", "optical", "custom"] = "exp",
    mb_c0: float = 0.25,
    mb_weights_custom: Dict[str, float] | None = None,
):
    """
    Grid of subplots: one panel per centrality bin (+ optional MB),
    curves = components (loss, broad, total) vs pT.

    Layout:
        panels  = centrality bins (+ MB)
        curves  = components in show_components
        y-range fixed for this figure.
    """
    # Only request the components we actually need
    comp_list = tuple(sorted(set(show_components)))
    pT_cent, bands, labels = rpa_band_vs_pT(
        P, roots_GeV, qp_base,
        glauber, cent_bins,
        pT_edges, y_range,
        components=comp_list,
        q0_pair=q0_pair,
        p0_scale_pair=p0_scale_pair,
        Ny_bin=Ny_bin, Npt_bin=Npt_bin,
        weight_kind=weight_kind,
        weight_ref_y=weight_ref_y,
        mb_weight_mode=mb_weight_mode,
        mb_c0=mb_c0,
        mb_weights_custom=mb_weights_custom,
    )

    # Panels: centrality tags (+ MB)
    cent_tags = [f"{a}-{b}%" for (a, b) in cent_bins if (a, b) != (0, 100)]
    if include_MB:
        any_comp = next(iter(bands.values()))
        Rc_dict_any = any_comp[0]
        if "MB" in Rc_dict_any:
            cent_tags.append("MB")

    # Colours & labels per component (consistent with y-plots)
    comp_colors = {
        "loss":  "C0",
        "broad": "C1",
        "total": "C3",
    }
    comp_labels = {
        "loss":  r"eloss",
        "broad": r"$p_T$ broad.",
        "total": r"eloss $\times$ $p_T$ broad.",
    }

    # Note inside each panel: y-range + pT-range
    note = (
        rf"${y_range[0]:.2f}<y<{y_range[1]:.2f}$" + "\n"
        rf"$p_T\in[{pT_edges[0]:.1f},{pT_edges[-1]:.1f}]$ GeV"
    )

    # Figure / axes layout
    n_panels = len(cent_tags)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 3.2 * nrows),
        dpi=130,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for iax, (ax, tag) in enumerate(zip(axes, cent_tags)):
        for comp in show_components:
            Rc_dict, Rlo_dict, Rhi_dict = bands[comp]

            Rc  = np.asarray(Rc_dict[tag])
            Rlo = np.asarray(Rlo_dict[tag])
            Rhi = np.asarray(Rhi_dict[tag])

            col   = comp_colors.get(comp, "k")
            # Only first panel gets legend labels
            label = comp_labels.get(comp, comp) if iax == 0 else None

            if step:
                x_edges, y_c  = _step_from_centers(pT_cent, Rc)
                _,       y_lo = _step_from_centers(pT_cent, Rlo)
                _,       y_hi = _step_from_centers(pT_cent, Rhi)

                ax.step(x_edges, y_c, where="post",
                        lw=1.6, color=col, label=label)
                ax.fill_between(
                    x_edges, y_lo, y_hi,
                    step="post", alpha=0.25, color=col, linewidth=0.0
                )
            else:
                ax.plot(pT_cent, Rc, lw=1.6, color=col, label=label)
                ax.fill_between(
                    pT_cent, Rlo, Rhi,
                    alpha=0.25, color=col, linewidth=0.0
                )

        # R=1 reference line
        ax.axhline(1.0, color="k", ls=":", lw=0.8)

        # Panel title = centrality / MB
        if tag == "MB":
            ax.set_title("MB", fontsize=9)
        else:
            ax.set_title(tag, fontsize=9)

        # Left column gets y-label
        if iax % ncols == 0:
            ax.set_ylabel(ylabel)

        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_xlim(pT_edges[0], pT_edges[-1])
        ax.grid(False)

        # Note inside panel
        ax.text(
            0.03, 0.97, note,
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

        # Only first axis gets legend
        if iax == 0:
            ax.legend(frameon=False, fontsize=7, loc="lower left")

    # Remove any unused axes
    for j in range(n_panels, len(axes)):
        fig.delaxes(axes[j])

    if suptitle is not None:
        fig.suptitle(suptitle, y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if suptitle else None)
    return fig, axes


def plot_RpA_vs_centrality_components_band(
    cent_bins, labels,
    RL_c=None, RL_lo=None, RL_hi=None, RMB_loss=None,
    RB_c=None, RB_lo=None, RB_hi=None, RMB_broad=None,
    RT_c=None, RT_lo=None, RT_hi=None, RMB_tot=None,
    show=("total",),                  # e.g. ("loss","broad","total")
    ax=None,
    ylabel=r"$R_{pA}(\mathrm{cent})$",
    note: str | None = None,
    system_label: str | None = None,  # e.g. r"$5.02$ TeV p+Pb"
):
    """
    Step-style RpA vs centrality, with optional bands for
    loss, broad, and total components, plus MB horizontal band.

    Parameters
    ----------
    cent_bins : list of (a,b) centrality edges.
    labels    : list of matching strings "a-b%".
    RL_*      : dict[lab] -> scalar, loss band (central, low, high).
    RB_*      : dict[lab] -> scalar, broad band.
    RT_*      : dict[lab] -> scalar, total band.
    RMB_*     : (Rc_MB, Rlo_MB, Rhi_MB) tuples per component.
    system_label : string appended to legend label for "total".
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.5), dpi=130)
    else:
        fig = ax.figure

    # colours & labels per component
    comp_color = {
        "loss":  "C1",
        "broad": "C2",
        "total": "C0",
    }
    comp_label = {
        "loss":  r"loss",
        "broad": r"broad",
        "total": r"total",
    }
    if system_label is not None:
        comp_label["total"] = system_label  # e.g. "5.02 TeV p+Pb"

    # helper to step-plot one component
    def _plot_comp(comp, Cc, Clo, Chi):
        vals_c  = np.array([Cc[lab]  for lab in labels])
        vals_lo = np.array([Clo[lab] for lab in labels])
        vals_hi = np.array([Chi[lab] for lab in labels])

        x_edges, y_c  = centrality_step_arrays(cent_bins, vals_c)
        _,       y_lo = centrality_step_arrays(cent_bins, vals_lo)
        _,       y_hi = centrality_step_arrays(cent_bins, vals_hi)

        col = comp_color[comp]
        lab = comp_label[comp]

        ax.step(x_edges, y_c, where="post", lw=2.0, color=col, label=lab)
        ax.fill_between(x_edges, y_lo, y_hi,
                        step="post", alpha=0.25, color=col, linewidth=0.0)

    # loss
    if "loss" in show and RL_c is not None:
        _plot_comp("loss", RL_c, RL_lo, RL_hi)

        if RMB_loss is not None:
            Rc_MB, Rlo_MB, Rhi_MB = RMB_loss
            x_band = np.array([cent_bins[0][0], cent_bins[-1][1]], float)
            ax.hlines(Rc_MB, x_band[0], x_band[1],
                      colors=comp_color["loss"], linestyles="--",
                      linewidth=1.2, label=r"MB loss")
            ax.fill_between(
                x_band,
                [Rlo_MB, Rlo_MB],
                [Rhi_MB, Rhi_MB],
                color=comp_color["loss"], alpha=0.12, linewidth=0.0,
            )

    # broad
    if "broad" in show and RB_c is not None:
        _plot_comp("broad", RB_c, RB_lo, RB_hi)

        if RMB_broad is not None:
            Rc_MB, Rlo_MB, Rhi_MB = RMB_broad
            x_band = np.array([cent_bins[0][0], cent_bins[-1][1]], float)
            ax.hlines(Rc_MB, x_band[0], x_band[1],
                      colors=comp_color["broad"], linestyles="--",
                      linewidth=1.2, label=r"MB broad")
            ax.fill_between(
                x_band,
                [Rlo_MB, Rlo_MB],
                [Rhi_MB, Rhi_MB],
                color=comp_color["broad"], alpha=0.12, linewidth=0.0,
            )

    # total (this is usually the main one, with darker MB)
    if "total" in show and RT_c is not None:
        _plot_comp("total", RT_c, RT_lo, RT_hi)

        if RMB_tot is not None:
            Rc_MB, Rlo_MB, Rhi_MB = RMB_tot
            x_band = np.array([cent_bins[0][0], cent_bins[-1][1]], float)
            ax.hlines(Rc_MB, x_band[0], x_band[1],
                      colors="k", linestyles="--",
                      linewidth=1.6, label=r"MB total")
            ax.fill_between(
                x_band,
                [Rlo_MB, Rlo_MB],
                [Rhi_MB, Rhi_MB],
                color="gray", alpha=0.30, linewidth=0.0,
            )

    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("centrality [%]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(cent_bins[0][0], cent_bins[-1][1])
    ax.grid(False)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    if note is not None:
        ax.text(
            0.03, 0.97, note,
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

    return fig, ax
