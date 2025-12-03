# quenching.py
# Coherent energy-loss (Arleo–Peigné) kernels + stable, fast integrators
# Units:   L, lp in fm;  qhat0 in GeV^2/fm;  ℓ², Λ_p², mT², pT² in GeV²;  αs dimensionless
# Physics: P̂ is the derivative of the Sudakov factor built from the single-gluon spectrum
#          with ℓ² = q̂(x) L,  Λ_p² = max(l_p^2 = q̂(x) L_p , λ_QCD²).  See notes in docstrings below.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math, numpy as np
from functools import lru_cache

# ----------------- numerics --------------------------------------------------
try:
    import mpmath as mp
    mp.mp.dps = 60
    def Li2(z: float) -> float:            # real dilogarithm
        return float(mp.polylog(2, z).real)
except Exception:
    # stable series / reflection fallback
    def Li2(z: float) -> float:
        x = -float(z)
        if x <= 1.0:
            s, t, k = 0.0, -x, 1
            while abs(t) > 1e-16 and k < 20000:
                s += t/(k*k); k += 1; t *= -x
            return s
        ln = math.log(x)
        return -math.pi**2/6.0 - 0.5*ln*ln - Li2(-1.0/x)

HBARC = 0.1973269804   # GeV*fm
M_PROTON = 0.938       # GeV
LOG2 = math.log(2.0)
Z_FLOOR = 1e-12

@lru_cache(maxsize=None)
def _gl_nodes_cached(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(n))
    xm, xc = 0.5*(b-a), 0.5*(b+a)
    return (xc + xm*x, xm*w)

@lru_cache(maxsize=None)
def _phi_nodes_cached(nphi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p, wp = _gl_nodes_cached(0.0, 2.0*math.pi, nphi)
    w = wp/(2.0*math.pi)  # average over φ (not integrate)
    return p, w, np.cos(p), np.sin(p)

# ----------------- parameters ------------------------------------------------
@dataclass(frozen=True)
class QuenchParams:
    r"""
    Quenching inputs (AP):
      qhat0   : baseline \hat{q}(x=1e-2) at ρ=ρ0 (GeV^2/fm)
      lp_fm   : proton formation length entering Λ_p^2
      LA_fm   : path length through A (fm) (use Glauber L_eff)
      LB_fm   : path length through B (fm) (AA only; =LA for pA in this model)
      lambdaQCD : infrared scale [GeV] setting Λ_p² floor via λ_QCD²
      alpha_of_mu(mu): αs(μ) provider (constant or running)
      alpha_scale: μ choice for αs in P̂:
           - "mT"  (default; matches your current C++)
           - "dpt" (older python choice: μ = ΔpT_side)
    """
    nc: int = 3
    qhat0: float = 0.075
    lp_fm: float = 1.5
    LA_fm: float = 10.0
    LB_fm: float = 10.0
    lambdaQCD: float = 0.308
    roots_GeV: float = 5023.0
    alpha_of_mu: Callable[[float], float] = lambda mu: 0.5
    alpha_scale: str = "mT"  # "mT" | "dpt"

# ----------------- small-x ingredients --------------------------------------
def qhat_of_x(x: np.ndarray | float, qhat0: float) -> np.ndarray | float:
    r""" \hat{q}(x) = qhat0 * (1e-2/x)^0.3    (dimension: GeV^2/fm) """
    xx = np.maximum(np.asarray(x, float), 1e-12)
    return qhat0 * (1.0e-2/xx)**0.3

def xA0_from_L(L_fm: float, m_p: float = M_PROTON) -> float:
    """ Coherence bound: x0(L) = 1 / (2 m_p L/ħc).  L[fm] → L/(ħc)[GeV^-1] → x0 dimensionless. """
    return 1.0/(2.0*m_p*(L_fm/HBARC))

# ----------------- rapidity window ------------------------------------------
def dymax(y: float, y_max_of_pt: float) -> float:
    """ δy_max(y) = min(ln 2, y_max(pT) - y) with floor at 0. """
    r = min(LOG2, max(y_max_of_pt - y, 0.0))
    return max(r, 0.0)

# ----------------- scales ℓ² and Λ_p² ---------------------------------------
def _l2(qpar: QuenchParams, x: np.ndarray | float, L_fm: float) -> np.ndarray | float:
    r""" ℓ² = \hat{q}(x) L   [GeV²] """
    return qhat_of_x(x, qpar.qhat0) * L_fm

def _Lambda_p2(qpar: QuenchParams, x: np.ndarray | float) -> np.ndarray | float:
    """
    Correct: Λ_p² = max(ℓ_p², λ_QCD²) with ℓ_p² = q̂(x) * ℓ_p   (GeV²)
    """
    qx = qhat_of_x(x, qpar.qhat0)                  # GeV²/fm
    lp2 = np.asarray(qx, float) * float(qpar.lp_fm)  # GeV²
    lam2 = float(qpar.lambdaQCD) ** 2               # GeV²
    return np.maximum(lp2, lam2)

def _soft_pos(x: float, width: float = 1e-3) -> float:
    # smooth max(x,0) ~ eliminates cusps when lA2 ≈ lp2
    # width ~ O(1e-3) in GeV^2 works well; pure numerical, no physics change at percent level
    return 0.5*(x + math.sqrt(x*x + width*width))
    
def dpt_L_fm(qpar: QuenchParams, x: float, L_fm: float) -> float:
    """
    ΔpT = sqrt( max(ℓ_A² - ℓ_p², 0) ), with ℓ_A² = q̂(x)*L_A, ℓ_p² = q̂(x)*L_p; L_p = 1.5 fm
    """
    qx  = float(qhat_of_x(x, qpar.qhat0))
    lA2 = qx * float(L_fm)
    lp2 = qx * float(qpar.lp_fm)
    # return math.sqrt(max(lA2 - lp2, 0.0))
    return math.sqrt(_soft_pos(lA2 - lp2))

def dpt_side(qpar: QuenchParams, x: float, L_fm: float) -> float:
    """
    Side-specific ΔpT used in φ-averaging.
    """
    return dpt_L_fm(qpar, x, L_fm)

# ----------------- αs(μ) selection ------------------------------------------
def _alpha_mu(qpar: QuenchParams, side: str, y: float, pt: float, mT: float, x: float) -> float:
    """
    μ choice:
      "mT" : μ = mT  (robust at low pT; matches your C++)
      "dpt": μ = ΔpT_side
    """
    if qpar.alpha_scale.lower() == "mt":
        mu = float(mT) # max(float(mT), 1.01*qpar.lambdaQCD)
    else:
        L = qpar.LA_fm if side == "A" else qpar.LB_fm
        mu = max(dpt_side(qpar, x, L), 1.01*qpar.lambdaQCD)
    return float(qpar.alpha_of_mu(mu))

# ----------------- quenching kernel P̂(z; …) ---------------------------------
def _Phat_core(z: float, Mperp2: float, l2: float, Lp2: float, a: float, Nc: int) -> float:
    """
    P̂(z) =  (αs Nc / 2π) * d/dz [ ln(1 + ℓ²/(z² M⊥²)) - ln(1 + Λ_p²/(z² M⊥²)) ]
             × exp{ (αs Nc / 2π) [ Li2( -ℓ²/(z² M⊥²) ) - Li2( -Λ_p²/(z² M⊥²) ) ] }
    Notes:
      • z = e^{δy} - 1 ≥ 0.  We clamp to Z_FLOOR to avoid z=0.
    """
    if not (z > Z_FLOOR):
        z = Z_FLOOR
    if l2 <= Lp2:
        return 0.0
    inv  = 1.0/(z*z*Mperp2)
    expo = a*Nc*(Li2(-l2*inv) - Li2(-Lp2*inv)) / (2.0*math.pi)
    expo = max(min(expo, 700.0), -700.0)                        
    deriv_f = 2.0*(math.log1p(l2*inv) - math.log1p(Lp2*inv))/z
    val = (a * math.exp(expo) * Nc * deriv_f) / (2.0*math.pi)
    return val if (val > 0.0 and math.isfinite(val)) else 0.0

def _Phat_core_vec(z: np.ndarray, Mperp2: float, l2: float, Lp2: float, a: float, Nc: int) -> np.ndarray:
    z = np.maximum(np.asarray(z, float), Z_FLOOR)
    if l2 <= Lp2:
        return np.zeros_like(z)
    inv  = 1.0/(z*z*Mperp2)
    expo = (a*Nc/(2.0*math.pi))*(np.vectorize(Li2)(-l2*inv) - np.vectorize(Li2)(-Lp2*inv))
    expo = np.clip(expo, -700.0, 700.0)
    deriv_f  = 2.0*(np.log1p(l2*inv) - np.log1p(Lp2*inv))/z
    val = (a*np.exp(expo)*Nc*deriv_f)/(2.0*math.pi)
    val[~np.isfinite(val) | (val < 0.0)] = 0.0
    return val

def z_from_dy(dy: float) -> float:
    """z = e^{δy} - 1"""
    return math.expm1(float(dy))
    
def PhatA_vec(z_arr: np.ndarray, y: float, pt: float, mT: float, xA: float, qpar: QuenchParams) -> np.ndarray:
    l2  = float(_l2(qpar, xA, qpar.LA_fm))
    Lp2 = float(_Lambda_p2(qpar, xA))
    a   = _alpha_mu(qpar, "A", y, pt, mT, xA)
    return _Phat_core_vec(z_arr, mT*mT, l2, Lp2, a, qpar.nc)

def PhatB_vec(z_arr: np.ndarray, y: float, pt: float, mT: float, xB: float, qpar: QuenchParams) -> np.ndarray:
    l2  = float(_l2(qpar, xB, qpar.LB_fm))
    Lp2 = float(_Lambda_p2(qpar, xB))
    a   = _alpha_mu(qpar, "B", y, pt, mT, xB)
    return _Phat_core_vec(z_arr, mT*mT, l2, Lp2, a, qpar.nc)

def PhatA(z: float, y: float, pt: float, mT: float, xA: float, qpar: QuenchParams) -> float:
    return float(PhatA_vec(np.array([z]), y, pt, mT, xA, qpar)[0])

def PhatB(z: float, y: float, pt: float, mT: float, xB: float, qpar: QuenchParams) -> float:
    return float(PhatB_vec(np.array([z]), y, pt, mT, xB, qpar)[0])

# ----------------- kinematic helpers ----------------------------------------
def shifted_pT_pA(pt: float, dpta: float, phiA: float) -> float:
    return math.sqrt(pt*pt + dpta*dpta + 2.0*pt*dpta*math.cos(phiA))

def shifted_pT_AB(pt: float, dptb: float, dpta: float, phiB: float, phiA: float) -> float:
    cA, sA = math.cos(phiA), math.sin(phiA)
    cB, sB = math.cos(phiB), math.sin(phiB)
    comp1 = pt - dpta*cA - dptb*cB
    comp2 =       dpta*sA + dptb*sB
    return math.sqrt(comp1*comp1 + comp2*comp2)

# ----------------- public cross-section integrals ----------------------------
def pA_cross_section(y: float, pt: float, mT: float,
                     xA: float, xB_unused: float, y_max_pt: float,
                     dsig_pp: Callable[[float, np.ndarray], np.ndarray],
                     qpar: QuenchParams, Ny: int = 80, Nphi: int = 48,
                     adaptive: bool = False) -> float:
    """
    σ_pA(y,pt) = ∫_{0}^{δy_max(+y)} d(δy) ∫ dφ_A/(2π)  P̂_A(z) · σ_pp(y+δy, |p⃗_T − Δp⃗_T^A|)
    z = e^{δy} − 1;   Δp_T^A = sqrt( max(ℓ_A² - l_p², 0) )
    """
    dym = dymax(+y, y_max_pt)
    if dym <= 1e-12 or _l2(qpar, xA, qpar.LA_fm) <= _Lambda_p2(qpar, xA):
        return float(dsig_pp(y, np.array([pt]))[0])

    if adaptive:
        Ny = max(16, int(round(Ny * min(1.0, dym / LOG2))))

    dy_nodes, dy_w = _gl_nodes_cached(0.0, dym, Ny)
    phi, wphi, cphi, sphi = _phi_nodes_cached(Nphi)
    dpta = dpt_side(qpar, xA, qpar.LA_fm)

    pshift_phi = np.sqrt((pt - dpta*cphi)**2 + (dpta*sphi)**2)  # (Nphi,)
    zA = np.expm1(dy_nodes)
    PhA = PhatA_vec(zA, y, pt, mT, xA, qpar)                    # (Ny,)

    acc = 0.0
    for δy, wy, ph in zip(dy_nodes, dy_w, PhA):
        if ph <= 0.0:
            continue
        # rapidity shift → y + δy
        sig = dsig_pp(y + float(δy), pshift_phi)             # (Nphi,)
        avg = float(np.dot(sig, wphi))
        acc += wy * ph * avg
    return float(acc)

def AB_cross_section(y: float, pt: float, mT: float,
                     xA: float, xB: float, y_max_pt: float,
                     dsig_pp: Callable[[float, np.ndarray], np.ndarray],
                     qpar: QuenchParams, Ny: int = 48, Nphi: int = 24,
                     adaptive: bool = True) -> float:
    """
    σ_AB(y,pt) = ∬ d(δy_A) d(δy_B) ∫∫ dφ_A dφ_B  P̂_A(z_A) P̂_B(z_B)
                 × σ_pp( y+δy_B−δy_A, |p⃗_T − Δp⃗_T^A − Δp⃗_T^B| )
    δy_A ∈ [0, δy_max(−y)],  δy_B ∈ [0, δy_max(+y)]
    """
    dymA = dymax(-y, y_max_pt)
    dymB = dymax(+y, y_max_pt)
    if (dymA <= 1e-12 or dymB <= 1e-12 or
        _l2(qpar, xA, qpar.LA_fm) <= _Lambda_p2(qpar, xA) or
        _l2(qpar, xB, qpar.LB_fm) <= _Lambda_p2(qpar, xB)):
        return float(dsig_pp(y, np.array([pt]))[0])

    if adaptive:
        sclA = min(1.0, dymA/LOG2); sclB = min(1.0, dymB/LOG2)
        Ny = max(16, int(round(Ny * 0.5*(sclA+sclB))))

    dyA, wA = _gl_nodes_cached(0.0, dymA, Ny)
    dyB, wB = _gl_nodes_cached(0.0, dymB, Ny)
    phi, wphi, cphi, sphi = _phi_nodes_cached(Nphi)
    dpta = dpt_side(qpar, xA, qpar.LA_fm); dptb = dpt_side(qpar, xB, qpar.LB_fm)

    cA, sA = cphi[:,None], sphi[:,None]
    cB, sB = cphi[None,:], sphi[None,:]

    zA = np.expm1(dyA); zB = np.expm1(dyB)
    PhA = PhatA_vec(zA, y, pt, mT, xA, qpar)  # (Ny,)
    PhB = PhatB_vec(zB, y, pt, mT, xB, qpar)  # (Ny,)

    acc = 0.0
    for δyB, wBj, phB in zip(dyB, wB, PhB):
        if phB <= 0.0:
            continue
        comp1 = (pt - dpta*cA - dptb*cB)       # (Nphi,Nphi)
        comp2 = (      dpta*sA + dptb*sB)      # (Nphi,Nphi)
        pshift = np.sqrt(comp1*comp1 + comp2*comp2).ravel()
        w2 = (wphi[:,None]*wphi[None,:]).ravel()
        for δyA, wAi, phA in zip(dyA, wA, PhA):
            if phA <= 0.0:
                continue
            sig = dsig_pp(y + float(δyB) - float(δyA), pshift)
            avg = float(np.dot(sig, w2))
            acc += wBj * wAi * phB * phA * avg
    return float(acc)

# ----------------- diagnostics (single point) --------------------------------
def diagnostics_row(*, y: float, pt: float, mT: float, xA: float, xB: float,
                    y_max_pt: float, qpar: QuenchParams) -> dict:
    dym = dymax(+y, y_max_pt)
    z_med = math.expm1(0.5*dym)
    return dict(
        zA_med=z_med,
        PhA_med=PhatA(z_med, y, pt, mT, xA, qpar),
        zB_med=z_med,
        PhB_med=PhatB(z_med, y, pt, mT, xB, qpar),
    )
