"""
QCD running coupling α_s(μ) with selectable loop order (1–4),
faithful to the C++ logic and normalization used in your code base.

Key points:
- Normalization: α_s(muRef) = alphaRef (default: muRef=1.5 GeV, alphaRef=0.326),
  matching your C++ (alphasRef = 0.326/(4π) and muRef = 1.5).
- Λ_QCD is used to set the proximity to the Landau pole. In perturbation theory,
  α_s(μ=Λ_QCD) → ∞. By default we *freeze* the output to alpha_max=1.0 to avoid NaNs,
  mimicking your C++ safeguard around the pole. You can set freeze=False to disable it.
- Loop orders: 1, 2, 3, 4 (MS-bar β-function coefficients for Nf=3 by default).
- Methods:
    'ode'   : integrates da/d(ln μ) = -2 Σ_{k=0}^{ℓ-1} b_k a^{k+2}, a ≡ α_s/(4π)
              (robust and accurate for all μ > 0, except right at the pole).
    'asym'  : multi-loop asymptotic series in t = 2 ln(μ/Λ), accurate at large μ,
              with 1-loop fallback near the pole.

Refs (standard):
- Beta coefficients to 4-loops in MS-bar (e.g. van Ritbergen, Vermaseren, Larin; Czakon 2004).
- Physical fact: α_s diverges at μ=Λ_QCD (Landau pole). Your C++ clamps it numerically.

Author: you + ChatGPT (finalized)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Callable, Dict

# ------------------------ Beta-function coefficients -------------------------

def _beta_coeffs(Nf: float, loops: int) -> Dict[str, float]:
    """
    Return {b0,b1,b2,b3} with higher ones zeroed as needed.
    Defaults match your C++ for Nf=3.
    """
    zeta3 = 1.2020569031595942
    b0 = (11.0 - 2.0*Nf/3.0)
    b1 = (102.0 - 38.0*Nf/3.0)
    b2 = (2857.0/2.0 - (5033.0/18.0)*Nf + (325.0/54.0)*Nf*Nf)
    b3 = ((149753.0/6.0 + 3564.0*zeta3)
          - (1078361.0/162.0 + 6508.0*zeta3/27.0)*Nf
          + (50065.0/162.0 + 6472.0*zeta3/81.0)*Nf*Nf
          + 1093.0*Nf*Nf*Nf/729.0)
    if loops <= 1: return dict(b0=b0, b1=0.0, b2=0.0, b3=0.0)
    if loops == 2: return dict(b0=b0, b1=b1, b2=0.0, b3=0.0)
    if loops == 3: return dict(b0=b0, b1=b1, b2=b2, b3=0.0)
    return dict(b0=b0, b1=b1, b2=b2, b3=b3)  # loops >= 4

# ------------------------ Config dataclass -----------------------------------

@dataclass(frozen=True)
class AlphaSConfig:
    """
    Configuration for α_s(μ).

    Parameters
    ----------
    Nf : float
        Number of active flavors (default 3 for your quarkonia kinematics).
    loops : int
        Loop order (1,2,3,4).
    muRef : float
        Reference scale in GeV where α_s is fixed to alphaRef.
    alphaRef : float
        α_s(muRef) value (default 0.326 per your C++).
    LambdaQCD : float
        Λ_QCD in GeV (default 0.308, matching your C++).
    method : {'ode','asym'}
        ODE in ln μ (robust) or asymptotic series in t=2 ln(μ/Λ) (fast for large μ).
    freeze : bool
        If True, freeze α_s to alpha_max near the Landau pole and cap to [0, alpha_max].
    alpha_max : float
        Maximum allowed α_s (used when freeze=True).
    """
    Nf: float = 3.0
    loops: int = 4
    muRef: float = 1.5
    alphaRef: float = 0.326
    LambdaQCD: float = 0.308
    method: Literal["ode", "asym"] = "ode"
    freeze: bool = True
    alpha_max: float = 1.0

# ------------------------ Asymptotic series (1–4 loops) ----------------------

def _asym_series(mu: float, cfg: AlphaSConfig) -> float:
    """
    Multi-loop asymptotic expansion:
      a ≡ α_s/(4π), t = 2 ln(μ/Λ), ℓ=cfg.loops
      1-loop : a = 1/(b0 t)
      2-loop : a = 1/(b0 t) * [1 - b1/b0^2 * (ln t)/t]
      3,4-loop: include higher terms as in your C++ (Li expansion in logs of t).
    """
    b = _beta_coeffs(cfg.Nf, cfg.loops)
    b0, b1, b2, b3 = b["b0"], b["b1"], b["b2"], b["b3"]

    mu = max(mu, 1.01*cfg.LambdaQCD)  # avoid log of ≤1
    t = 2.0*math.log(mu/cfg.LambdaQCD)
    if not math.isfinite(t) or t <= 1e-12:
        # near the pole fallback to 1-loop floor
        a = 1.0/(b0*max(t, 1e-12))
        val = 4.0*math.pi*a
        if cfg.freeze:
            val = min(max(val, 0.0), cfg.alpha_max)
        return val

    # 1-loop
    a1 = 1.0/(b0*t)
    if cfg.loops == 1:
        val = 4.0*math.pi*a1
        return min(max(val, 0.0), cfg.alpha_max) if cfg.freeze else val

    # 2-loop
    L = math.log(t)
    a2 = a1*(1.0 - (b1/(b0*b0))*L/t)
    if cfg.loops == 2:
        val = 4.0*math.pi*a2
        return min(max(val, 0.0), cfg.alpha_max) if cfg.freeze else val

    # 3-loop
    b02 = b0*b0
    a3 = a1*(1.0
             - (b1/b02)*L/t
             + ( (b1*b1)*(L*L - L - 1.0) + b0*b2 )/(b02*b02*t*t) )
    if cfg.loops == 3:
        val = 4.0*math.pi*a3
        return min(max(val, 0.0), cfg.alpha_max) if cfg.freeze else val

    # 4-loop (matches your C++ structure)
    b04 = b02*b02; b06 = b02*b04; b14 = (b1*b1)*(b1*b1)
    t2, t3, t4 = t*t, t*t*t, t*t*t*t
    L2, L3, L4 = L*L, L*L*L, L*L*L*L
    res = 1.0
    res += -(b1)*L/(b02*t)
    res += ( (b1*b1)*(L2 - L - 1.0) + b0*b2 )/(b04*t2)
    res += ( (b1*b1*b1)*(-2*L3 + 5*L2 + 4*L - 1.0) - 6*b0*b1*b2*L + b02*b3 )/(2.0*b06*t3)
    # the b4 term is 0 in your C++; we keep the same
    a4 = a1*res
    val = 4.0*math.pi*a4
    return min(max(val, 0.0), cfg.alpha_max) if cfg.freeze else val

# ------------------------ ODE integrator in ln μ (robust) --------------------

def _rhs_a(lnmu: float, a: float, b0: float, b1: float, b2: float, b3: float) -> float:
    """
    da/d(ln μ) = -2 [ b0 a^2 + b1 a^3 + b2 a^4 + b3 a^5 ]
    (works for any loop order by zeroing the unused coefficients)
    """
    a2 = a*a
    s = (b0*a2
         + b1*a2*a
         + b2*a2*a2
         + b3*a2*a2*a)
    return -2.0*s

def _rk4(a: float, lnmu: float, h: float, b0: float, b1: float, b2: float, b3: float) -> float:
    k1 = _rhs_a(lnmu,       a,                 b0,b1,b2,b3)
    k2 = _rhs_a(lnmu+0.5*h, a + 0.5*h*k1,      b0,b1,b2,b3)
    k3 = _rhs_a(lnmu+0.5*h, a + 0.5*h*k2,      b0,b1,b2,b3)
    k4 = _rhs_a(lnmu+h,     a + h*k3,          b0,b1,b2,b3)
    return a + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def _ode_solve(mu: float, cfg: AlphaSConfig) -> float:
    """
    Integrate from (muRef, aRef=alphaRef/(4π)) to target μ in steps of ln μ.
    This mirrors your C++ ODE but is more stable (log-step) and supports any loop order.
    """
    b = _beta_coeffs(cfg.Nf, cfg.loops)
    b0,b1,b2,b3 = b["b0"], b["b1"], b["b2"], b["b3"]

    # Guard rails near the pole
    mu = max(float(mu), 1.01*cfg.LambdaQCD)
    if not math.isfinite(mu):
        mu = 1.01*cfg.LambdaQCD

    a = cfg.alphaRef/(4.0*math.pi)   # initial condition a(muRef)
    ln0 = math.log(cfg.muRef)
    ln1 = math.log(mu)
    if abs(ln1 - ln0) < 1e-15:
        val = 4.0*math.pi*a
        return min(max(val, 0.0), cfg.alpha_max) if cfg.freeze else val

    # Step count proportional to |Δ ln μ|
    nstep = max(30, int(300*abs(ln1 - ln0)))
    h = (ln1 - ln0)/nstep
    ln = ln0
    for _ in range(nstep):
        a = _rk4(a, ln, h, b0,b1,b2,b3)
        # keep a non-negative and finite
        if not math.isfinite(a) or a < 0.0:
            # fallback to asymptotic series at this μ
            return _asym_series(math.exp(ln), cfg)
        ln += h

    val = 4.0*math.pi*a
    if not math.isfinite(val) or val < 0.0:
        val = _asym_series(mu, cfg)
    if cfg.freeze:
        val = min(max(val, 0.0), cfg.alpha_max)
    return val

# ------------------------ Public API -----------------------------------------

@lru_cache(maxsize=8192)
def alpha_s(mu: float,
            Nf: float = 3.0,
            loops: int = 4,
            muRef: float = 1.5,
            alphaRef: float = 0.326,
            LambdaQCD: float = 0.308,
            method: Literal["ode","asym"] = "ode",
            freeze: bool = True,
            alpha_max: float = 1.0) -> float:
    """
    α_s(μ) with selectable loop order and method.

    Notes:
    - If you expect α_s ≈ 0.326, evaluate at μ = muRef (default 1.5 GeV).
      At μ = Λ_QCD, perturbative α_s is not finite (Landau pole).
    """
    cfg = AlphaSConfig(Nf=Nf, loops=loops, muRef=muRef, alphaRef=alphaRef,
                       LambdaQCD=LambdaQCD, method=method,
                       freeze=freeze, alpha_max=alpha_max)
    mu = float(mu)

    if method == "asym":
        return _asym_series(mu, cfg)
    return _ode_solve(mu, cfg)

def alpha_s_provider(*,
                     mode: Literal["constant","running"] = "running",
                     alpha0: float = 0.5,
                     Nf: float = 3.0,
                     loops: int = 4,
                     muRef: float = 1.5,
                     alphaRef: float = 0.326,
                     LambdaQCD: float = 0.308,
                     method: Literal["ode","asym"] = "ode",
                     freeze: bool = True,
                     alpha_max: float = 1.0) -> Callable[[float], float]:
    """
    Returns a callable μ ↦ α_s(μ).

    Examples:
      a_run  = alpha_s_provider(mode="running", loops=4, method="ode")
      a_1L   = alpha_s_provider(mode="running", loops=1, method="ode")
      a_cst  = alpha_s_provider(mode="constant", alpha0=0.326)
    """
    if mode == "constant":
        c = float(alpha0)
        return lambda mu: c
    # running:
    def f(mu: float) -> float:
        return alpha_s(mu, Nf=Nf, loops=loops, muRef=muRef, alphaRef=alphaRef,
                       LambdaQCD=LambdaQCD, method=method, freeze=freeze,
                       alpha_max=alpha_max)
    return f

# ------------------------ Quick self-test (optional) -------------------------

if __name__ == "__main__":
    # Expect ~0.326 at muRef for any loop/method
    for L in (1,2,3,4):
        val = alpha_s(1.5, loops=L, method="ode")
        print(f"loops={L}: alpha_s(1.5 GeV) ≈ {val:.3f}")
    # Near Λ_QCD we freeze ~1.0 by default
    print("alpha_s(Λ_QCD) with freeze:", alpha_s(0.308))
