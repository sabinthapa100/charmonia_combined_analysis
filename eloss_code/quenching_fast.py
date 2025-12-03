# quenching_fast.py
# Robust, GPU-ready coherent energy-loss (Arleo–Peigné) + pT broadening.
# Minimal-change hardening (finalized):
#  * δy upper-limit respects σ_pp table y-range (avoids “flat vs y” when clamped)
#  * Peripheral-safe Cronin taper → 0 as L→lp (fixes low-L_eff artefacts)
#  * Thin-phase auto-Ny tightened; exp-mapping on dy with δ-peak p0 piece
#  * α(μ) hook used consistently inside Φ(z) (mT/pT/fixed)
#  * AB: one-sided fallback if only A or B is active (no premature σ_pp return)
#  * Binning weights: default y=0; pass weight_ref_y="local" to use local y
#  * Units are consistent:  q̂0 in GeV^2/fm; ℓ^2=q̂·L [GeV^2]; Λ_p^2=max(λ_QCD^2, q̂·lp)

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple, Literal, Union
import math, numpy as np

# ----- torch / torchquad (optional acceleration) ----------------------------
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from torchquad import GaussLegendre as TQ_GaussLegendre
    _HAS_TQ = True
except Exception:
    _HAS_TQ = False

# ----- constants / numerics --------------------------------------------------
M_PROTON = 0.938                 # GeV
LOG2     = math.log(2.0)
Z_FLOOR  = 1e-12
_PI2_12  = (math.pi**2)/12.0
DY_EPS   = 1e-6
HBARC    = 0.1973269804          # GeV·fm

def _torch_device(dev: Optional[str] = None):
    if not _HAS_TORCH:
        return None
    if isinstance(dev, str):
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- GL nodes (NumPy / Torch) ---------------------------------------------
def _gl_nodes_np(a: float, b: float, n: int):
    x, w = np.polynomial.legendre.leggauss(int(n))
    xm, xc = 0.5*(b-a), 0.5*(b+a)
    return xc + xm*x, xm*w

def _gl_nodes_torch(a: float, b: float, n: int, device):
    x, w = np.polynomial.legendre.leggauss(int(n))
    xm, xc = 0.5*(b-a), 0.5*(b+a)
    xt = torch.tensor(xc + xm*x, dtype=torch.float64, device=device)
    wt = torch.tensor(xm*w,      dtype=torch.float64, device=device)
    return xt, wt

def _phi_nodes_gl_torch(nphi: int, device):
    phi, w = _gl_nodes_torch(0.0, 2.0*math.pi, nphi, device)
    wbar   = w/(2.0*math.pi)  # normalized average over 2π
    return phi, wbar, torch.cos(phi), torch.sin(phi)

# ----- Dilog Li2 (real) ------------------------------------------------------
def _li2_series_unit_np(x: np.ndarray, K: int = 256) -> np.ndarray:
    k = np.arange(1, K+1, dtype=float)
    return np.sum(((-x[:,None])**k)/(k*k), axis=1)

def Li2_np(z: np.ndarray) -> np.ndarray:
    x = -z
    res = np.empty_like(x, dtype=float)
    m_small = (x <= 1.0)
    if np.any(m_small):
        res[m_small] = _li2_series_unit_np(x[m_small])
    if np.any(~m_small):
        xs = x[~m_small]
        ln = np.log(xs)
        inv = -1.0/xs
        res_large = -_PI2_12 - 0.5*ln*ln - _li2_series_unit_np(inv)
        res[~m_small] = res_large
    return res

def _li2_series_unit_t(x, K: int = 256):
    k = torch.arange(1, K+1, device=x.device, dtype=torch.float64)
    term = torch.pow(-x.unsqueeze(-1), k)/(k*k)
    return term.sum(dim=-1)

def Li2_torch(z):
    x = -z
    res = torch.empty_like(x, dtype=torch.float64)
    m_small = (x <= 1.0)
    if m_small.any():
        res[m_small] = _li2_series_unit_t(x[m_small])
    if (~m_small).any():
        xs = x[~m_small]
        ln = torch.log(xs)
        inv = -1.0/xs
        res_large = -_PI2_12 - 0.5*ln*ln - _li2_series_unit_t(inv)
        res[~m_small] = res_large
    return res

# ----- Parameters -------------------------------------------------------------
@dataclass(frozen=True)
class QuenchParams:
    qhat0: float = 0.075    # GeV^2/fm at x=1e-2
    lp_fm: float = 1.5      # fm
    LA_fm: float = 10.0     # fm
    LB_fm: float = 10.0     # fm
    lambdaQCD: float = 0.25 # GeV
    roots_GeV: float = 8160.0
    alpha_of_mu: Callable[[float], float] = lambda mu: 0.5
    alpha_scale: Literal["mT","pT","fixed"] = "mT"
    Nc: int = 3
    use_hard_cronin: bool = True
    mapping: Literal["exp","linear"] = "exp"
    device: Optional[str] = None

# ----- helpers (NumPy) -------------------------------------------------------
def xA0_from_L(L_fm: float) -> float:
    # x0 = ħc / (2 m_p L)
    return HBARC / (2.0 * M_PROTON * max(L_fm, 1e-12))

def qhat_of_x(qp: QuenchParams, xA: float) -> float:
    # qhat(x) = qhat0 * (1e-2/x)^0.3  [GeV^2/fm]
    x_eff = max(min(xA, xA0_from_L(qp.LA_fm)), 1e-9)
    return qp.qhat0 * (1e-2/x_eff)**0.3

def _l2(qp: QuenchParams, xA: float, L_fm: float) -> float:
    # ℓ^2 = qhat(x) * L  [GeV^2]
    return qhat_of_x(qp, xA) * L_fm

# ----- Torch qhat / ℓ² / Λp² / αs / ΔpT -------------------------------------
def _qhat_t(qhat0: float, x):
    x = torch.clamp(x, min=1e-12)
    return qhat0 * torch.pow(1e-2/x, 0.3)

def _l2_t(qp: QuenchParams, x, L_fm: float):
    return _qhat_t(qp.qhat0, x) * L_fm

def _Lambda_p2_t(qp: QuenchParams, x):
    lp2 = _qhat_t(qp.qhat0, x) * qp.lp_fm
    lam2= torch.tensor(qp.lambdaQCD**2, dtype=torch.float64, device=lp2.device)
    return torch.maximum(lp2, lam2)

def _alpha_mu_t(qp: QuenchParams, mT: float, pT: float):
    if qp.alpha_scale == "mT":
        mu = float(mT)
    elif qp.alpha_scale == "pT":
        mu = max(pT, 0.5)
    else:
        mu = 1.5
    return float(qp.alpha_of_mu(mu))

# Peripheral-safe Cronin taper
def _dpt_from_xL_t(qp: QuenchParams, x, L_fm: float, hard=True):
    qL  = _qhat_t(qp.qhat0, x) * L_fm
    qlp = _qhat_t(qp.qhat0, x) * qp.lp_fm
    base = torch.clamp(qL - qlp, min=0.0) if hard else \
           0.5*((qL-qlp) + torch.sqrt((qL-qlp)**2 + 1e-12))
    # taper(L) = sqrt(1 - exp(-(L-lp)/lp)) → 0 as L→lp
    L   = torch.tensor(L_fm, dtype=torch.float64, device=base.device)
    xL  = torch.clamp(L - qp.lp_fm, min=0.0)
    taper = torch.sqrt(torch.clamp(1.0 - torch.exp(-xL / max(qp.lp_fm, 1e-9)), 0.0, 1.0))
    return torch.sqrt(base) * taper

# ----- kinematics -------------------------------------------------------------
def y_max(roots_GeV: float, mT: float) -> float:
    return math.log(max(roots_GeV/mT, 1.0 + 1e-12))

def dymax(y: float, y_max_pt: float) -> float:
    return max(0.0, min(LOG2, max(y_max_pt - y, 0.0)))

# ----- AP kernel (Torch) -----------------------------------------------------
def _Phat_core_t(z, Mperp2, l2, Lp2, alpha, Nc):
    z  = torch.clamp(z, min=Z_FLOOR)
    mask0 = (l2 <= Lp2)
    inv = 1.0/(z*z*Mperp2)
    expo = (alpha*Nc/(2.0*math.pi)) * (Li2_torch(-l2*inv) - Li2_torch(-Lp2*inv))
    expo = torch.clamp(expo, min=-700.0, max=+700.0)
    deriv= 2.0*(torch.log1p(l2*inv) - torch.log1p(Lp2*inv))/z
    val  = (alpha*torch.exp(expo)*Nc*deriv)/(2.0*math.pi)
    val  = torch.where(mask0, torch.zeros_like(val), val)
    val  = torch.where(torch.isfinite(val) & (val>0.0), val, torch.zeros_like(val))
    return val

def PhatA_t(z, mT, xA, qp: QuenchParams, pT: float | None=None):
    a   = torch.full_like(z, _alpha_mu_t(qp, mT, 0.0 if pT is None else float(pT)))
    l2  = _l2_t(qp, xA, qp.LA_fm)
    Lp2 = _Lambda_p2_t(qp, xA)
    return _Phat_core_t(z, mT*mT, l2, Lp2, a, qp.Nc)

def PhatB_t(z, mT, xB, qp: QuenchParams, pT: float | None=None):
    a   = torch.full_like(z, _alpha_mu_t(qp, mT, 0.0 if pT is None else float(pT)))
    l2  = _l2_t(qp, xB, qp.LB_fm)
    Lp2 = _Lambda_p2_t(qp, xB)
    return _Phat_core_t(z, mT*mT, l2, Lp2, a, qp.Nc)

# ----- Torch σ_pp table ------------------------------------------------------
class TorchSigmaPPTable:
    """Torch bilinear in y and linear in pT (broadcast-friendly)."""
    def __init__(self, P, roots_GeV: float, y_grid: np.ndarray, pt_grid: np.ndarray, device: Optional[str]=None):
        assert _HAS_TORCH, "Torch not available."
        self.P, self.roots = P, float(roots_GeV)
        self.device = _torch_device(device)
        self.y  = torch.tensor(np.asarray(y_grid, float),  dtype=torch.float64, device=self.device)
        self.pt = torch.tensor(np.asarray(pt_grid, float), dtype=torch.float64, device=self.device)
        Z = np.empty((self.y.numel(), self.pt.numel()), float)
        for i, yy in enumerate(self.y.cpu().numpy()):
            for j, pp in enumerate(self.pt.cpu().numpy()):
                Z[i, j] = float(P.d2sigma_pp(float(yy), float(pp), self.roots))
        self.Z = torch.tensor(Z, dtype=torch.float64, device=self.device)

    def __call__(self, y, pt):
        yv = y.to(self.device, dtype=torch.float64)
        pv = pt.to(self.device, dtype=torch.float64)
        # Clamp queries to domain (prevents artificial “same for all L” at edges)
        yv = torch.clamp(yv, min=self.y[0],  max=self.y[-1])
        pv = torch.clamp(pv, min=self.pt[0], max=self.pt[-1])

        i  = torch.searchsorted(self.y, yv) - 1
        i  = torch.clamp(i, 0, self.y.numel()-2)
        y0 = self.y[i]; y1 = self.y[i+1]
        ty = torch.where((y1>y0), (yv-y0)/(y1-y0), torch.zeros_like(yv))

        def interp_row(idx):
            row = self.Z[idx]
            j = torch.searchsorted(self.pt, pv) - 1
            j = torch.clamp(j, 0, self.pt.numel()-2)
            p0 = self.pt[j]; p1 = self.pt[j+1]
            pv_safe = torch.clamp(pv, min=p0, max=p1)
            r0 = row[j];     r1 = row[j+1]
            u  = torch.where((p1>p0), (pv_safe-p0)/(p1-p0), torch.zeros_like(pv_safe))
            return (1.0-u)*r0 + u*r1

        z0 = interp_row(i)
        z1 = interp_row(i+1)
        return (1.0 - ty.unsqueeze(-1))*z0 + ty.unsqueeze(-1)*z1

def _dsigpp_from_table(table: TorchSigmaPPTable):
    def f(y_t, pt_t):
        return table(y_t, pt_t)
    return f

# ----- φ kinematics (Torch) --------------------------------------------------
def _shift_pT_pA(pt: float, dpta, cphi, sphi):
    ptv = torch.tensor(pt, dtype=torch.float64, device=dpta.device)
    return torch.sqrt((ptv - dpta*cphi)**2 + (dpta*sphi)**2)

def _shift_pT_AB(pt: float, dptB, dptA, cA, sA, cB, sB):
    ptv = torch.tensor(pt, dtype=torch.float64, device=dptA.device)
    comp1 = ptv - dptA*cA - dptB*cB
    comp2 =        dptA*sA + dptB*sB
    return torch.sqrt(comp1*comp1 + comp2*comp2)

# ----- adaptive Ny for thin phase space -------------------------------------
def _Ny_from_dymax(dym: float) -> int:
    if dym < 0.005: return 96  
    if dym < 0.02:  return 64
    if dym < 0.05:  return 48
    if dym < 0.10:  return 40
    return 32

# ----- pA σ integral (Torch / NumPy) ----------------------------------------
def pA_cross_section(
    y: float, pT: float, mT: float, xA_scalar: float, _xB_unused: float, y_max_pt: float,
    dsig_pp: Callable, qp: QuenchParams, Ny: Optional[int] = None, Nphi: int = 16, use_torch: bool = True
) -> float:
    if use_torch and _HAS_TORCH and hasattr(dsig_pp, "__call__") and not isinstance(dsig_pp, np.ndarray):
        device = _torch_device(qp.device)
        with torch.no_grad():
            dym = dymax(+y, y_max_pt)
            # Cap δy by σ_pp table y-range to avoid clamped integrand
            if isinstance(dsig_pp, TorchSigmaPPTable):
                y_hi = float(dsig_pp.y[-1].item())
                dym_tbl = max(0.0, y_hi - y - 1e-8)
                dym = min(dym, dym_tbl)
            if Ny is None: Ny = _Ny_from_dymax(dym)

            xA  = torch.tensor([xA_scalar], dtype=torch.float64, device=device)
            l2A = _l2_t(qp, xA, qp.LA_fm)
            Lp2 = _Lambda_p2_t(qp, xA)
            dspp = _dsigpp_from_table(dsig_pp) if isinstance(dsig_pp, TorchSigmaPPTable) else dsig_pp
            table_obj = dsig_pp if isinstance(dsig_pp, TorchSigmaPPTable) else None

            # No medium → σ_pp
            if torch.all(l2A <= Lp2):
                return float(dspp(torch.tensor([y], dtype=torch.float64, device=device),
                                  torch.tensor([pT], dtype=torch.float64, device=device))[0,0].item())

            # φ nodes and Cronin shift
            _, wphi, cphi, sphi = _phi_nodes_gl_torch(Nphi, device)
            dpta = _dpt_from_xL_t(qp, xA, qp.LA_fm, hard=qp.use_hard_cronin)[0]
            pphi = _shift_pT_pA(pT, dpta, cphi, sphi)
            if table_obj is not None:
                pphi = torch.clamp(pphi, min=table_obj.pt[0], max=table_obj.pt[-1])

            # Thin phase space → δ-peak
            if dym <= DY_EPS:
                y0   = torch.tensor([y], dtype=torch.float64, device=device).unsqueeze(-1)
                sig0 = dspp(y0, pphi.unsqueeze(0))
                avg0 = torch.sum(sig0*wphi.unsqueeze(0), dim=-1)[0]
                return float(avg0.item())

            # Full δy integral (exp mapping)
            if qp.mapping == "exp":
                umin = -30.0
                umax = math.log(max(dym, 1e-300))

                def _integrand_u(u):
                    u  = u.squeeze(-1)
                    dy = torch.exp(u)
                    z  = torch.expm1(dy).clamp_min(Z_FLOOR)
                    ph = PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT)
                    if (ph <= 0).all():
                        return torch.zeros_like(u)
                    yshift = torch.tensor(y, dtype=torch.float64, device=device) + dy
                    if table_obj is not None:
                        yshift = torch.clamp(yshift, min=table_obj.y[0], max=table_obj.y[-1])
                    sig    = dspp(yshift.unsqueeze(-1), pphi.unsqueeze(0))
                    avg    = torch.sum(sig*wphi.unsqueeze(0), dim=-1)
                    return torch.exp(u)*ph*avg  # Jacobian

                if _HAS_TQ:
                    integ = TQ_GaussLegendre()
                    val = integ.integrate(f=_integrand_u, dim=1, N=Ny,
                                          integration_domain=[[umin, umax]],
                                          device=device, dtype=torch.float64)

                    def _Zc_u(u):
                        u  = u.squeeze(-1)
                        dy = torch.exp(u)
                        z  = torch.expm1(dy).clamp_min(Z_FLOOR)
                        ph = PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT)
                        return torch.exp(u)*ph
                    Zc = integ.integrate(f=_Zc_u, dim=1, N=Ny,
                                         integration_domain=[[umin, umax]],
                                         device=device, dtype=torch.float64)
                    p0 = torch.clamp(1.0 - Zc, 0.0, 1.0)

                    y0   = torch.tensor([y], dtype=torch.float64, device=device).unsqueeze(-1)
                    sig0 = dspp(y0, pphi.unsqueeze(0))
                    avg0 = torch.sum(sig0*wphi.unsqueeze(0), dim=-1)[0]

                    return float((p0*avg0 + val).item())
                else:
                    u, wu = _gl_nodes_torch(umin, umax, Ny, device)
                    dy = torch.exp(u); z = torch.expm1(dy).clamp_min(Z_FLOOR)
                    ph = PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT)
                    yshift = torch.tensor(y, dtype=torch.float64, device=device) + dy
                    if table_obj is not None:
                        yshift = torch.clamp(yshift, min=table_obj.y[0], max=table_obj.y[-1])
                    sig    = dspp(yshift.unsqueeze(-1), pphi.unsqueeze(0))
                    avg    = torch.sum(sig*wphi.unsqueeze(0), dim=-1)

                    val = torch.sum(wu*torch.exp(u)*ph*avg)
                    Zc  = torch.sum(wu*torch.exp(u)*ph)
                    p0  = torch.clamp(1.0 - Zc, 0.0, 1.0)

                    y0   = torch.tensor([y], dtype=torch.float64, device=device).unsqueeze(-1)
                    sig0 = dspp(y0, pphi.unsqueeze(0))
                    avg0 = torch.sum(sig0*wphi.unsqueeze(0), dim=-1)[0]

                    return float((p0*avg0 + val).item())
            else:
                # Linear mapping on dy (rare; kept for completeness)
                dy_nodes, dy_w = _gl_nodes_torch(0.0, dym, Ny, device)
                z  = torch.expm1(dy_nodes).clamp_min(Z_FLOOR)
                ph = PhatA_t(z, mT, xA.expand_as(z), qp, pT=pT)
                yshift = torch.tensor(y, dtype=torch.float64, device=device) + dy_nodes
                if table_obj is not None:
                    yshift = torch.clamp(yshift, min=table_obj.y[0], max=table_obj.y[-1])
                sig    = dspp(yshift.unsqueeze(-1), pphi.unsqueeze(0))
                avg    = torch.sum(sig*wphi.unsqueeze(0), dim=-1)

                val = torch.sum(dy_w*ph*avg)
                Zc  = torch.sum(dy_w*ph)
                p0  = torch.clamp(1.0 - Zc, 0.0, 1.0)

                y0   = torch.tensor([y], dtype=torch.float64, device=device).unsqueeze(-1)
                sig0 = dspp(y0, pphi.unsqueeze(0))
                avg0 = torch.sum(sig0*wphi.unsqueeze(0), dim=-1)[0]

                return float((p0*avg0 + val).item())

    # ---- NumPy fallback ----
    dym = dymax(+y, y_max_pt)
    if Ny is None:
        Ny = _Ny_from_dymax(dym)

    xA  = xA_scalar
    l2A = _l2(qp, xA, qp.LA_fm)
    lp2 = _l2(qp, xA0_from_L(qp.lp_fm), qp.lp_fm)
    Lp2 = max(lp2, qp.lambdaQCD**2)

    if l2A <= Lp2:
        return float(dsig_pp(y, np.array([pT]))[0])

    phi, wphi = _gl_nodes_np(0.0, 2.0 * math.pi, Nphi)
    dpta = math.sqrt(max(l2A - Lp2, 0.0)) if qp.use_hard_cronin else math.sqrt(max(l2A - Lp2, 0.0) + 1e-4)
    pphi = np.sqrt((pT - dpta*np.cos(phi))**2 + (dpta*np.sin(phi))**2)

    if dym <= DY_EPS:
        sig0 = dsig_pp(y, pphi)
        return float(np.sum(sig0*wphi) / (2.0*math.pi))

    dy, wy = _gl_nodes_np(0.0, dym, Ny)
    z = np.expm1(dy).clip(min=Z_FLOOR)
    inv = 1.0 / (z*z*mT*mT)

    a    = _alpha_mu_t(qp, mT, pT)
    Li   = (Li2_np(-l2A*inv) - Li2_np(-Lp2*inv))
    expo = np.clip((a * qp.Nc / (2.0 * math.pi)) * Li, -700.0, 700.0)
    deriv= 2.0 * (np.log1p(l2A*inv) - np.log1p(Lp2*inv)) / z
    Ph   = (a * np.exp(expo) * qp.Nc * deriv) / (2.0 * math.pi)
    Ph   = np.where(np.isfinite(Ph) & (Ph > 0.0), Ph, 0.0)

    val_cont = 0.0
    for yi, wi, Phi in zip(y + dy, wy, Ph):
        sig = dsig_pp(yi, pphi)
        val_cont += wi * Phi * np.sum(sig * wphi) / (2.0*math.pi)

    Zc   = float(np.sum(wy * Ph))
    Zc   = min(max(Zc, 0.0), 1.0)
    p0   = 1.0 - Zc
    sig0 = dsig_pp(y, pphi)
    avg0 = float(np.sum(sig0 * wphi) / (2.0 * math.pi))

    return float(p0 * avg0 + val_cont)

# ----- AB σ integral (Torch / NumPy) ----------------------------------------
def AB_cross_section(
    y: float, pT: float, mT: float, xA_scalar: float, xB_scalar: float, y_max_pt: float,
    dsig_pp: Callable, qp: QuenchParams, Ny: Optional[int] = None, Nphi: int = 12, use_torch: bool = True
) -> float:
    """
    Full AB integral. If only one side (A or B) is active (ℓ^2<=Λ_p^2 or δy_max≈0 on the
    other side), fall back to a one-sided pA-like integral (still including Cronin from both).
    """
    if use_torch and _HAS_TORCH and hasattr(dsig_pp, "__call__"):
        device = _torch_device(qp.device)
        with torch.no_grad():
            dymA = dymax(-y, y_max_pt)
            dymB = dymax(+y, y_max_pt)

            # Cap both δy by σ_pp table y-range
            if isinstance(dsig_pp, TorchSigmaPPTable):
                y_lo = float(dsig_pp.y[0].item())
                y_hi = float(dsig_pp.y[-1].item())
                dymA = min(dymA, max(0.0, y - y_lo - 1e-8))
                dymB = min(dymB, max(0.0, y_hi - y - 1e-8))

            if Ny is None: Ny = _Ny_from_dymax(min(dymA, dymB))
            xA = torch.tensor([xA_scalar], dtype=torch.float64, device=device)
            xB = torch.tensor([xB_scalar], dtype=torch.float64, device=device)
            l2A, l2B = _l2_t(qp, xA, qp.LA_fm), _l2_t(qp, xB, qp.LB_fm)
            Lp2A, Lp2B = _Lambda_p2_t(qp, xA), _Lambda_p2_t(qp, xB)
            dspp = _dsigpp_from_table(dsig_pp) if isinstance(dsig_pp, TorchSigmaPPTable) else dsig_pp
            table_obj = dsig_pp if isinstance(dsig_pp, TorchSigmaPPTable) else None

            _, wphi, cphi, sphi = _phi_nodes_gl_torch(Nphi, device)
            dpta = _dpt_from_xL_t(qp, xA, qp.LA_fm, hard=qp.use_hard_cronin)[0]
            dptb = _dpt_from_xL_t(qp, xB, qp.LB_fm, hard=qp.use_hard_cronin)[0]
            # two-angle average for Cronin (both sides)
            cA, sA = cphi[:,None], sphi[:,None]
            cB, sB = cphi[None,:], sphi[None,:]
            w2 = (wphi[:,None]*wphi[None,:]).reshape(-1)
            pshift = _shift_pT_AB(pT, dptb, dpta, cA, sA, cB, sB).reshape(-1)
            if table_obj is not None:
                pshift = torch.clamp(pshift, min=table_obj.pt[0], max=table_obj.pt[-1])

            # If both sides inactive or both δy almost zero → σ_pp (with Cronin folded into pshift=0)
            both_inactive = (torch.all(l2A <= Lp2A) and torch.all(l2B <= Lp2B))
            both_tiny     = (dymA <= DY_EPS and dymB <= DY_EPS)
            if both_inactive or both_tiny:
                return float(dspp(torch.tensor([y],dtype=torch.float64,device=device),
                                  torch.tensor([pT],dtype=torch.float64,device=device))[0,0].item())

            # ---- One-sided fallbacks (NEW) ----
            only_A_active = (dymA > DY_EPS and torch.any(l2A > Lp2A)) and (dymB <= DY_EPS or torch.all(l2B <= Lp2B))
            only_B_active = (dymB > DY_EPS and torch.any(l2B > Lp2B)) and (dymA <= DY_EPS or torch.all(l2A <= Lp2A))

            if only_A_active:
                # yshift = y - dyA ; kernel = PhatA
                uA, wA = _gl_nodes_torch(-30.0, math.log(max(dymA, 1e-300)), Ny, device)
                dyA = torch.exp(uA); zA = torch.expm1(dyA).clamp_min(Z_FLOOR)
                phA = PhatA_t(zA, mT, xA.expand_as(zA), qp, pT=pT)
                acc = torch.tensor(0.0, dtype=torch.float64, device=device)
                for i in range(Ny):
                    if phA[i] <= 0.0: continue
                    yshift = torch.tensor(y, dtype=torch.float64, device=device) - dyA[i]
                    if table_obj is not None:
                        yshift = torch.clamp(yshift, min=table_obj.y[0], max=table_obj.y[-1])
                    sig    = dspp(yshift.unsqueeze(-1), pshift.unsqueeze(0))
                    avg    = torch.sum(sig*w2.unsqueeze(0), dim=-1)[0]
                    acc   += wA[i] * torch.exp(uA[i]) * phA[i] * avg
                return float(acc.item())

            if only_B_active:
                # yshift = y + dyB ; kernel = PhatB
                uB, wB = _gl_nodes_torch(-30.0, math.log(max(dymB, 1e-300)), Ny, device)
                dyB = torch.exp(uB); zB = torch.expm1(dyB).clamp_min(Z_FLOOR)
                phB = PhatB_t(zB, mT, xB.expand_as(zB), qp, pT=pT)
                acc = torch.tensor(0.0, dtype=torch.float64, device=device)
                for j in range(Ny):
                    if phB[j] <= 0.0: continue
                    yshift = torch.tensor(y, dtype=torch.float64, device=device) + dyB[j]
                    if table_obj is not None:
                        yshift = torch.clamp(yshift, min=table_obj.y[0], max=table_obj.y[-1])
                    sig    = dspp(yshift.unsqueeze(-1), pshift.unsqueeze(0))
                    avg    = torch.sum(sig*w2.unsqueeze(0), dim=-1)[0]
                    acc   += wB[j] * torch.exp(uB[j]) * phB[j] * avg
                return float(acc.item())

            # ---- Two-sided (default) ----
            if qp.mapping == "exp" and _HAS_TQ:
                uminA = -30.0; umaxA = math.log(max(dymA, 1e-300))
                uminB = -30.0; umaxB = math.log(max(dymB, 1e-300))
                integ = TQ_GaussLegendre()
                def f_u(U):
                    uA = U[...,0]; uB = U[...,1]
                    dyA = torch.exp(uA); dyB = torch.exp(uB)
                    zA  = torch.expm1(dyA).clamp_min(Z_FLOOR)
                    zB  = torch.expm1(dyB).clamp_min(Z_FLOOR)
                    phA = PhatA_t(zA, mT, xA.expand_as(zA), qp, pT=pT)
                    phB = PhatB_t(zB, mT, xB.expand_as(zB), qp, pT=pT)
                    if (phA<=0).all() or (phB<=0).all():
                        return torch.zeros_like(uA)
                    yshift = torch.tensor(y, dtype=torch.float64, device=device) + dyB - dyA
                    if table_obj is not None:
                        yshift = torch.clamp(yshift, min=table_obj.y[0], max=table_obj.y[-1])
                    sig    = dspp(yshift.unsqueeze(-1), pshift.unsqueeze(0))
                    avg    = torch.sum(sig*w2.unsqueeze(0), dim=-1)
                    return torch.exp(uA+uB)*phA*phB*avg
                val = integ.integrate(f=f_u, dim=2, N=Ny,
                                      integration_domain=[[uminA, umaxA],[uminB, umaxB]],
                                      device=device, dtype=torch.float64)
                return float(val.item())

            # GL on u (torch)
            uA, wA = _gl_nodes_torch(-30.0, math.log(max(dymA, 1e-300)), Ny, device)
            uB, wB = _gl_nodes_torch(-30.0, math.log(max(dymB, 1e-300)), Ny, device)
            dyA = torch.exp(uA); zA = torch.expm1(dyA).clamp_min(Z_FLOOR)
            dyB = torch.exp(uB); zB = torch.expm1(dyB).clamp_min(Z_FLOOR)
            phA = PhatA_t(zA, mT, xA.expand_as(zA), qp, pT=pT)
            phB = PhatB_t(zB, mT, xB.expand_as(zB), qp, pT=pT)
            acc = torch.tensor(0.0, dtype=torch.float64, device=device)
            for j in range(Ny):
                if phB[j] <= 0.0:
                    continue
                row = torch.tensor(0.0, dtype=torch.float64, device=device)
                for i in range(Ny):
                    if phA[i] <= 0.0: continue
                    yshift = torch.tensor(y, dtype=torch.float64, device=device) + dyB[j] - dyA[i]
                    if table_obj is not None:
                        yshift = torch.clamp(yshift, min=table_obj.y[0], max=table_obj.y[-1])
                    sig    = dspp(yshift.unsqueeze(-1), pshift.unsqueeze(0))
                    avg    = torch.sum(sig*w2.unsqueeze(0), dim=-1)[0]
                    row   += wA[i] * torch.exp(uA[i]) * phA[i] * avg
                acc += wB[j] * torch.exp(uB[j]) * phB[j] * row
            return float(acc.item())

    # ---- NumPy fallback ----
    dymA = dymax(-y, y_max_pt); dymB = dymax(+y, y_max_pt)
    if Ny is None: Ny = _Ny_from_dymax(min(dymA, dymB))
    xA, xB = xA_scalar, xB_scalar
    l2A, l2B = _l2(qp, xA, qp.LA_fm), _l2(qp, xB, qp.LB_fm)
    lp2 = _l2(qp, xA0_from_L(qp.lp_fm), qp.lp_fm)
    Lp2 = max(lp2, qp.lambdaQCD**2)

    phi, wphi = _gl_nodes_np(0.0, 2.0*math.pi, Nphi)
    w2 = (wphi[:,None]*wphi[None,:]).ravel()/(2.0*math.pi)
    dpta = math.sqrt(max(l2A - Lp2, 0.0))
    dptb = math.sqrt(max(l2B - Lp2, 0.0))
    cA, sA = np.cos(phi)[:,None], np.sin(phi)[:,None]
    cB, sB = np.cos(phi)[None,:], np.sin(phi)[None,:]
    pshift = np.sqrt((pT - dpta*cA - dptb*cB)**2 + (dpta*sA + dptb*sB)**2).ravel()

    both_inactive = (l2A <= Lp2 and l2B <= Lp2)
    both_tiny = (dymA <= DY_EPS and dymB <= DY_EPS)
    if both_inactive or both_tiny:
        return float(dsig_pp(y, np.array([pT]))[0])

    only_A_active = (dymA > DY_EPS and l2A > Lp2) and (dymB <= DY_EPS or l2B <= Lp2)
    only_B_active = (dymB > DY_EPS and l2B > Lp2) and (dymA <= DY_EPS or l2A <= Lp2)

    a = _alpha_mu_t(qp, mT, pT)

    if only_A_active:
        uA, wA = _gl_nodes_np(-30.0, math.log(max(dymA, 1e-300)), Ny)
        dyA = np.exp(uA); zA = np.expm1(dyA).clip(min=Z_FLOOR)
        invA = 1.0/(zA*zA*mT*mT)
        expoA = np.clip((a*qp.Nc/(2.0*math.pi))*(Li2_np(-l2A*invA)-Li2_np(-Lp2*invA)), -700.0, 700.0)
        derivA= 2.0*(np.log1p(l2A*invA) - np.log1p(Lp2*invA))/zA
        PhA = (a*np.exp(expoA)*qp.Nc*derivA)/(2.0*math.pi)
        PhA = np.where(np.isfinite(PhA) & (PhA>0.0), PhA, 0.0)
        acc = 0.0
        for i in range(Ny):
            if PhA[i] <= 0.0: continue
            sig = dsig_pp(y - dyA[i], pshift)
            acc += wA[i] * np.exp(uA[i]) * PhA[i] * np.sum(sig * w2)
        return float(acc)

    if only_B_active:
        uB, wB = _gl_nodes_np(-30.0, math.log(max(dymB, 1e-300)), Ny)
        dyB = np.exp(uB); zB = np.expm1(dyB).clip(min=Z_FLOOR)
        invB = 1.0/(zB*zB*mT*mT)
        expoB = np.clip((a*qp.Nc/(2.0*math.pi))*(Li2_np(-l2B*invB)-Li2_np(-Lp2*invB)), -700.0, 700.0)
        derivB= 2.0*(np.log1p(l2B*invB) - np.log1p(Lp2*invB))/zB
        PhB = (a*np.exp(expoB)*qp.Nc*derivB)/(2.0*math.pi)
        PhB = np.where(np.isfinite(PhB) & (PhB>0.0), PhB, 0.0)
        acc = 0.0
        for j in range(Ny):
            if PhB[j] <= 0.0: continue
            sig = dsig_pp(y + dyB[j], pshift)
            acc += wB[j] * np.exp(uB[j]) * PhB[j] * np.sum(sig * w2)
        return float(acc)

    # two-sided (NumPy)
    uA, wA = _gl_nodes_np(-30.0, math.log(max(dymA, 1e-300)), Ny)
    uB, wB = _gl_nodes_np(-30.0, math.log(max(dymB, 1e-300)), Ny)
    dyA = np.exp(uA); zA = np.expm1(dyA).clip(min=Z_FLOOR)
    dyB = np.exp(uB); zB = np.expm1(dyB).clip(min=Z_FLOOR)
    invA = 1.0/(zA*zA*mT*mT); invB = 1.0/(zB*zB*mT*mT)
    expoA = np.clip((a*qp.Nc/(2.0*math.pi))*(Li2_np(-l2A*invA)-Li2_np(-Lp2*invA)), -700.0, 700.0)
    expoB = np.clip((a*qp.Nc/(2.0*math.pi))*(Li2_np(-l2B*invB)-Li2_np(-Lp2*invB)), -700.0, 700.0)
    derivA= 2.0*(np.log1p(l2A*invA) - np.log1p(Lp2*invA))/zA
    derivB= 2.0*(np.log1p(l2B*invB) - np.log1p(Lp2*invB))/zB
    PhA = (a*np.exp(expoA)*qp.Nc*derivA)/(2.0*math.pi)
    PhB = (a*np.exp(expoB)*qp.Nc*derivB)/(2.0*math.pi)
    PhA = np.where(np.isfinite(PhA) & (PhA>0.0), PhA, 0.0)
    PhB = np.where(np.isfinite(PhB) & (PhB>0.0), PhB, 0.0)
    acc = 0.0
    for j in range(Ny):
        if PhB[j] <= 0.0: continue
        row = 0.0
        for i in range(Ny):
            if PhA[i] <= 0.0: continue
            sig = dsig_pp(y + dyB[j] - dyA[i], pshift)
            row += wA[i] * np.exp(uA[i]) * PhA[i] * np.sum(sig * w2)
        acc += wB[j] * np.exp(uB[j]) * PhB[j] * row
    return float(acc)

# ----- Convenience wrappers ---------------------------------------------------
def sigma_pA(P, roots_GeV: float, qp: QuenchParams, y: float, pT: float,
             table_or_callable, xA_scalar: float, y_max_pt: float,
             Ny: Optional[int] = None, Nphi: int = 16, use_torch: bool = True) -> float:
    if _HAS_TORCH and isinstance(table_or_callable, TorchSigmaPPTable):
        mT = float(P.mT(pT))
        return pA_cross_section(y, pT, mT, xA_scalar, 0.0, y_max_pt, table_or_callable, qp, Ny=Ny, Nphi=Nphi, use_torch=use_torch)
    else:
        def dspp_np(y_scalar, pt_arr):
            return np.asarray([P.d2sigma_pp(y_scalar, float(pt), roots_GeV) for pt in pt_arr], float)
        mT = float(P.mT(pT))
        return pA_cross_section(y, pT, mT, xA_scalar, 0.0, y_max_pt, dspp_np, qp, Ny=Ny, Nphi=Nphi, use_torch=False)

def sigma_AB(P, roots_GeV: float, qp: QuenchParams, y: float, pT: float,
             table_or_callable, xA_scalar: float, xB_scalar: float, y_max_pt: float,
             Ny: Optional[int] = None, Nphi: int = 12, use_torch: bool = True) -> float:
    if _HAS_TORCH and isinstance(table_or_callable, TorchSigmaPPTable):
        mT = float(P.mT(pT))
        return AB_cross_section(y, pT, mT, xA_scalar, xB_scalar, y_max_pt, table_or_callable, qp, Ny=Ny, Nphi=Nphi, use_torch=use_torch)
    else:
        def dspp_np(y_scalar, pt_arr):
            return np.asarray([P.d2sigma_pp(y_scalar, float(pt), roots_GeV) for pt in pt_arr], float)
        mT = float(P.mT(pT))
        return AB_cross_section(y, pT, mT, xA_scalar, xB_scalar, y_max_pt, dspp_np, qp, Ny=Ny, Nphi=Nphi, use_torch=False)

# ----- Nuclear modification (RAW) -------------------------------------------
def nuclear_modification(
    P, roots_GeV: float, qp: QuenchParams, y: float, pT: float,
    table_or_callable, kind: Literal["pA","AA"]="pA", use_torch: bool = True
) -> float:
    mT = float(P.mT(pT))
    yM = y_max(roots_GeV, mT)
    if (_HAS_TORCH and isinstance(table_or_callable, TorchSigmaPPTable)):
        sig_pp = float(table_or_callable(torch.tensor([y], dtype=torch.float64, device=table_or_callable.device),
                                         torch.tensor([pT],dtype=torch.float64, device=table_or_callable.device))[0,0].item())
    else:
        sig_pp = float(P.d2sigma_pp(y, pT, roots_GeV))
    if kind == "pA":
        xA = min(xA0_from_L(qp.LA_fm), (mT/roots_GeV)*math.exp(-y))
        sig = sigma_pA(P, roots_GeV, qp, y, pT, table_or_callable, xA, yM, use_torch=use_torch)
    else:
        xA = min(xA0_from_L(qp.LA_fm), (mT/roots_GeV)*math.exp(-y))
        xB = min(xA0_from_L(qp.LB_fm), (mT/roots_GeV)*math.exp(+y))
        sig = sigma_AB(P, roots_GeV, qp, y, pT, table_or_callable, xA, xB, yM, use_torch=use_torch)
    return sig/sig_pp

# ----- Convenience RAW helpers ----------------------------------------------
def rpa_raw_vs_L(P, roots, qp_base, table, L_list, y: float, pT: float, kind="pA", use_torch: bool=True):
    vals = []
    for L in L_list:
        qpL = replace(qp_base, LA_fm=float(L))
        vals.append(nuclear_modification(P, roots, qpL, y, pT, table, kind=kind, use_torch=use_torch))
    return np.array(vals)

def rpa_raw_vs_y(P, roots, qp, table, y_grid, pT_fix, kind="pA", use_torch: bool=True):
    return np.array([nuclear_modification(P, roots, qp, float(y), float(pT_fix), table, kind=kind, use_torch=use_torch) for y in y_grid])

def rpa_raw_vs_pT(P, roots, qp, table, p_grid, y_fix, kind="pA", use_torch: bool=True):
    return np.array([nuclear_modification(P, roots, qp, float(y_fix), float(p), table, kind=kind, use_torch=use_torch) for p in p_grid])

# ----- BINNING (y,pT) -------------------------------------------------------
def nuclear_modification_binned(
    P, roots_GeV: float, qp: QuenchParams,
    y_range: Tuple[float, float], pt_range: Tuple[float, float],
    table: TorchSigmaPPTable | Callable,
    kind: Literal["pA", "AA"] = "pA",
    Ny_bin: int = 24, Npt_bin: int = 48,
    weight_kind: Literal["auto", "flat", "pp", "pA", "AA"] = "auto",
    # NEW: pass "local" to use local y; None (default) means y=0
    weight_ref_y: Union[float, Literal["local"], None] = None,
    Nphi_weight: int = 12,
    use_torch: bool = True
) -> float:
    yl, yr = y_range; pl, pr = pt_range
    y_nodes, y_w = _gl_nodes_np(yl, yr, Ny_bin)
    p_nodes, p_w = _gl_nodes_np(pl, pr, Npt_bin)
    acc_num, acc_den = 0.0, 0.0

    if weight_kind == "auto":
        wkind = "pA" if kind == "pA" else ("AA" if kind == "AA" else "pp")
    else:
        wkind = weight_kind

    # Interpret weight_ref_y:
    #   None (default) → y=0 weighting (per your default)
    #   "local"        → use local y inside the bin loops
    if isinstance(weight_ref_y, str) and weight_ref_y.lower() == "local":
        y_fixed_for_w = None  # local y
    elif weight_ref_y is None:
        y_fixed_for_w = 0.0   # default y=0
    else:
        y_fixed_for_w = float(weight_ref_y)

    pre_W = None
    # Precompute p-weights at a fixed y if requested
    if (y_fixed_for_w is not None) and (wkind in {"pp", "pA", "AA"}):
        y_wref = float(y_fixed_for_w)

        if wkind == "pp":
            if _HAS_TORCH and isinstance(table, TorchSigmaPPTable):
                y_t = torch.tensor([y_wref], dtype=torch.float64, device=table.device)
                p_t = torch.tensor(np.asarray(p_nodes, float), dtype=torch.float64, device=table.device)
                with torch.no_grad():
                    w_sig = table(y_t, p_t)[0].detach().cpu().numpy()
            else:
                w_sig = np.array([float(P.d2sigma_pp(y_wref, float(pj), roots_GeV)) for pj in p_nodes], float)
            pre_W = w_sig * np.maximum(p_nodes, 1e-8)

        elif wkind == "pA":
            W = np.empty_like(p_nodes, dtype=float)
            for j, pj in enumerate(p_nodes):
                mT = float(P.mT(pj)); yM = y_max(roots_GeV, mT)
                xA = min(xA0_from_L(qp.LA_fm), (mT/roots_GeV) * math.exp(-y_wref))
                W[j] = sigma_pA(P, roots_GeV, qp, y_wref, float(pj), table, xA, yM,
                                Nphi=Nphi_weight, use_torch=use_torch)
            pre_W = W * np.maximum(p_nodes, 1e-8)

        else:  # "AA"
            W = np.empty_like(p_nodes, dtype=float)
            for j, pj in enumerate(p_nodes):
                mT = float(P.mT(pj)); yM = y_max(roots_GeV, mT)
                xA = min(xA0_from_L(qp.LA_fm), (mT/roots_GeV) * math.exp(-y_wref))
                xB = min(xA0_from_L(qp.LB_fm), (mT/roots_GeV) * math.exp(+y_wref))
                W[j] = sigma_AB(P, roots_GeV, qp, y_wref, float(pj), table, xA, xB, yM,
                                Nphi=Nphi_weight, use_torch=use_torch)
            pre_W = W * np.maximum(p_nodes, 1e-8)

    for yi, wy in zip(y_nodes, y_w):
        if pre_W is not None:
            # fixed-y weighting (default y=0)
            for pj, wp, wj in zip(p_nodes, p_w, pre_W):
                R = nuclear_modification(P, roots_GeV, qp, float(yi), float(pj), table,
                                         kind=kind, use_torch=use_torch)
                acc_num += wy * wp * R * wj
                acc_den += wy * wp * wj
        else:
            # local-y weighting
            y_for_w = float(yi)
            for pj, wp in zip(p_nodes, p_w):
                R = nuclear_modification(P, roots_GeV, qp, float(yi), float(pj), table,
                                         kind=kind, use_torch=use_torch)
                if wkind == "flat":
                    wgt = 1.0
                elif wkind == "pp":
                    if _HAS_TORCH and isinstance(table, TorchSigmaPPTable):
                        wgt = float(table(torch.tensor([y_for_w], dtype=torch.float64, device=table.device),
                                          torch.tensor([pj],      dtype=torch.float64, device=table.device))[0, 0].item())
                    else:
                        wgt = float(P.d2sigma_pp(y_for_w, float(pj), roots_GeV))
                    wgt *= float(max(pj, 1e-8))
                elif wkind == "pA":
                    mT = float(P.mT(pj)); yM = y_max(roots_GeV, mT)
                    xA = min(xA0_from_L(qp.LA_fm), (mT/roots_GeV) * math.exp(-y_for_w))
                    wgt = sigma_pA(P, roots_GeV, qp, y_for_w, float(pj), table, xA, yM,
                                   Nphi=Nphi_weight, use_torch=use_torch) * float(max(pj, 1e-8))
                else:  # "AA"
                    mT = float(P.mT(pj)); yM = y_max(roots_GeV, mT)
                    xA = min(xA0_from_L(qp.LA_fm), (mT/roots_GeV) * math.exp(-y_for_w))
                    xB = min(xA0_from_L(qp.LB_fm), (mT/roots_GeV) * math.exp(+y_for_w))
                    wgt = sigma_AB(P, roots_GeV, qp, y_for_w, float(pj), table, xA, xB, yM,
                                   Nphi=Nphi_weight, use_torch=use_torch) * float(max(pj, 1e-8))
                acc_num += wy * wp * R * wgt
                acc_den += wy * wp * wgt

    return float(acc_num / acc_den if acc_den > 0 else acc_num)

# ----- Centrality (Glauber-driven) ------------------------------------------
_MB_TO_FM2 = 0.1  # 1 mb = 0.1 fm^2

def _normalize_cent_bins(cent_edges_or_bins):
    arr = np.asarray(cent_edges_or_bins, dtype=float)
    if arr.ndim == 1:
        edges = np.unique(np.sort(arr))
        if edges.size < 2:
            raise ValueError("cent_edges must have ≥2 values, e.g. [0,20,40,60,80,100].")
        bins_list = [(float(edges[i]), float(edges[i+1])) for i in range(edges.size-1)]
    else:
        bins_list = [(float(a), float(b)) for (a, b) in cent_edges_or_bins]
        edges = np.unique(np.sort(np.array([v for ab in bins_list for v in ab], float)))
    labels  = [f"{int(a)}-{int(b)}%" for (a,b) in bins_list]
    centers = 0.5 * np.array([a + b for (a,b) in bins_list], float)
    return edges, bins_list, labels, centers

def _optical_bin_weight_pA(glauber, c0_percent: float, c1_percent: float, n_sub: int = 1200) -> float:
    bmin = float(glauber.b_from_percentile(c0_percent/100.0, kind="pA"))
    bmax = float(glauber.b_from_percentile(c1_percent/100.0, kind="pA"))
    if bmax <= bmin:
        return 0.0
    b_sub = np.linspace(bmin, bmax, n_sub)
    TpA_sub = np.interp(b_sub, np.asarray(glauber.b_grid, float), np.asarray(glauber.TpA_b, float))
    sigma_fm2 = float(glauber.spec.sigma_nn_mb) * _MB_TO_FM2
    pinel = 1.0 - np.exp(-sigma_fm2 * np.maximum(TpA_sub, 0.0))
    integrand = 2.0 * math.pi * b_sub * pinel
    numer_fm2 = float(np.trapezoid(integrand, b_sub))
    sigma_tot_fm2 = float(glauber.sigma_pA_tot_mb) * _MB_TO_FM2
    return numer_fm2 / max(sigma_tot_fm2, 1e-30)

def nuclear_modification_centrality_binned(
    P, roots_GeV: float, qp_base: QuenchParams,
    table: TorchSigmaPPTable | callable,
    glauber, cent_edges_or_bins,
    y_range: tuple[float, float], pt_range: tuple[float, float],
    kind: Literal["pA", "AA"] = "pA",
    Ny_bin: int = 18, Npt_bin: int = 36,
    weight_kind: Literal["auto", "flat", "pp", "pA", "AA"] = "auto",
    weight_ref_y: Union[float, Literal["local"], None] = None,
    use_torch: bool = True
) -> dict[str, np.ndarray]:
    _, bins_list, labels, centers = _normalize_cent_bins(cent_edges_or_bins)

    # (A) Optical bin probabilities
    w = np.array([_optical_bin_weight_pA(glauber, a, b) for (a, b) in bins_list], float)
    w = w / max(w.sum(), 1e-30)

    # (B) L_eff per bin
    try:
        L_by = glauber.leff_bins_pA([(int(a), int(b)) for (a, b) in bins_list], method="optical")
    except TypeError:
        L_by = glauber.leff_bins_pA([(int(a), int(b)) for (a, b) in bins_list])
    Leff = np.array([float(L_by[tag]) for tag in labels], float)

    # (C) ⟨R⟩ in each bin with that bin’s L_eff
    Rvals = []
    for L in Leff:
        qpL = (replace(qp_base, LA_fm=float(L), LB_fm=float(L)) if kind == "AA"
               else replace(qp_base, LA_fm=float(L)))
        Rbar = nuclear_modification_binned(
            P, roots_GeV, qpL, y_range, pt_range, table,
            kind=kind, Ny_bin=Ny_bin, Npt_bin=Npt_bin,
            weight_kind=weight_kind, weight_ref_y=weight_ref_y,
            use_torch=use_torch
        )
        Rvals.append(Rbar)
    Rvals = np.asarray(Rvals, float)

    # (D) Min-bias (over the provided bins)
    R_mb = float(np.sum(w * Rvals))

    return dict(
        centers=centers.astype(float),
        R=Rvals,
        Leff=Leff,
        weights=w.astype(float),
        labels=np.array(labels),
        R_minbias=R_mb,
        glauber_method="optical",
    )
