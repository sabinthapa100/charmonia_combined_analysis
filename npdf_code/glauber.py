# glauber.py — Optical & Binomial Glauber (AA / pA) + L_eff for quenching
# (c) you — PEP8, type hints, robust numerics, fast radial TA LUT
#
# Key fixes:
#   • _gl_integrate_2d now builds full (nx,ny) grids for X,Y → no mask mismatch
#   • TA(r) LUT integrates z∈[-zmax,+zmax] correctly (no spurious ×2)
#   • pA L_eff: "binomial" (A&P Eq. B.9) and "optical" (Tp⊗TA) both available
#   • pA mean ⟨N_coll⟩ per bin (point-like path), centrality pretty-printer
#   • Safe TA(x,y): bilinear on-grid, radial LUT fallback off-grid
#
# Dependencies: numpy (matplotlib optional for plotting helpers)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Dict, List, Literal, Optional, Sequence
import math
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False


# ----------------------------- Utilities ------------------------------------

def _leggauss(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)

def _gl_integrate_1d(f: Callable[[float], float], a: float, b: float, n: int = 64) -> float:
    x, w = _leggauss(n)
    xm, xc = 0.5 * (b - a), 0.5 * (b + a)
    tot = 0.0
    for xi, wi in zip(x, w):
        tot += wi * float(f(float(xm * xi + xc)))
    return float(xm * tot)

def _gl_integrate_2d(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ax: float, bx: float, ay: float, by: float, nx: int = 64, ny: int = 64
) -> float:
    # Build full (nx,ny) grids to avoid boolean-mask shape issues downstream.
    x, wx = _leggauss(nx)
    y, wy = _leggauss(ny)
    xm, xc = 0.5 * (bx - ax), 0.5 * (bx + ax)
    ym, yc = 0.5 * (by - ay), 0.5 * (by + ay)
    X = xm * np.broadcast_to(x[:, None], (nx, ny)) + xc
    Y = ym * np.broadcast_to(y[None, :], (nx, ny)) + yc
    W = (xm * ym) * (wx[:, None] * wy[None, :])  # outer product
    F = f(X, Y)
    return float(np.sum(W * F))

def _interp1_linear(x: np.ndarray, y: np.ndarray, xq: np.ndarray | float) -> np.ndarray | float:
    return np.interp(xq, x, y, left=y[0], right=y[-1])

def _bilinear(z: np.ndarray, x: np.ndarray, y: np.ndarray,
              xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    xq = np.asarray(xq, float); yq = np.asarray(yq, float)
    xi = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    yi = np.clip(np.searchsorted(y, yq) - 1, 0, len(y) - 2)
    x0, x1 = x[xi], x[xi + 1]
    y0, y1 = y[yi], y[yi + 1]
    z00, z10 = z[xi, yi], z[xi + 1, yi]
    z01, z11 = z[xi, yi + 1], z[xi + 1, yi + 1]
    tx = (xq - x0) / np.maximum(x1 - x0, 1e-15)
    ty = (yq - y0) / np.maximum(y1 - y0, 1e-15)
    return ((1 - tx) * (1 - ty) * z00 +
            tx * (1 - ty) * z10 +
            (1 - tx) * ty * z01 +
            tx * ty * z11)


# --------------------------- Physics constants ------------------------------

MB_TO_FM2 = 0.1
FM2_TO_MB = 10.0
DEFAULT_RHO0 = 0.17  # fm^-3
HBARC = 0.1973269804 # GeV·fm

_SIGMA_NN_BY_ROOTS_MB = {200.0: 42.0, 2760.0: 62.0, 5023.0: 67.6, 8160.0: 71.0}
_DIFFUSENESS_BY_ROOTS = {200.0: 0.535, 2760.0: 0.549, 5023.0: 0.549, 8160.0: 0.549}


# --------------------------- Data classes -----------------------------------

@dataclass(frozen=True)
class WoodsSaxon:
    A: int
    n0: float = DEFAULT_RHO0
    d_fm: float = 0.549
    rmax_fm: float = 50.0
    dr_fm: float = 0.02
    zmax_fm: float = 50.0
    nz: int = 200

    def radius_rn(self) -> float:
        a13 = self.A ** (1 / 3)
        return 1.12 * a13 - 0.86 / a13

    def n_of_r(self, r: np.ndarray) -> np.ndarray:
        Rn = self.radius_rn()
        return self.n0 / (1.0 + np.exp((r - Rn) / self.d_fm))

    def tabulate_TA_of_r(self) -> Tuple[np.ndarray, np.ndarray]:
        r = np.arange(0.0, self.rmax_fm + 1e-12, self.dr_fm)
        x, w = _leggauss(self.nz)                # nodes on [-1,1]
        z = self.zmax_fm * x                     # map to [-zmax, +zmax]
        T = []
        for ri in r:
            rr = np.sqrt(ri * ri + z * z)
            # Correct mapping: ∫_{-zmax}^{+zmax} f(z) dz ≈ zmax * Σ w_i f(zmax x_i)
            T.append(self.zmax_fm * np.sum(w * self.n_of_r(rr)))
        return r, np.array(T)

@dataclass
class ProtonProfile:
    # C++-matched normalized shape:
    # Tp(x,y) = 0.400905 * exp( -1.28022 * (x^2 + y^2)^0.925 )
    def T_p(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.400905 * np.exp(-1.28022 * np.power(x * x + y * y, 0.925))

@dataclass
class SystemSpec:
    system: Literal["AA", "pA"]
    roots_GeV: float
    A: int
    sigma_nn_mb: Optional[float] = None
    diffuseness_fm: Optional[float] = None
    def resolve(self) -> Tuple[float, float]:
        sigma = self.sigma_nn_mb if self.sigma_nn_mb is not None else \
            _SIGMA_NN_BY_ROOTS_MB.get(self.roots_GeV, 67.6)
        dval = self.diffuseness_fm if self.diffuseness_fm is not None else \
            _DIFFUSENESS_BY_ROOTS.get(self.roots_GeV, 0.549)
        return float(sigma), float(dval)


# ----------------------- Optical/Binomial Glauber ---------------------------

class OpticalGlauber:
    """
    Unified AA/pA Glauber:
      • TA(r) radial LUT (r≤50 fm) + TA(x,y) bilinear (safe fallback to radial)
      • T_AA(b) and T_pA(b) with your window (x∈[b±5], y∈[-15,15])
      • centrality CDF from P_inel(b) = 1 - exp(-σ_nn T(b))
      • pA L_eff:
          - method="binomial": Arleo–Peigné Eq. (B.9) (point-like proton)
          - method="optical" : smeared proton (Tp ⊗ TA)
    """

    def __init__(self,
                 spec: SystemSpec,
                 verbose: bool = True,
                 xylim_fm: float = 20.0,
                 nx: int = 160, ny: int = 160,
                 pa_x_half_width_fm: float = 5.0,
                 pa_y_half_width_fm: float = 15.0,
                 nx_pa: int = 160, ny_pa: int = 160) -> None:

        self.spec = spec
        self.sigma_nn_mb, self.d_fm = spec.resolve()
        self.verbose = bool(verbose)

        # 1) TA(r) LUT
        self.ws = WoodsSaxon(A=spec.A, d_fm=self.d_fm)
        if self.verbose:
            print(f"[Glauber] TA(r) LUT: A={spec.A} d={self.d_fm:.3f} "
                  f"r≤{self.ws.rmax_fm:g} fm, dr={self.ws.dr_fm:g}, z≤{self.ws.zmax_fm:g} fm")
        self.r_grid, self.T_r = self.ws.tabulate_TA_of_r()

        # 2) TA(x,y) grid
        self.xylim_fm = float(xylim_fm)
        self.nx, self.ny = int(nx), int(ny)
        self.x_grid = np.linspace(-self.xylim_fm, self.xylim_fm, self.nx)
        self.y_grid = np.linspace(-self.xylim_fm, self.ylim, self.ny) if hasattr(self, 'ylim') else np.linspace(-self.xylim_fm, self.xylim_fm, self.ny)
        X, Y = np.meshgrid(self.x_grid, self.y_grid, indexing="ij")
        R = np.sqrt(X * X + Y * Y)
        self.T_xy = _interp1_linear(self.r_grid, self.T_r, R)

        if self.verbose:
            integ = np.trapezoid(np.trapezoid(self.T_xy, self.y_grid, axis=1),
                                 self.x_grid, axis=0)
            print(f"[Glauber] ∫T_A d^2x ≈ {float(integ):.3f} (target A={self.spec.A})")

        # 3) proton
        self.proton = ProtonProfile()

        # 4) pA convolution window (C++-matched)
        self.pa_x_hw = float(pa_x_half_width_fm)
        self.pa_y_hw = float(pa_y_half_width_fm)
        self.nx_pa, self.ny_pa = int(nx_pa), int(ny_pa)

        # 5) b-grid & T(b)
        self.bmax_fm = 20.0
        self.db_fm = self.bmax_fm / 200.0
        self.b_grid = np.arange(0.0, self.bmax_fm + 1e-12, self.db_fm)

        if self.verbose:
            print("[Glauber] Tabulating T_AA(b), T_pA(b)…")
        self.TAA_b = np.array([self._TAA_of_b(b) for b in self.b_grid])
        self.TpA_b = np.array([self._TpA_conv_of_b(b) for b in self.b_grid])

        # Cross sections & CDFs
        self.sigma_AA_tot_mb = self._sigma_tot_mb(kind="AA")
        self.sigma_pA_tot_mb = self._sigma_tot_mb(kind="pA")
        self.cum_AA = self._cdf(kind="AA")
        self.cum_pA = self._cdf(kind="pA")
        if self.verbose:
            print(f"[Glauber] σ_tot^AA ≈ {self.sigma_AA_tot_mb:.2f} mb, "
                  f"σ_tot^pA ≈ {self.sigma_pA_tot_mb:.2f} mb")

    # --- TA evaluators ---

    def TA_r(self, r: np.ndarray | float) -> np.ndarray | float:
        r = np.asarray(r, float)
        m = (r <= self.r_grid[-1])
        out = np.zeros_like(r)
        if np.any(m):
            out[m] = _interp1_linear(self.r_grid, self.T_r, r[m])
        return out if out.shape else float(out)

    def TA_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float); y = np.asarray(y, float)
        r = np.hypot(x, y)
        out = np.empty_like(r)
        inside = (np.abs(x) <= self.xylim_fm) & (np.abs(y) <= self.xylim_fm)
        if inside.any():
            out[inside] = _bilinear(self.T_xy, self.x_grid, self.y_grid, x[inside], y[inside])
        if (~inside).any():
            out[~inside] = _interp1_linear(self.r_grid, self.T_r, r[~inside])
        return out

    # --- thickness integrals ---

    def _TAA_of_b(self, b_fm: float) -> float:
        bx = b_fm / 2.0
        def f(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return self.TA_xy(X + bx, Y) * self.TA_xy(X - bx, Y)
        lim = 15.0
        return _gl_integrate_2d(f, -lim, lim, -lim, lim, nx=120, ny=120)

    def _TpA_conv_of_b(self, b_fm: float) -> float:
        bx = b_fm / 2.0
        xa, xb = b_fm - self.pa_x_hw, b_fm + self.pa_x_hw
        ya, yb = -self.pa_y_hw, self.pa_y_hw
        def f(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            Ta = self.TA_xy(X + bx, Y)
            Tp = self.proton.T_p(X - bx, Y)
            return Ta * Tp
        return _gl_integrate_2d(f, xa, xb, ya, yb, nx=self.nx_pa, ny=self.ny_pa)

    # --- σ_tot and CDF ---

    def _sigma_tot_mb(self, kind: Literal["AA", "pA"]) -> float:
        sigma_fm2 = self.sigma_nn_mb * MB_TO_FM2
        T = self.TAA_b if kind == "AA" else self.TpA_b
        integrand = self.b_grid * (1.0 - np.exp(-sigma_fm2 * T))
        val_fm2 = 2.0 * math.pi * np.trapezoid(integrand, self.b_grid)
        return float(val_fm2 * FM2_TO_MB)

    def _cdf(self, kind: Literal["AA", "pA"]) -> np.ndarray:
        sigma_fm2 = self.sigma_nn_mb * MB_TO_FM2
        T = self.TAA_b if kind == "AA" else self.TpA_b
        integrand = self.b_grid * (1.0 - np.exp(-sigma_fm2 * T))
        cum = 2.0 * math.pi * np.cumsum(integrand) * self.db_fm
        sig_tot = self.sigma_AA_tot_mb if kind == "AA" else self.sigma_pA_tot_mb
        return (cum * FM2_TO_MB) / max(sig_tot, 1e-30)

    def b_from_percentile(self, c: float, kind: Literal["AA","pA"]) -> float:
        c = float(np.clip(c, 0.0, 1.0))
        cum = self.cum_AA if kind == "AA" else self.cum_pA
        return float(_interp1_linear(cum, self.b_grid, c))

    # --- pA helpers: P_inel(b) for bin edges ---

    def _pinel_pointlike(self, b: float) -> float:
        sigma = self.sigma_nn_mb * MB_TO_FM2
        lam = sigma * float(self.TA_r(b))
        if lam <= 0.0: return 0.0
        return 1.0 - math.exp(max(-700.0, -lam))

    def _pinel_smeared(self, b: float) -> float:
        sigma = self.sigma_nn_mb * MB_TO_FM2
        lam = sigma * float(_interp1_linear(self.b_grid, self.TpA_b, b))
        if lam <= 0.0: return 0.0
        return 1.0 - math.exp(max(-700.0, -lam))

    def _bin_edges_pA(self, c0: float, c1: float,
                      method: Literal["binomial","optical"]) -> Tuple[float,float]:
        pinel = self._pinel_smeared if method == "optical" else self._pinel_pointlike
        bMax, Nb, db = 20.0, 600, 20.0/600.0
        cum = np.zeros(Nb + 1); total = 0.0
        for i in range(Nb + 1):
            b = i * db
            total += pinel(b) * (2.0 * math.pi * b * db)
            cum[i] = total
        tmin, tmax = c0*total, c1*total
        bmin = next((i*db for i in range(Nb+1) if cum[i] >= tmin), 0.0)
        bmax = next((i*db for i in range(Nb+1) if cum[i] >= tmax), 20.0)
        return float(bmin), float(bmax)

    # --- pA mean Ncoll in a bin (point-like) ---

    def ncoll_mean_bin_pA(self, c0: float, c1: float) -> float:
        bmin, bmax = self._bin_edges_pA(c0, c1, method="binomial")
        Nb, db = 600, (bmax - bmin)/600.0
        sigma = self.sigma_nn_mb * MB_TO_FM2
        num = den = 0.0
        for i in range(Nb):
            b = bmin + (i + 0.5) * db
            lam = sigma * float(self.TA_r(b))
            pinel = 0.0 if lam <= 0.0 else (1.0 - math.exp(max(-700.0, -lam)))
            if pinel <= 0.0: continue
            Kcond = lam / pinel
            w = 2.0 * math.pi * b * db * pinel
            num += Kcond * w; den += w
        return float(num / max(den, 1e-30))

    def ncoll_mean_bin_pA_optical(self, c0: float, c1: float) -> float:
        bmin, bmax = self._bin_edges_pA(c0, c1, method="optical")
        Nb = 600; db = (bmax - bmin) / Nb
        sigma = self.sigma_nn_mb * MB_TO_FM2
        num = den = 0.0
        for i in range(Nb):
            b = bmin + (i + 0.5) * db
            lam = sigma * float(_interp1_linear(self.b_grid, self.TpA_b, b))
            pinel = 1.0 - math.exp(max(-700.0, -lam))
            if pinel <= 0.0:  # skip empty weights
                continue
            Kcond = lam / pinel                 # E[N | inel] at fixed b
            w = 2.0 * math.pi * b * db * pinel  # inelastic weight
            num += Kcond * w; den += w
        return float(num / max(den, 1e-30))

    # --- pA L_eff: optical (smeared) ---

    def _leff_bin_pA_optical(self, cmin: float, cmax: float,
                             rho0_fm3: float, Lp_fm: float) -> float:
        bmin, bmax = self._bin_edges_pA(cmin, cmax, method="optical")
        Nb, db = 600, (bmax - bmin)/600.0
        sigma = self.sigma_nn_mb * MB_TO_FM2
        num = den = 0.0
        for i in range(Nb):
            b = bmin + (i + 0.5) * db
            lam = sigma * float(_interp1_linear(self.b_grid, self.TpA_b, b))
            w = 2.0 * math.pi * b * db  # P cancels
            num += (lam * lam) * w; den += lam * w
        if den <= 0.0: return float(Lp_fm)
        return float(Lp_fm + num / (sigma * rho0_fm3 * den))

    # --- pA L_eff: binomial (A&P Eq. B.9) ---

    def _leff_bin_pA_binomial(self, cmin: float, cmax: float,
                              rho0_fm3: float, Lp_fm: float,
                              allow_fallback_optical: bool = True) -> float:
        A = int(self.spec.A)
        sigma = self.sigma_nn_mb * MB_TO_FM2
        # Step 1: σ_N = ∫ d^2b C(A,N) p(b)^N (1-p)^{A-N}, p=σTA/A
        bMax, Nb, db = 20.0, 1000, 20.0/1000.0
        logC = np.array([math.lgamma(A + 1) - math.lgamma(N + 1) - math.lgamma(A - N + 1)
                         for N in range(A + 1)])
        sigmaN = np.zeros(A + 1)
        for i in range(Nb + 1):
            b = i * db
            w = 0.5 if (i == 0 or i == Nb) else 1.0  # trapezoid endcaps
            jac = w * (2.0 * math.pi) * b * db
            TA_b = float(self.TA_r(b))
            p = sigma * TA_b / float(A)
            if p <= 0.0: continue
            p = min(p, 1.0 - 1e-12)
            lp, lq = math.log(p), math.log(1.0 - p)
            for N in range(1, A + 1):
                wN = math.exp(logC[N] + N * lp + (A - N) * lq)
                if wN > 0.0: sigmaN[N] += jac * wN

        sig_inel = float(np.sum(sigmaN[1:]))
        if sig_inel <= 0.0:
            return self._leff_bin_pA_optical(cmin, cmax, rho0_fm3, Lp_fm) if allow_fallback_optical else float(Lp_fm)

        # Step 2: map centrality → [N_low..N_high] via tail CDF
        F = np.zeros(A + 2); tail = 0.0
        for N in range(A, 0, -1):
            tail += sigmaN[N]
            F[N] = tail / sig_inel
        F[A + 1] = 0.0
        def N_from_c(c: float) -> int:
            if c <= 0.0: return A + 1
            if c >= 1.0: return 1
            for N in range(A, 0, -1):
                if F[N] >= c and F[N + 1] < c: return N
            return 1
        N_low  = max(1, N_from_c(cmax))
        N_high = min(A, (A if cmin <= 0.0 else N_from_c(cmin) - 1))
        if N_low > N_high:
            return self._leff_bin_pA_optical(cmin, cmax, rho0_fm3, Lp_fm) if allow_fallback_optical else float(Lp_fm)

        # Step 3: Eq. (B.9)
        Num = Den = 0.0
        for N in range(N_low, N_high + 1):
            Num += N * (N - 1) * sigmaN[N]
            Den += N * sigmaN[N]
        if Den <= 0.0:
            return self._leff_bin_pA_optical(cmin, cmax, rho0_fm3, Lp_fm) if allow_fallback_optical else float(Lp_fm)
        LA = Lp_fm + Num / (sigma * rho0_fm3 * Den)
        return float(max(Lp_fm, LA))

    # --- public L_eff selector ---

    def leff_bin_pA(self, cmin: float, cmax: float,
                    rho0_fm3: float = DEFAULT_RHO0, Lp_fm: float = 1.5,
                    method: Literal["binomial","optical"] = "binomial") -> float:
        if method == "optical":
            return self._leff_bin_pA_optical(cmin, cmax, rho0_fm3, Lp_fm)
        return self._leff_bin_pA_binomial(cmin, cmax, rho0_fm3, Lp_fm, allow_fallback_optical=True)

    def leff_minbias_pA(self, rho0_fm3: float = DEFAULT_RHO0, Lp_fm: float = 1.5) -> float:
        """
        L_eff^MB = L_p + ((A-1)/(A^2 * n0)) * ∬_{[-20,20]^2} T_A(x,y)^2 dx dy
        Uses the TA(x,y) grid built with xylim_fm=20 by default.
        """
        A   = float(self.spec.A)
        n0  = float(rho0_fm3)

        # Ensure the TA(x,y) grid covers [-20,20]^2; this is true if xylim_fm>=20.
        # If you configured xylim_fm != 20, this still integrates the full square [-xylim, xylim]^2.
        TA2_xy_int = np.trapezoid(
            np.trapezoid(self.T_xy * self.T_xy, self.y_grid, axis=1),
            self.x_grid, axis=0
        )
        return float(Lp_fm + ((A - 1.0) / (A * A * n0)) * TA2_xy_int)

    # --- convenience: multiple bins ---

    def leff_bins_pA(self, bins_percent: Sequence[Tuple[float, float]],
                     rho0_fm3: float = DEFAULT_RHO0, Lp_fm: float = 1.5,
                     method: Literal["binomial","optical"] = "binomial") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for (c0, c1) in bins_percent:
            tag = f"{int(c0)}-{int(c1)}%"
            out[tag] = self.leff_bin_pA(c0/100.0, c1/100.0, rho0_fm3, Lp_fm, method=method)
        return out

    # --- debug / pretty print ---

    def print_pA_centrality_table(self, edges_percent: Sequence[float],
                                  rho0_fm3: float = DEFAULT_RHO0,
                                  Lp_fm: float = 1.5,
                                  method: Literal["binomial","optical"] = "binomial") -> None:
        if len(edges_percent) < 2: return
        
        print("--------------------------------------------------------------------------")
        print(f"Centrality  |  N_coll(mean)  |  L_eff [fm]   (method: {method})")
        print("--------------------------------------------------------------------------")
        for i in range(len(edges_percent) - 1):
            c0, c1 = edges_percent[i] / 100.0, edges_percent[i + 1] / 100.0
            if method == "optical":
                ncoll = self.ncoll_mean_bin_pA_optical(c0, c1)
            else:
                ncoll = self.ncoll_mean_bin_pA(c0, c1)
            leff  = self.leff_bin_pA(c0, c1, rho0_fm3, Lp_fm, method=method)
            print(f" {edges_percent[i]:>3.0f}-{edges_percent[i+1]:<3.0f}%   | "
                  f" {ncoll:10.3f}   |  {leff:8.3f}")
        print("--------------------------------------------------------------------------")
