# -*- coding: utf-8 -*-
"""
EPPS21 gluon (and any flavour) nuclear ratios, Hessian uncertainties,
and centrality dependence with Woods–Saxon — final, robust module
===================================================================

This mirrors your C++ (EPPS21.cpp + gluonratios.cpp) and extends it with:
- explicit handling of 107 LHAPDF members (central + 48 nuclear + 58 proton)
- choice of uncertainty source: "nuclear", "proton", or "all" (quadrature)
- symmetric and asymmetric Hessian combinations (90% CL, optional 68% CL)
- (y,pT) helpers with your pA rapidity convention (A at negative y by default)
- centrality dependence: S_A, S_A,WS(b), and K(b)=S_A,WS/S_A using Woods–Saxon
- raw-member accessors and plotting utilities for debugging

Indexing (THIS MODULE IS 1-BASED TO MATCH YOUR PYTHON CODE):
  1            = central (EPPS21 central × CT18A central)
  2 … 49       = nuclear error members  (24 eigenvector pairs)
  50 … 107     = proton  error members  (29 eigenvector pairs)

So there are 53 pairs total: (2,3),(4,5),…,(106,107).

Public classes
--------------
- EPPS21Ratio:  low-level reader/interpolator for S_A^f(x,Q) for any flavour
- WoodsSaxon:   simple optical WS geometry with alpha(b) and normalization
- GluonEPPSProvider: high-level (y,pT) and centrality helpers for gluon S_A

Author: ChatGPT (faithful Python reimplementation + robustness)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

ArrayLike = Union[float, np.ndarray]

# ----------------------------- constants -----------------------------
# PDG-ish masses (GeV) for convenience
PDG_MASS: Dict[str, float] = {
    # charmonia
    "J/psi": 3.0969,
    "psi(2S)": 3.6861,
    # bottomonia
    "Upsilon(1S)": 9.4603,
    "Upsilon(2S)": 10.0233,
    "Upsilon(3S)": 10.3552,
    # categories (defaults): average mass
    "charmonium": 3.43,
    "bottomonium": 10.00,
}

# Flavour name → EPPS21 column (1..8)
FLAV_ID: Dict[str, int] = {
    "uv": 1, "dv": 2, "u": 3, "d": 4, "s": 5, "c": 6, "b": 7,
    "g": 8, "gluon": 8, "gl": 8,
}
_FLAV_LABEL = {i: n for n, i in FLAV_ID.items()}

# Member ranges (1-based here)
MEMBER_RANGES = {
    "nuclear": (2, 49),    # 48 members = 24 eigenvector pairs
    "proton" : (50, 107),  # 58 members = 29 eigenvector pairs
    "all"    : (2, 107),   # 106 members = 53 pairs
}


def _eigen_pairs(source: str) -> Iterable[Tuple[int, int]]:
    """Return list of (minus, plus) member-id pairs (1-based) for `source`."""
    if source not in ("nuclear", "proton", "all"):
        raise ValueError("source must be 'nuclear', 'proton', or 'all'")
    if source == "all":
        return list(_eigen_pairs("nuclear")) + list(_eigen_pairs("proton"))
    lo, hi = MEMBER_RANGES[source]
    s = lo
    out = []
    while s + 1 <= hi:
        out.append((s, s + 1))
        s += 2
    return out


def _corner(ax, text, loc="tr"):
    pos = {"tl": (0.02, 0.98), "tr": (0.98, 0.98), "bl": (0.02, 0.02), "br": (0.98, 0.02)}[loc]
    ha = "left" if "l" in loc else "right"
    va = "top" if "t" in loc else "bottom"
    ax.text(pos[0], pos[1], text, transform=ax.transAxes, ha=ha, va=va, fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))


# =============================== EPPS21 ===============================
@dataclass
class EPPS21Ratio:
    A: int                    # 197 (Au) or 208 (Pb)
    path: Union[str, Path]    # directory containing EPPS21NLOR_A

    # grid constants (match C++)
    _NSET: int = 107
    _NFLAV: int = 8
    _XSTEPS: int = 250
    _QSTEPS: int = 30
    _XMIN: float = 1e-7
    _Q2MIN: float = 1.690        # Q^2 min in GeV^2
    _Q2MAX: float = 1.0e8        # Q^2 max in GeV^2

    _grid: Optional[np.ndarray] = None  # shape: (NSET, NFLAV, QROWS=31, XSTEPS=250)

    # ---------- file ----------
    def _filename(self) -> Path:
        fname = f"EPPS21NLOR_{int(self.A)}"
        p = Path(self.path) / fname
        if not p.exists():
            raise FileNotFoundError(f"Could not find {fname} under {self.path}")
        return p

    def _ensure_loaded(self) -> None:
        if self._grid is not None:
            return
        qrows = self._QSTEPS + 1
        grid = np.empty((self._NSET, self._NFLAV, qrows, self._XSTEPS), float)

        def numbers():
            with open(self._filename(), "r") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    for tok in line.split():
                        yield float(tok.replace("D", "E"))

        it = numbers()
        for s in range(self._NSET):              # sets 0..106 (Python 0-based)
            for k in range(qrows):               # Q rows 0..30
                _ = next(it)                     # discard row marker
                for t in range(self._XSTEPS):    # x columns 0..249
                    for flav in range(self._NFLAV):
                        grid[s, flav, k, t] = next(it)
        self._grid = grid

    # ---------- Neville interpolation (as in luovi) ----------
    @staticmethod
    def _neville(f: np.ndarray, arg: np.ndarray, z: float) -> float:
        m = len(f)
        cof = f.astype(float).copy()
        for i in range(1, m):
            for j in range(i, m):
                idx = m - j - 1
                k = idx + i
                cof[k] = (cof[k] - cof[k - 1]) / (arg[k] - arg[idx])
        s = cof[-1]
        for i in range(1, m):
            k = m - i - 1
            s = (z - arg[k]) * s + cof[k]
        return float(s)

    # ---------- index maps (C++ formulas) ----------
    def _nx(self, x: float) -> float:
        x = float(np.clip(x, self._XMIN, 1.0 - 1e-12))
        xi = self._XMIN
        LSTEP = (0.0 - (np.log(1.0 / xi) + 5.0 * (1.0 - xi))) / (1.0 * self._XSTEPS)
        n_x = ((np.log(1.0 / x) + 5.0 * (1.0 - x)) -
               (np.log(1.0 / xi) + 5.0 * (1.0 - xi))) / LSTEP
        return float(n_x)

    def _nq(self, Q: float) -> float:
        Q2 = float(Q) * float(Q)
        Q2 = float(np.clip(Q2, self._Q2MIN, self._Q2MAX))
        num = np.log(np.log(Q2) / np.log(self._Q2MIN))
        den = np.log(np.log(self._Q2MAX) / np.log(self._Q2MIN))
        return float(self._QSTEPS * (num / den))

    # ---------- core evaluator ----------
    def ratio(self, flav: Union[int, str], x: ArrayLike, Q: ArrayLike, set: int = 1) -> ArrayLike:
        """R_f^A(x,Q) for flavour `flav` (1..8 or 'g','u','s','uv',...). 1-based set id."""
        self._ensure_loaded()
        if isinstance(flav, str):
            if flav not in FLAV_ID:
                raise KeyError(f"Unknown flavour '{flav}'. Valid: {list(FLAV_ID)}")
            flav = FLAV_ID[flav]
        if not (1 <= flav <= 8):
            raise ValueError("flav must be 1..8 or a known flavour name")
        if not (1 <= set <= self._NSET):
            raise ValueError(f"set must be 1..{self._NSET}")

        F = self._grid[set - 1, flav - 1, :, :]  # 0-based in numpy
        x_arr = np.asarray(x, float)
        Q_arr = np.asarray(Q, float)
        xb, Qb = np.broadcast_arrays(x_arr, Q_arr)
        out = np.empty_like(xb, float)

        qrows = self._QSTEPS + 1
        xcols = self._XSTEPS

        def interp(xx: float, QQ: float) -> float:
            # physical domain guard (like the C++)
            if (xx < self._XMIN) or (xx >= 1.0) or (QQ < 1.3) or (QQ > 1.0e4):
                return np.nan
            nx = self._nx(xx)
            nq = self._nq(QQ)
            xpoint = int(np.floor(nx))
            qpoint = int(np.floor(nq))
            # guardrails
            if xpoint == 0:
                xpoint = 1
            elif xpoint > (xcols - 4):
                xpoint = xcols - 4
            if qpoint == 0:
                qpoint = 1
            elif qpoint > (self._QSTEPS - 2):
                qpoint = self._QSTEPS - 2

            # heavy-flavour tweaks exactly as in C++
            charmflag = 0
            bottomflag = 0
            if flav == 6 and qpoint == 1:  # charm
                qpoint = 2
                charmflag = 1
            if flav == 7 and (qpoint < 17 and qpoint > 1):  # bottom
                bottomflag = qpoint
                qpoint = 17

            argx = np.array([xpoint - 1, xpoint, xpoint + 1, xpoint + 2], float)
            fg = []
            for qrow in (qpoint - 1, qpoint, qpoint + 1, qpoint + 2):
                fvals = F[qrow, xpoint - 1:xpoint + 3]
                fg.append(self._neville(fvals, argx, nx))
            argq = np.array([qpoint - 1, qpoint, qpoint + 1, qpoint + 2], float)
            res = self._neville(np.array(fg, float), argq, nq)

            if charmflag == 1:
                qpoint = 1
            if bottomflag > 1:
                qpoint = bottomflag
            if flav == 7 and QQ < 4.75:  # bottom below threshold
                return 0.0
            return float(res)

        it = np.nditer([xb, Qb, out], flags=['multi_index'],
                       op_flags=[['readonly'], ['readonly'], ['writeonly']])
        for xx, QQ, oo in it:
            oo[...] = interp(float(xx), float(QQ))
        return out if out.shape else float(out)

    # ---------- helpers over sets ----------
    def list_set_ids(self, source: str = "nuclear", include_central: bool = True):
        """Return 1-based member ids for source. If include_central, central is prepended."""
        if source not in ("nuclear", "proton", "all"):
            raise ValueError("source must be 'nuclear', 'proton', or 'all'")
        if source == "all":
            ids = list(range(2, 108))
        else:
            lo, hi = MEMBER_RANGES[source]
            ids = list(range(lo, hi + 1))
        return ([1] + ids) if include_central else ids

    def list_eigen_pairs(self, source: str = "nuclear"):
        return list(_eigen_pairs(source))

    def ratio_all_sets(self, flav: Union[int, str], x: ArrayLike, Q: ArrayLike) -> np.ndarray:
        xb, Qb = np.broadcast_arrays(np.asarray(x, float), np.asarray(Q, float))
        out = np.empty((self._NSET,) + xb.shape, float)
        for s in range(1, self._NSET + 1):
            out[s - 1, ...] = self.ratio(flav, xb, Qb, set=s)
        return out

    def ratio_sets(self, flav: Union[int, str], x: ArrayLike, Q: ArrayLike,
                   source: str = "nuclear", include_central: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (values, ids) where values has shape (Nsets, *xshape) in member-id order.
        """
        ids = self.list_set_ids(source, include_central=include_central)
        xb, Qb = np.broadcast_arrays(np.asarray(x, float), np.asarray(Q, float))
        val = np.empty((len(ids),) + xb.shape, float)
        for i, sid in enumerate(ids):
            val[i, ...] = self.ratio(flav, xb, Qb, set=sid)
        return val, np.array(ids, int)

    # ---------- (y,pT) kinematics ----------
    @staticmethod
    def xA_of(y: ArrayLike, pT: ArrayLike, sqrt_sNN_GeV: float,
              m_state_GeV: Union[str, float] = "charmonium", y_sign_for_xA: int = -1) -> ArrayLike:
        m = PDG_MASS.get(m_state_GeV, m_state_GeV)
        mT = np.hypot(m, np.asarray(pT, float))
        return (2.0 * mT / float(sqrt_sNN_GeV)) * np.exp(y_sign_for_xA * np.asarray(y, float))

    @staticmethod
    def Q_of(pT: ArrayLike, m_state_GeV: Union[str, float] = "charmonium") -> ArrayLike:
        m = PDG_MASS.get(m_state_GeV, m_state_GeV)
        return np.hypot(m, np.asarray(pT, float))

    def ratio_ypt(self, flav: Union[int, str], y: ArrayLike, pT: ArrayLike, sqrt_sNN_GeV: float,
                  *, set: int = 1, m_state_GeV: Union[str, float] = "charmonium",
                  y_sign_for_xA: int = -1) -> ArrayLike:
        xA = self.xA_of(y, pT, sqrt_sNN_GeV, m_state_GeV, y_sign_for_xA)
        Q = self.Q_of(pT, m_state_GeV)
        return self.ratio(flav, xA, Q, set=set)

    def ratio_ypt_sets(self, flav: Union[int, str], y: ArrayLike, pT: ArrayLike, sqrt_sNN_GeV: float,
                       *, m_state_GeV: Union[str, float] = "J/psi", y_sign_for_xA: int = -1,
                       source: str = "nuclear", include_central: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        ids = self.list_set_ids(source, include_central=include_central)
        xA = self.xA_of(y, pT, sqrt_sNN_GeV, m_state_GeV, y_sign_for_xA)
        Q = self.Q_of(pT, m_state_GeV)
        val = np.empty((len(ids),) + np.asarray(xA).shape, float)
        for i, sid in enumerate(ids):
            val[i, ...] = self.ratio(flav, xA, Q, set=sid)
        return val, np.array(ids, int)

    # ---------- Hessian uncertainties ----------
    def hessian_symm(self, flav, x, Q, *, cl: float = 90.0, source: str = "nuclear"):
        """Symmetric Hessian Δ for `source` in {"nuclear","proton","all"}."""
        def _symm_over(pairs):
            acc = 0.0
            for mminus, mplus in pairs:
                xm = self.ratio(flav, x, Q, set=mminus)
                xp = self.ratio(flav, x, Q, set=mplus)
                acc = acc + (xp - xm) ** 2
            d90 = 0.5 * np.sqrt(acc)
            return d90 if abs(cl - 68.0) > 1e-9 else d90 / 1.645

        if source == "all":
            dN = _symm_over(_eigen_pairs("nuclear"))
            dP = _symm_over(_eigen_pairs("proton"))
            return np.sqrt(dN ** 2 + dP ** 2)    # add in quadrature
        else:
            return _symm_over(_eigen_pairs(source))

    def hessian_asymm(self, flav, x, Q, *, cl: float = 90.0, source: str = "nuclear"):
        """
        Asymmetric Hessian (PDF 'master formula'):
            Δ^+ = sqrt( Σ_k [max(S_k^+ - S_0, S_k^- - S_0, 0)]^2 )
            Δ^- = sqrt( Σ_k [max(S_0 - S_k^+, S_0 - S_k^-, 0)]^2 )
        If source=='all', nuclear and proton components are added IN QUADRATURE
        separately for + and −.
        Returns (dm, dp) = (Δ^-, Δ^+).
        """
        S0 = self.ratio(flav, x, Q, set=1)

        def _one(source_part: str):
            ap = 0.0
            am = 0.0
            for mminus, mplus in _eigen_pairs(source_part):
                Sm = self.ratio(flav, x, Q, set=mminus)
                Sp = self.ratio(flav, x, Q, set=mplus)
                ap += np.maximum(np.maximum(Sp - S0, Sm - S0), 0.0) ** 2
                am += np.maximum(np.maximum(S0 - Sp, S0 - Sm), 0.0) ** 2
            ap = np.sqrt(ap)
            am = np.sqrt(am)
            if abs(cl - 68.0) < 1e-9:
                ap /= 1.645
                am /= 1.645
            return am, ap

        if source == "all":
            amN, apN = _one("nuclear")
            amP, apP = _one("proton")
            return np.sqrt(amN ** 2 + amP ** 2), np.sqrt(apN ** 2 + apP ** 2)
        else:
            return _one(source)

    # ---------- discovery ----------
    def n_sets(self) -> int:
        return self._NSET

    def list_flavours(self) -> Dict[str, int]:
        return dict(FLAV_ID)

    def list_sets(self) -> Dict[int, str]:
        out = {1: "central"}
        # 24 nuclear eigenvector pairs → ids 2..49
        for i in range(1, 25):
            out[2*i]   = f"S-{i}"
            out[2*i+1] = f"S+{i}"
        # 29 proton eigenvector pairs → ids 50..107 (continue numbering)
        for j in range(25, 54):
            out[2*j]   = f"S-{j}"
            out[2*j+1] = f"S+{j}"
        return out


    # ---------- plotting helpers ----------
    def _apply_grid(self, ax):
        ax.grid(True, which="major", ls=":", alpha=0.5)
        try:
            if ax.get_xscale() == "log":
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
        except Exception:
            pass
        try:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        except Exception:
            pass

    # Bands (choose source)
    def plot_SA_vs_x(self, flav: Union[int, str], x: np.ndarray, Q: float, *,
                     cl: float = 90.0, source: str = "nuclear",
                     annotate: bool = True, ylim=(0.3, 1.25)):
        fid = FLAV_ID.get(flav, flav)
        cen = self.ratio(fid, x, Q, set=1)
        d = self.hessian_symm(fid, x, Q, cl=cl, source=source)
        fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=180)
        ax.fill_between(x, cen - d, cen + d, alpha=0.25, step="mid",
                        label=f"EPPS21 {source} error (CL={int(cl)})")
        ax.plot(x, cen, lw=2, label="EPPS21 central")
        ax.set_xscale("log")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$S_A(x,Q^2)$")
        ax.set_ylim(*ylim)
        if annotate:
            _corner(ax, f"A={self.A}\nQ={Q:.3g} GeV, Q$^2$={Q*Q:.3g} GeV$^2$\n"
                        f"Flavor={_FLAV_LABEL[int(fid)]}", "tr")
        ax.legend(frameon=False)
        fig.tight_layout(); return fig, ax

    def plot_SA_vs_Q(self, flav: Union[int, str], Q: np.ndarray, x: float, *,
                     cl: float = 90.0, source: str = "nuclear",
                     annotate: bool = True, ylim=(0.3, 1.25)):
        fid = FLAV_ID.get(flav, flav)
        cen = self.ratio(fid, x, Q, set=1)
        d = self.hessian_symm(fid, x, Q, cl=cl, source=source)
        fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=180)
        ax.fill_between(Q, cen - d, cen + d, alpha=0.25, step="mid",
                        label=f"EPPS21 {source} error (CL={int(cl)})")
        ax.plot(Q, cen, lw=2, label="EPPS21 central")
        ax.set_xscale("log")
        ax.set_xlabel(r"$Q\, [\mathrm{GeV}]$")
        ax.set_ylabel(r"$S_A(x,Q^2)$")
        ax.set_ylim(*ylim)
        if annotate:
            _corner(ax, f"A={self.A}\nx={x:.2e}\nFlavor={_FLAV_LABEL[int(fid)]}", "br")
        ax.legend(frameon=False)
        fig.tight_layout(); return fig, ax

    def plot_SA_members_vs_y(self, flav: Union[int, str], y: np.ndarray, pT: float,
                             sqrt_sNN_GeV: float, *, m_state_GeV: Union[str, float] = "J/psi",
                             y_sign_for_xA: int = -1, source: str = "nuclear",
                             alpha_members: float = 0.35, lw_members: float = 1.0,
                             show_central: bool = True, ylim=(0.3, 1.25)):
        """Raw member curves (great for debugging)."""
        fid = FLAV_ID.get(flav, flav)
        xA = self.xA_of(y, pT, sqrt_sNN_GeV, m_state_GeV, y_sign_for_xA)
        Q = self.Q_of(pT, m_state_GeV)
        ids = self.list_set_ids(source, include_central=False)
        fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=180)
        for sid in ids:
            Sa = self.ratio(fid, xA, Q, set=sid)
            ax.plot(y, Sa, lw=lw_members, alpha=alpha_members)
        if show_central:
            Sc = self.ratio(fid, xA, Q, set=1)
            ax.plot(y, Sc, lw=2.0, label="central", zorder=5)
        ax.set_xlabel(r"$y$"); ax.set_ylabel(r"$S_A(y,p_T)$"); ax.set_ylim(*ylim)
        ax.legend(frameon=False)
        fig.tight_layout(); return fig, ax


# ============================ Woods–Saxon =============================
@dataclass
class WoodsSaxon:
    A: int = 208
    R: float = 6.624
    a: float = 0.549
    rho0: float = 0.17
    zmax_mult: float = 10.0
    nz: int = 2001
    b_max: float = 20.0
    nb: int = 2001

    def rho(self, r):
        r = np.asarray(r, float)
        return self.rho0 / (1.0 + np.exp((r - self.R) / self.a))

    def thickness(self, b):
        b = float(b)
        zmax = self.zmax_mult * self.R
        z = np.linspace(-zmax, zmax, self.nz)
        return float(np.trapezoid(self.rho(np.hypot(b, z)), z))

    def T_grid(self):
        b = np.linspace(0.0, self.b_max, self.nb)
        T = np.array([self.thickness(bi) for bi in b], float)
        return b, T

    @staticmethod
    def sigma_mb_to_fm2(sigma_mb):  # 1 mb = 0.1 fm^2
        return 0.1 * float(sigma_mb)

    def inel_pdf(self, sigmaNN_mb=70.0):
        b, T = self.T_grid()
        s = self.sigma_mb_to_fm2(sigmaNN_mb)
        w = 2 * np.pi * b * (1.0 - np.exp(-s * T))
        Z = float(np.trapezoid(w, b))
        w = w / Z if Z > 0 else w
        return b, T, w

    def Nnorm(self):
        b, T = self.T_grid()
        I = float(np.trapezoid(2 * np.pi * b * (T ** 2), b))
        return (self.A * float(T[0])) / I   # enforces <K>_b = 1

    def alpha_of_b(self, b):
        B, T = self.T_grid()
        T0 = float(T[0])
        return float(np.interp(float(b), B, T)) / T0 if T0 > 0 else 0.0

    def alpha_bar_for_bin(self, pL, pR, sigmaNN_mb=70.0):
        b, T, w = self.inel_pdf(sigmaNN_mb)
        T0 = float(T[0])
        cdf = np.r_[0.0, np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(b))]

        def inv_cdf(p):
            j = int(np.searchsorted(cdf, p, 'left'))
            if j <= 0:
                return float(b[0])
            if j >= len(b):
                return float(b[-1])
            f = 0.0 if (cdf[j] - cdf[j - 1]) == 0 else (p - cdf[j - 1]) / (cdf[j] - cdf[j - 1])
            return float((1 - f) * b[j - 1] + f * b[j])

        bl, br = inv_cdf(pL / 100), inv_cdf(pR / 100)
        m = (b >= bl) & (b < br)
        den = float(np.trapezoid(w[m], b[m]))
        num = float(np.trapezoid((T[m] / T0) * w[m], b[m]))
        return (num / den) if den > 0 else np.nan


# ========================= GluonEPPSProvider =========================
class GluonEPPSProvider:
    """
    High-level helpers specifically for the gluon flavour (but you can change
    'g' to another flavour id/name if desired).
    """
    def __init__(self, epps: EPPS21Ratio, sqrt_sNN_GeV: float,
                 m_state_GeV: Union[str, float] = "charmonium", y_sign_for_xA: int = -1):
        self.epps = epps
        self.sqrt = float(sqrt_sNN_GeV)
        self.m = m_state_GeV
        self.sign = int(y_sign_for_xA)
        self.A = int(getattr(epps, "A", 208))
        self._geom: Optional[WoodsSaxon] = None

    # --- geometry ---
    def with_geometry(self, geom: WoodsSaxon | None = None):
        self._geom = geom if geom is not None else WoodsSaxon(A=self.A)
        return self

    def _need_geom(self):
        if self._geom is None:
            self._geom = WoodsSaxon(A=self.A)

    def alpha_of_b(self, b): self._need_geom(); return self._geom.alpha_of_b(b)
    def Nnorm(self):        self._need_geom(); return self._geom.Nnorm()

    # --- EPPS21 gluon S_A ---
    def SA_ypt_set(self, y_arr, pt_arr, set_id: int, flav: Union[int, str] = "g"):
        return self.epps.ratio_ypt(flav, np.asarray(y_arr), np.asarray(pt_arr),
                                   self.sqrt, set=set_id,
                                   m_state_GeV=self.m, y_sign_for_xA=self.sign)

    def SA_ypt_sets(self, y, pT, *, source: str = "nuclear",
                    include_central: bool = True, flav: Union[int, str] = "g"):
        return self.epps.ratio_ypt_sets(flav, y, pT, self.sqrt,
                                        m_state_GeV=self.m, y_sign_for_xA=self.sign,
                                        source=source, include_central=include_central)

    # --- centrality dependence ---
    def SAWS_ypt_b_set(self, y, pT, b, *, set_id=1, alpha=None, Nnorm=None,
                        flav: Union[int, str] = "g"):
        """
        S_A,WS(b;y,pT) = 1 + Nnorm * (S_A - 1) * alpha(b), with central S_A from chosen set.
        """
        self._need_geom()
        SA = self.SA_ypt_set(y, pT, set_id=set_id, flav=flav)
        if alpha is None:
            alpha = self.alpha_of_b(b)
        if Nnorm is None:
            Nnorm = self.Nnorm()
        S_AWS = 1.0 + float(Nnorm) * (SA - 1.0) * float(alpha)
        # print("DEBUG: SAWS_ypt_b_set: set_id={}, b={}, alpha={}, Nnorm={}, S_A={}, S_AWS={}".format(
        #    set_id, b, alpha, Nnorm, SA, S_AWS))
        return S_AWS

    def SAWS_ypt_b_sets(self, y, pT, b, *, source: str = "nuclear",
                         include_central: bool = True, flav: Union[int, str] = "g"):
        """Raw member curves of S_A,WS(b). Useful for debugging."""
        ids = self.epps.list_set_ids(source, include_central=include_central)
        SAWS = []
        for sid in ids:
            SAWS.append(self.SAWS_ypt_b_set(y, pT, b, set_id=sid, flav=flav))
        return np.asarray(SAWS), np.array(ids, int)

    def K_ypt_b_set(self, y, pT, b, *, set_id=1, alpha=None, Nnorm=None,
                     flav: Union[int, str] = "g"):
        SA = self.SA_ypt_set(y, pT, set_id=set_id, flav=flav)
        SAWS = self.SAWS_ypt_b_set(y, pT, b, set_id=set_id, alpha=alpha, Nnorm=Nnorm, flav=flav)
        return SAWS / np.clip(SA, 1e-12, None)

    def K_of(self, y: float, pT: float, b_val: float) -> float:
        if self._matched and "epps" in self._matched and hasattr(self._matched["epps"], "K_ypt_b_set"):
            return float(self._matched["epps"].K_ypt_b_set(y, pT, b_val, set_id=1))
        if hasattr(self.gluon, "K_ypt_b_set"):
            return float(self.gluon.K_ypt_b_set(y, pT, b_val, set_id=1))
        # fallback: compute (1+N*(S_A-1)*alpha)/S_A locally
        SA = (float(self.gluon.SA_ypt(y, pT)) if hasattr(self.gluon,"SA_ypt")
            else float(np.asarray(self.gluon.SA_ypt_set([y],[pT], set_id=1), float)[0]))
        SA = max(SA, 1e-12)
        alpha = self.alpha_of_b(float(b_val))
        return (1.0 + self.Nnorm*(SA-1.0)*alpha) / SA

    # --- Hessian bands for SAWS and K (choose uncertainty source) ---
    def SAWS_band_ypt_b(self, y, pT, b, *, cl=68.0, mode="asymm",
                        alpha=None, Nnorm=None, source="nuclear",
                        flav: Union[int, str] = "g"):
        """
        Error band for S_A,WS at fixed b. `source` in {"nuclear","proton","all"}.
        `mode` in {"symm","asymm"} (asymmetric is PDF master formula).
        """
        self._need_geom()
        y = np.asarray(y, float); pT = np.asarray(pT, float)
        if alpha is None:
            alpha = self.alpha_of_b(b)
        if Nnorm is None:
            Nnorm = self.Nnorm()
            
        # central curves
        Sc = self.SA_ypt_set(y, pT, set_id=1, flav=flav)
        Wc = 1.0 + Nnorm * (Sc - 1.0) * alpha

        if mode == "symm":
            def _symm_for(source_part: str):
                acc = 0.0
                for mminus, mplus in _eigen_pairs(source_part):
                    Sm = self.SA_ypt_set(y, pT, set_id=mminus, flav=flav); Wm = 1 + Nnorm * (Sm - 1) * alpha
                    Sp = self.SA_ypt_set(y, pT, set_id=mplus,  flav=flav); Wp = 1 + Nnorm * (Sp - 1) * alpha
                    acc += (Wp - Wm) ** 2
                d90 = 0.5 * np.sqrt(acc); d = d90 if cl == 90 else d90 / 1.645
                return d

            if source == "all":
                d = np.sqrt(_symm_for("nuclear") ** 2 + _symm_for("proton") ** 2)
            else:
                d = _symm_for(source)
            return Wc - d, Wc + d

        # asymmetric
        def _asymm_for(source_part: str):
            ap = 0.0; am = 0.0
            for mminus, mplus in _eigen_pairs(source_part):
                Sm = self.SA_ypt_set(y, pT, set_id=mminus, flav=flav); Wm = 1 + Nnorm * (Sm - 1) * alpha
                Sp = self.SA_ypt_set(y, pT, set_id=mplus,  flav=flav); Wp = 1 + Nnorm * (Sp - 1) * alpha
                # three-way max → nest two-arg maxima
                ap += np.maximum(np.maximum(Wp - Wc, Wm - Wc), 0.0) ** 2
                am += np.maximum(np.maximum(Wc - Wp, Wc - Wm), 0.0) ** 2
            k = 1.0 if cl == 90 else (1.0 / 1.645)
            return k * np.sqrt(am), k * np.sqrt(ap)  # (dm, dp)

        if source == "all":
            dmN, dpN = _asymm_for("nuclear")
            dmP, dpP = _asymm_for("proton")
            dm = np.sqrt(dmN ** 2 + dmP ** 2)
            dp = np.sqrt(dpN ** 2 + dpP ** 2)
        else:
            dm, dp = _asymm_for(source)
        return Wc - dm, Wc + dp

    def K_band_ypt_b(self, y, pT, b, *, cl=68.0, mode="asymm",
                      alpha=None, Nnorm=None, source="nuclear",
                      flav: Union[int, str] = "g"):
        """
        Error band for K(b)=S_A,WS/S_A (same options as SAWS_band_ypt_b).
        """
        self._need_geom()
        y = np.asarray(y, float); pT = np.asarray(pT, float)
        if alpha is None:
            alpha = self.alpha_of_b(b)
        if Nnorm is None:
            Nnorm = self.Nnorm()

        SA_c = self.SA_ypt_set(y, pT, set_id=1, flav=flav)
        Wc = 1.0 + Nnorm * (SA_c - 1.0) * alpha
        Kc = Wc / np.clip(SA_c, 1e-12, None)

        if mode == "symm":
            def _symm_for(source_part: str):
                acc = 0.0
                for mminus, mplus in _eigen_pairs(source_part):
                    SA_m = self.SA_ypt_set(y, pT, set_id=mminus, flav=flav)
                    SA_p = self.SA_ypt_set(y, pT, set_id=mplus,  flav=flav)
                    Km = (1 + Nnorm * (SA_m - 1) * alpha) / np.clip(SA_m, 1e-12, None)
                    Kp = (1 + Nnorm * (SA_p - 1) * alpha) / np.clip(SA_p, 1e-12, None)
                    acc += (Kp - Km) ** 2
                d90 = 0.5 * np.sqrt(acc); d = d90 if cl == 90 else d90 / 1.645
                return d

            if source == "all":
                d = np.sqrt(_symm_for("nuclear") ** 2 + _symm_for("proton") ** 2)
            else:
                d = _symm_for(source)
            return Kc - d, Kc + d

        # asymmetric
        def _asymm_for(source_part: str):
            ap = 0.0; am = 0.0
            for mminus, mplus in _eigen_pairs(source_part):
                SA_m = self.SA_ypt_set(y, pT, set_id=mminus, flav=flav)
                SA_p = self.SA_ypt_set(y, pT, set_id=mplus,  flav=flav)
                Km = (1 + Nnorm * (SA_m - 1) * alpha) / np.clip(SA_m, 1e-12, None)
                Kp = (1 + Nnorm * (SA_p - 1) * alpha) / np.clip(SA_p, 1e-12, None)
                # three-way max → nest two-arg maxima
                ap += np.maximum(np.maximum(Kp - Kc, Km - Kc), 0.0) ** 2
                am += np.maximum(np.maximum(Kc - Kp, Kc - Km), 0.0) ** 2
            k = 1.0 if cl == 90 else (1.0 / 1.645)
            return k * np.sqrt(am), k * np.sqrt(ap)


        if source == "all":
            dmN, dpN = _asymm_for("nuclear")
            dmP, dpP = _asymm_for("proton")
            dm = np.sqrt(dmN ** 2 + dmP ** 2)
            dp = np.sqrt(dpN ** 2 + dpP ** 2)
        else:
            dm, dp = _asymm_for(source)
        return Kc - dm, Kc + dp

    # --- Quick plotters for debugging/presentations ---
    def plot_SAWS_vs_y(self, y, pT, b, *, cl=68.0, source="nuclear",
                       ylim=(0.5, 1.3), flav: Union[int, str] = "g"):
        SAWS_lo, SAWS_hi = self.SAWS_band_ypt_b(y, pT, b, cl=cl, source=source, flav=flav)
        SAWS_c = self.SAWS_ypt_b_set(y, pT, b, set_id=1, flav=flav)
        fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=180)
        ax.fill_between(y, SAWS_lo, SAWS_hi, alpha=0.25, label=f"{source} error")
        ax.plot(y, SAWS_c, lw=2, label="central")
        ax.set_xlabel(r"$y$"); ax.set_ylabel(r"$S_{A,\mathrm{WS}}(y,p_T;b)$"); ax.set_ylim(*ylim)
        ax.legend(frameon=False); fig.tight_layout(); return fig, ax

    def plot_SAWS_over_SA_vs_y(self, y, pT, b, *, ylim=(0.8, 1.2), flav: Union[int, str] = "g"):
        SA = self.SA_ypt_set(y, pT, set_id=1, flav=flav)
        SAWS = self.SAWS_ypt_b_set(y, pT, b, set_id=1, flav=flav)
        K = SAWS / np.clip(SA, 1e-12, None)
        fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=180)
        ax.plot(y, K, lw=2)
        ax.set_xlabel(r"$y$"); ax.set_ylabel(r"$S_{A,\mathrm{WS}}/S_A \equiv K$"); ax.set_ylim(*ylim)
        fig.tight_layout(); return fig, ax

    def plot_K_vs_b(self, b_grid, *, y, pT, ylim=(0.8, 1.2), flav: Union[int, str] = "g"):
        Kc = [self.K_ypt_b_set(y, pT, b, set_id=1, flav=flav) for b in b_grid]
        fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=180)
        ax.plot(b_grid, Kc, lw=2)
        ax.set_xlabel(r"$b$ [fm]"); ax.set_ylabel(r"$K(b;y,p_T)$"); ax.set_ylim(*ylim)
        fig.tight_layout(); return fig, ax


# ------------------------- quick CLI self-test -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print S_A^f(x,Q) for a given member.")
    ap.add_argument("A", type=int, choices=[197, 208], help="197=Au, 208=Pb")
    ap.add_argument("path", type=str, help="Directory containing EPPS21NLOR_A")
    ap.add_argument("x", type=float)
    ap.add_argument("Q", type=float)
    ap.add_argument("--flav", type=str, default="g", help="flavour (name or 1..8)")
    ap.add_argument("--set", type=int, default=1, help="1-based member id (1=central)")
    args = ap.parse_args()

    epps = EPPS21Ratio(A=args.A, path=args.path)
    print(epps.ratio(args.flav, args.x, args.Q, set=args.set))
