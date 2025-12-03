## `gluon_ratio_fast.py` (final)
# -*- coding: utf-8 -*-
"""
gluon_ratio_fast.py — FINAL
==========================
EPPS21 gluon (any flavour) ratios + Woods–Saxon centrality helpers (GPU/CPU).

Drop-in classes preserved:
  - EPPS21Ratio
  - GluonEPPSProvider

Fixes:
  • Torch-only math in _nq_t (no numpy scalars) to avoid device/dtype mix
  • All index tensors (argx/argq) are on the SAME device as the EPPS grid
  • Convenience: K_ypt_b_set(s), SAWS/SA band helpers (68% default)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Iterable

import numpy as np

# optional torch
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

ArrayLike = Union[float, np.ndarray, "torch.Tensor"]

PDG_MASS: Dict[str, float] = {
    "J/psi": 3.0969, "psi(2S)": 3.6861,
    "Upsilon(1S)": 9.4603, "Upsilon(2S)": 10.0233, "Upsilon(3S)": 10.3552,
    "charmonium": 3.43, "bottomonium": 10.00,
}
FLAV_ID = {"uv":1,"dv":2,"u":3,"d":4,"s":5,"c":6,"b":7,"g":8,"gluon":8,"gl":8}
MEMBER_RANGES = {"nuclear": (2,49), "proton": (50,107), "all": (2,107)}

# -------------- tiny utils --------------

def _np(x: ArrayLike) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _same(x_like: ArrayLike, arr_np: np.ndarray) -> ArrayLike:
    if _HAS_TORCH and isinstance(x_like, torch.Tensor):
        return torch.as_tensor(arr_np, device=x_like.device, dtype=x_like.dtype)
    return arr_np

# --------------------- EPPS21 core ---------------------
@dataclass
class EPPS21Ratio:
    A: int
    path: Union[str, Path]
    device: Optional[str] = None     # "cuda","cpu", or None (auto)
    dtype: str = "float32"
    cache: bool = True
    cache_dir: Optional[str] = None

    _NSET: int = 107; _NFLAV: int = 8; _XSTEPS: int = 250; _QSTEPS: int = 30
    _XMIN: float = 1e-7; _Q2MIN: float = 1.690; _Q2MAX: float = 1.0e8
    _qrows: int = _QSTEPS + 1

    _grid_t: Optional["torch.Tensor"] = None
    _grid_n: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.device is None:
            self.device = ("cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu")
        if self.cache_dir is None:
            self.cache_dir = str(Path(self.path))
        self._dtype_t = (torch.float32 if self.dtype=="float32" else torch.float64) if _HAS_TORCH else None
        self._xi = float(self._XMIN)
        self._L0 = (np.log(1.0/self._xi) + 5.0*(1.0-self._xi))
        self._LSTEP = (0.0 - self._L0) / (1.0*self._XSTEPS)

    def _filename(self) -> Path:
        p = Path(self.path) / f"EPPS21NLOR_{int(self.A)}"
        if not p.exists():
            raise FileNotFoundError(f"[EPPS21] Missing {p.name} in {self.path}")
        return p

    def _cache_file(self) -> Path:
        base = f"EPPS21_{self.A}_grid_{self.dtype}.pt" if _HAS_TORCH else f"EPPS21_{self.A}_grid_{self.dtype}.npy"
        return Path(self.cache_dir) / base

    def _load_text_numpy(self) -> np.ndarray:
        qrows, xcols, nflav, nset = self._qrows, self._XSTEPS, self._NFLAV, self._NSET
        txt = self._filename().read_text().replace("D","E").replace("d","E")
        vals = np.fromstring(txt, sep=" ", dtype=np.float64)
        per_row = 1 + xcols*nflav
        per_set = qrows*per_row
        expected = nset*per_set
        if vals.size != expected:
            raise ValueError(f"[EPPS21] Parsed {vals.size}, expected {expected}")
        arr = vals.reshape(nset, qrows, per_row)[:, :, 1:]         # drop markers
        arr = arr.reshape(nset, qrows, xcols, nflav).transpose(0,3,1,2)  # (S,F,Q,X)
        return arr.astype(self.dtype, copy=False)

    def _ensure_loaded(self):
        if (self._grid_t is not None) or (self._grid_n is not None):
            return
        cf = self._cache_file()
        if self.cache and cf.exists():
            if _HAS_TORCH:
                self._grid_t = torch.load(cf, map_location=self.device)
            else:
                self._grid_n = np.load(cf)
            return
        grid_np = self._load_text_numpy()
        if _HAS_TORCH:
            self._grid_t = torch.from_numpy(grid_np).to(self.device, dtype=self._dtype_t)
            if self.cache:
                cf.parent.mkdir(parents=True, exist_ok=True); torch.save(self._grid_t, cf)
        else:
            self._grid_n = grid_np
            if self.cache:
                cf.parent.mkdir(parents=True, exist_ok=True); np.save(cf, self._grid_n)

    # index maps
    def _nx_np(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, self._XMIN, 1.0-1e-12)
        return ((np.log(1.0/x) + 5.0*(1.0-x)) - self._L0) / self._LSTEP
    def _nq_np(self, Q: np.ndarray) -> np.ndarray:
        Q2 = np.clip(Q*Q, self._Q2MIN, self._Q2MAX)
        num = np.log(np.log(Q2) / np.log(self._Q2MIN))
        den = np.log(np.log(self._Q2MAX) / np.log(self._Q2MIN))
        return self._QSTEPS * (num/den)
    def _nx_t(self, x: "torch.Tensor") -> "torch.Tensor":
        x = torch.clamp(x, min=self._XMIN, max=1.0-1e-12)
        L0 = torch.tensor(self._L0, device=x.device, dtype=x.dtype)
        LSTEP = torch.tensor(self._LSTEP, device=x.device, dtype=x.dtype)
        return ((torch.log(1.0/x) + 5.0*(1.0-x)) - L0) / LSTEP
    def _nq_t(self, Q: "torch.Tensor") -> "torch.Tensor":
        # torch-only path (no numpy scalars)
        Q2 = torch.clamp(Q*Q, min=self._Q2MIN, max=self._Q2MAX)
        Q2min = torch.tensor(self._Q2MIN, device=Q.device, dtype=Q.dtype)
        Q2max = torch.tensor(self._Q2MAX, device=Q.device, dtype=Q.dtype)
        num = torch.log(torch.log(Q2) / torch.log(Q2min))
        den = torch.log(torch.log(Q2max) / torch.log(Q2min))
        return self._QSTEPS * (num/den)

    # tiny 4-point Neville (vectorized)
    @staticmethod
    def _neville4_np(fNx4: np.ndarray, xNx4: np.ndarray, zN: np.ndarray) -> np.ndarray:
        P = fNx4.copy()
        for j in range(1,4):
            for i in range(0,4-j):
                P[:, i] = ((zN - xNx4[:, i+j]) * P[:, i] + (xNx4[:, i] - zN) * P[:, i+1]) / (xNx4[:, i] - xNx4[:, i+j])
        return P[:,0]
    @staticmethod
    def _neville4_t(fNx4: "torch.Tensor", xNx4: "torch.Tensor", zN: "torch.Tensor") -> "torch.Tensor":
        P = fNx4.clone()
        for j in range(1,4):
            for i in range(0,4-j):
                P[:, i] = ((zN - xNx4[:, i+j]) * P[:, i] + (xNx4[:, i] - zN) * P[:, i+1]) / (xNx4[:, i] - xNx4[:, i+j])
        return P[:,0]

    # core evaluator
    def ratio(self, flav: Union[int,str], x: ArrayLike, Q: ArrayLike, set: int = 1) -> ArrayLike:
        self._ensure_loaded()
        if isinstance(flav, str):
            if flav not in FLAV_ID: raise KeyError(f"Unknown flavour '{flav}'")
            flav = FLAV_ID[flav]
        if not (1 <= flav <= 8):
            raise ValueError("flav must be 1..8")
        if not (1 <= set <= self._NSET):
            raise ValueError(f"set must be 1..{self._NSET}")

        # torch fast path
        if _HAS_TORCH and (self._grid_t is not None):
            x_t = torch.as_tensor(x, device=self.device, dtype=self._dtype_t)
            Q_t = torch.as_tensor(Q, device=self.device, dtype=self._dtype_t)
            F   = self._grid_t[set-1, flav-1, :, :]                 # (Qrows, Xcols)
            xb, Qb = torch.broadcast_tensors(x_t, Q_t)
            xb1, Qb1 = xb.reshape(-1), Qb.reshape(-1)
            mask_bad = (xb1 < self._XMIN) | (xb1 >= 1.0) | (Qb1 < 1.3) | (Qb1 > 1.0e4)

            nx = self._nx_t(xb1); nq = self._nq_t(Qb1)
            xpoint = torch.clamp(torch.floor(nx).to(torch.int64), 1, self._XSTEPS-4)
            qpoint = torch.clamp(torch.floor(nq).to(torch.int64), 1, self._QSTEPS-2)

            # heavy-flavour tweaks (optional, kept for parity with your C++)
            if flav == 6:  # charm tweak
                qpoint = torch.where(qpoint == 1, torch.tensor(2, device=qpoint.device), qpoint)
            if flav == 7:  # bottom tweak
                m = (qpoint < 17) & (qpoint > 1)
                qpoint = torch.where(m, torch.tensor(17, device=qpoint.device), qpoint)

            argx = torch.stack([xpoint-1, xpoint, xpoint+1, xpoint+2], dim=1).to(device=F.device, dtype=torch.int64)
            argq = torch.stack([qpoint-1, qpoint, qpoint+1, qpoint+2], dim=1).to(device=F.device, dtype=torch.int64)

            # first interpolate along x for 4 q-rows, then along q
            gx = []
            for r in range(4):
                q_idx = argq[:, r]
                rows  = F.index_select(0, q_idx)               # (N, X)
                fNx4  = rows.gather(1, argx)                   # (N, 4)
                gx.append(self._neville4_t(fNx4, argx.to(xb1.dtype), nx))
            gNx4 = torch.stack(gx, dim=1)                      # (N, 4)
            out  = self._neville4_t(gNx4, argq.to(xb1.dtype), nq)

            if flav == 7:
                out = torch.where(Qb1 < 4.75, torch.zeros_like(out), out)
            out = torch.where(mask_bad, torch.full_like(out, float('nan')), out)
            return _same(x, out.reshape_as(xb).detach().cpu().numpy())

        # NumPy fallback
        F = self._grid_n[set-1, flav-1, :, :]                  # (Qrows, Xcols)
        x_n = _np(x).astype(self.dtype, copy=False); Q_n = _np(Q).astype(self.dtype, copy=False)
        xb, Qb = np.broadcast_arrays(x_n, Q_n); N = xb.size
        xb1, Qb1 = xb.reshape(-1), Qb.reshape(-1)
        mask_bad = (xb1 < self._XMIN) | (xb1 >= 1.0) | (Qb1 < 1.3) | (Qb1 > 1.0e4)

        nx = self._nx_np(xb1); nq = self._nq_np(Qb1)
        xpoint = np.clip(np.floor(nx).astype(np.int64), 1, self._XSTEPS-4)
        qpoint = np.clip(np.floor(nq).astype(np.int64), 1, self._QSTEPS-2)

        if flav == 6: qpoint = np.where(qpoint == 1, 2, qpoint)       # charm tweak
        if flav == 7:                                                 # bottom tweak
            m = (qpoint < 17) & (qpoint > 1)
            qpoint = np.where(m, 17, qpoint)

        argx = np.stack([xpoint-1, xpoint, xpoint+1, xpoint+2], axis=1)
        argq = np.stack([qpoint-1, qpoint, qpoint+1, qpoint+2], axis=1)

        gx = []
        for r in range(4):
            q_idx = argq[:, r]                         # (N,)
            rows  = F[q_idx, :]                        # (N, X)
            fNx4  = rows[np.arange(N)[:,None], argx]   # (N, 4)
            gx.append(self._neville4_np(fNx4, argx.astype(self.dtype), nx))
        gNx4 = np.stack(gx, axis=1)
        out = self._neville4_np(gNx4, argq.astype(self.dtype), nq)
        if flav == 7: out = np.where(Qb1 < 4.75, 0.0, out)
        out = np.where(mask_bad, np.nan, out)
        return _same(x, out.reshape(xb.shape))

    # (y,pT) helpers
    @staticmethod
    def xA_of(y: ArrayLike, pT: ArrayLike, sqrt_sNN_GeV: float,
              m_state_GeV: Union[str, float] = "charmonium", y_sign_for_xA: int = -1) -> ArrayLike:
        m = PDG_MASS.get(m_state_GeV, m_state_GeV)
        return _same(y, (2.0*np.hypot(m, _np(pT))/float(sqrt_sNN_GeV)) * np.exp(y_sign_for_xA * _np(y)))
    @staticmethod
    def Q_of(pT: ArrayLike, m_state_GeV: Union[str, float] = "charmonium") -> ArrayLike:
        m = PDG_MASS.get(m_state_GeV, m_state_GeV); return _same(pT, np.hypot(m, _np(pT)))

    def ratio_ypt(self, flav: Union[int,str], y: ArrayLike, pT: ArrayLike, sqrt_sNN_GeV: float, *,
                  set: int = 1, m_state_GeV: Union[str,float] = "charmonium", y_sign_for_xA: int = -1) -> ArrayLike:
        xA = self.xA_of(y, pT, sqrt_sNN_GeV, m_state_GeV, y_sign_for_xA)
        Q  = self.Q_of(pT, m_state_GeV)
        return self.ratio(flav, xA, Q, set=set)

    def ratio_ypt_sets(self, flav: Union[int,str], y: ArrayLike, pT: ArrayLike, sqrt_sNN_GeV: float, *,
                       m_state_GeV: Union[str,float] = "J/psi", y_sign_for_xA: int = -1,
                       source: str = "nuclear", include_central: bool = True):
        if source == "all":
            ids = ([1] + list(range(2,108))) if include_central else list(range(2,108))
        else:
            lo, hi = MEMBER_RANGES[source]
            ids = ([1] + list(range(lo,hi+1))) if include_central else list(range(lo,hi+1))
        xA = self.xA_of(y, pT, sqrt_sNN_GeV, m_state_GeV, y_sign_for_xA); Q = self.Q_of(pT, m_state_GeV)
        vals = []
        for sid in ids:
            vals.append(self.ratio(flav, xA, Q, set=sid))
        V = np.stack([_np(v) for v in vals], axis=0)
        return _same(xA, V), np.array(ids, int)

# ----------------- Woods–Saxon + centrality -----------------
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

    def rho(self, r): r = np.asarray(r, float); return self.rho0 / (1.0 + np.exp((r - self.R)/self.a))
    def thickness(self, b):
        b = float(b); zmax = self.zmax_mult * self.R
        z = np.linspace(-zmax, zmax, self.nz)
        return float(np.trapezoid(self.rho(np.hypot(b, z)), z))
    def T_grid(self):
        b = np.linspace(0.0, self.b_max, self.nb); T = np.array([self.thickness(bi) for bi in b], float); return b, T
    @staticmethod
    def sigma_mb_to_fm2(sigma_mb): return 0.1 * float(sigma_mb)
    def inel_pdf(self, sigmaNN_mb=70.0):
        b, T = self.T_grid(); s = self.sigma_mb_to_fm2(sigmaNN_mb)
        w = 2*np.pi*b*(1.0 - np.exp(-s*T)); Z = float(np.trapezoid(w, b)); w = w/Z if Z>0 else w
        return b, T, w
    def Nnorm(self):
        b, T = self.T_grid(); I = float(np.trapezoid(2*np.pi*b*(T**2), b)); return (self.A*float(T[0])) / I
    def alpha_of_b(self, b):
        B, T = self.T_grid(); T0 = float(T[0]); return float(np.interp(float(b), B, T))/T0 if T0>0 else 0.0

class GluonEPPSProvider:
    def __init__(self, epps: EPPS21Ratio, sqrt_sNN_GeV: float,
                 m_state_GeV: Union[str,float] = "charmonium", y_sign_for_xA: int = -1):
        self.epps = epps; self.sqrt = float(sqrt_sNN_GeV); self.m = m_state_GeV; self.sign = int(y_sign_for_xA)
        self.A = int(getattr(epps, "A", 208)); self._geom: Optional[WoodsSaxon] = None
    def with_geometry(self, geom: WoodsSaxon | None = None):
        self._geom = geom if geom is not None else WoodsSaxon(A=self.A); return self
    def _need_geom(self):
        if self._geom is None: self._geom = WoodsSaxon(A=self.A)
    def alpha_of_b(self, b): self._need_geom(); return self._geom.alpha_of_b(b)
    def Nnorm(self):        self._need_geom(); return self._geom.Nnorm()

    # --- S_A and S_A,WS (per-set) ---
    def SA_ypt_set(self, y_arr, pt_arr, set_id: int, flav: Union[int,str] = "g"):
        return self.epps.ratio_ypt(flav, y_arr, pt_arr, self.sqrt, set=set_id,
                                   m_state_GeV=self.m, y_sign_for_xA=self.sign)

    def SAWS_ypt_b_set(self, y, pT, b, *, set_id=1, alpha=None, Nnorm=None, flav: Union[int,str]="g"):
        self._need_geom()
        SA = self.SA_ypt_set(y, pT, set_id=set_id, flav=flav)
        if alpha is None: alpha = self.alpha_of_b(b)
        if Nnorm is None: Nnorm = self.Nnorm()
        return _same(SA, 1.0 + float(Nnorm) * (_np(SA) - 1.0) * float(alpha))

    def K_ypt_b_set(self, y, pT, b, *, set_id=1, alpha=None, Nnorm=None, flav: Union[int,str]="g"):
        """Return K = S_AWS / S_A for a single set."""
        SA   = self.SA_ypt_set(y, pT, set_id=set_id, flav=flav)
        SAWS = self.SAWS_ypt_b_set(y, pT, b, set_id=set_id, alpha=alpha, Nnorm=Nnorm, flav=flav)
        return _same(SA, _np(SAWS) / np.clip(_np(SA), 1e-12, None))

    # --- convenience: all sets (central + nuclear error members) ---
    def _ids(self, source: str = "nuclear", include_central: bool = True) -> np.ndarray:
        if source == "all":
            ids = ([1] + list(range(2,108))) if include_central else list(range(2,108))
        else:
            lo, hi = MEMBER_RANGES[source]
            ids = ([1] + list(range(lo,hi+1))) if include_central else list(range(lo,hi+1))
        return np.array(ids, int)

    def K_ypt_b_sets(self, y, pT, b, *, source: str = "nuclear", include_central: bool = True,
                      flav: Union[int,str] = "g"):
        ids = self._ids(source, include_central)
        out = []
        for sid in ids:
            out.append(_np(self.K_ypt_b_set(y, pT, b, set_id=sid, flav=flav)))
        return _same(_np(y) if not np.isscalar(y) else np.array([y]), np.stack(out, axis=0)), ids

    # --- Hessian bands (nuclear sets 2..49) ---
    @staticmethod
    def hessian_symmetric(center: np.ndarray, members_2to49: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 68% CL symmetric band from 24 eigenvector pairs. Shapes:
           center: (...,)
           members_2to49: (48, ...)
           Returns (lo, hi)."""
        M = members_2to49
        # pairwise diffs (2k-1, 2k): (2,3), (4,5), ... (48,49)
        D = M[0::2, ...] - M[1::2, ...]
        h = 0.5*np.sqrt(np.sum(D*D, axis=0))
        return center - h, center + h

    def SAWS_band_ypt_b(self, y, pT, b, *, cl: float = 68.0, flav: Union[int,str] = "g"):
        K_all, ids = self.K_ypt_b_sets(y, pT, b, source="nuclear", include_central=True, flav=flav)
        center = K_all[0]
        members = K_all[1:49]  # 48 nuclear error members
        lo, hi = self.hessian_symmetric(center, members)
        return lo, hi
