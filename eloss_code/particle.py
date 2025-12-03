# Quarkonia particle helper (mass, kinematics, pp spectra)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Tuple
import numpy as np

Family = Literal["bottomonia", "charmonia"]

# ---- constants ----
HBARC_GeV_fm = 0.1973269804
M_PROTON_GeV = 0.938

# ---- masses (GeV) matching your C++ defaults ----
_BOTTOM_AVG = 10.01
_BOTTOM_STATES: Dict[str, float] = {"1S": 9.4603, "2S": 10.02326, "3S": 10.3552}

_CHARMO_AVG  = 3.425
_CHARMO_STATES: Dict[str, float] = {"Jpsi": 3.0969, "psi2S": 3.6861}

# ---- pp spectrum shape: (p0, m, n) ----
_PP_DEFAULTS: Dict[Family, Tuple[float, float, float]] = {
    "bottomonia": (6.6, 2.8, 13.8),
    "charmonia":  (4.2, 3.5, 19.2),
}

def _tag(fam: Family, state: Optional[str]) -> str:
    if fam == "charmonia": return "JPsi"
    if (state is None) or (str(state).lower() == "avg"): return "Upsilon"
    s = str(state).upper()
    return f"Upsilon{s}" if s in ("1S","2S","3S") else "Upsilon"

def _mass(fam: Family, state: Optional[str], override: Optional[float]) -> float:
    if override is not None: return float(override)
    if fam == "bottomonia":
        if state is None or str(state).lower()=="avg": return _BOTTOM_AVG
        return float(_BOTTOM_STATES.get(str(state).upper(), _BOTTOM_AVG))
    # charmonia
    if state is None or str(state).lower()=="avg": return _CHARMO_AVG
    return float(_CHARMO_STATES.get(str(state), _CHARMO_AVG))

@dataclass(frozen=True)
class PPSpectrumParams:
    p0: float; m: float; n: float
    @staticmethod
    def from_family(fam: Family) -> "PPSpectrumParams":
        p0, m, n = _PP_DEFAULTS[fam]; return PPSpectrumParams(p0, m, n)

@dataclass(frozen=True)
class Particle:
    """All particle-dependent pieces: identity, mass, 2→1 kinematics, xA/xB, pp spectra."""
    family: Family                            # "bottomonia" | "charmonia"
    state: Optional[str] = "avg"              # "avg" | "1S"/"2S"/"3S" | "Jpsi"/"psi2S"
    mass_override_GeV: Optional[float] = None
    pp_params: Optional[PPSpectrumParams] = None

    # ---- identity ----
    @property
    def tag(self) -> str: return _tag(self.family, self.state)

    @property
    def M_GeV(self) -> float: return _mass(self.family, self.state, self.mass_override_GeV)

    @property
    def pp(self) -> PPSpectrumParams:
        return self.pp_params if self.pp_params is not None else PPSpectrumParams.from_family(self.family)

    # ---- kinematics ----
    def mT(self, pT_GeV) -> np.ndarray:
        pT = np.asarray(pT_GeV, float)
        return np.sqrt(self.M_GeV**2 + pT**2)

    @staticmethod
    def y_max(roots_GeV: float, mT_GeV) -> np.ndarray:
        return np.log(np.asarray(roots_GeV, float) / np.asarray(mT_GeV, float))

    # 2→1 model: x1=(M⊥/√s)e^{+y}, x2=(M⊥/√s)e^{-y}
    @staticmethod
    def x1x2_2to1(self, y, pT_GeV, roots_GeV) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, float)
        mT = self.mT(pT_GeV)
        fac = mT/float(roots_GeV)
        return fac*np.exp(+y), fac*np.exp(-y)

    # @staticmethod # 2-->1 process, x_1 = mT/sqrt(s_NN) * exp(+y)
    def x1(self, y, pT_GeV, roots_GeV) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, float)
        mT = self.mT(pT_GeV)
        fac = mT/float(roots_GeV)
        return fac*np.exp(+y)
    ## Equation 44
    # @staticmethod # 2-->1 process, x_2, x_2 = mT/sqrt(s_NN) * exp(-y)
    def x2(self, y, pT_GeV, roots_GeV) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, float)
        mT = self.mT(pT_GeV)
        fac = mT/float(roots_GeV)
        return fac*np.exp(-y)
        
    ## Centrality Dependent x_0 = 1 / (2 m_p_GeV * L_eff_fm / ħc)
    @staticmethod
    def xA0_from_LA(LA_fm: float, m_p_GeV: float = M_PROTON_GeV) -> float:
        # x_{A0} = 1 / (2 m_p L_A / ħc)
        return 1.0 / (2.0 * m_p_GeV * (LA_fm / HBARC_GeV_fm))
    
    @staticmethod
    def xB0_from_LB(LB_fm: float, m_p_GeV: float = M_PROTON_GeV) -> float:
        # x_{A0} = 1 / (2 m_p L_B / ħc)
        return 1.0 / (2.0 * m_p_GeV * (LB_fm / HBARC_GeV_fm))
    # A/B sides
    #  x_A = min(x_A0, x_2);  x_B = min(x_B0, x_2)
    # @staticmethod
    def xA(self, y, pT_GeV, roots_GeV, LA_fm: float):  
        x2 = self.x2(y, pT_GeV, roots_GeV)
        x0 = self.xA0_from_LA(LA_fm)
        return np.minimum(x2, x0)
    # x_B = min(x_B0, x_2)
    # @staticmethod
    def xB(self, y, pT_GeV, roots_GeV, LB_fm: float): 
        x2 = self.x2(y, pT_GeV, roots_GeV)
        x0 = self.xB0_from_LB(LB_fm)
        return np.minimum(x2, x0)

    # ---- pp spectrum (shape only; overall N handled elsewhere) ----
    def _F1(self, pT_GeV):
        p0, m, _ = self.pp.p0, self.pp.m, self.pp.n
        pT = np.asarray(pT_GeV, float)
        return (p0*p0/(p0*p0 + pT*pT))**m

    def _F2(self, y, pT_GeV, roots_GeV):
        _, _, n = self.pp.p0, self.pp.m, self.pp.n
        y = np.asarray(y, float); mT = self.mT(pT_GeV)
        arg = 1.0 - (2.0*mT/float(roots_GeV))*np.cosh(y)
        return np.where(arg > 0.0, arg, 1e-30)**n  # stable near boundary

    def d2sigma_pp(self, y, pT_GeV, roots_GeV):
        mT = self.mT(pT_GeV); ymax = self.y_max(roots_GeV, mT)
        inside = np.abs(np.asarray(y, float)) <= ymax
        val = self._F1(pT_GeV) * self._F2(y, pT_GeV, roots_GeV)
        return np.where(inside, val, 1e-30)

    # ---- small printout ----
    def print_summary(self, roots_GeV: float | None = None) -> None:
        p0, m, n = self.pp.p0, self.pp.m, self.pp.n
        srt = f", sqrt(s)={roots_GeV:g} GeV" if roots_GeV is not None else ""
        print(f"[Particle] {self.tag}  M={self.M_GeV:.4f} GeV  pp:(p0={p0:g}, m={m:g}, n={n:g}){srt}")
