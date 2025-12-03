# primordial_code/tamu_traj_primordial.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Dict, Iterable, Optional

import math
import numpy as np
import pandas as pd

HBARC = 0.1973269804  # GeV * fm

# ---------------------------------------------------------------------
#  Quarkonium state definition (for charmonium here)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class QuarkoniumState:
    """Single quarkonium species."""
    name: str           # e.g. "1S", "1P", "2S"
    mass_GeV: float     # vacuum mass
    tau_form_fm: float  # proper formation time [fm/c]
    rate_index: int     # which column in rate table (0→1S, 1→1P, 2→2S for charmonium)


# Charmonium defaults (you can tweak these centrally)
CHARM_STATES_DEFAULT: List[QuarkoniumState] = [
    QuarkoniumState("1S", mass_GeV=3.097, tau_form_fm=1.0, rate_index=0),   # J/ψ
    QuarkoniumState("1P", mass_GeV=3.5,   tau_form_fm=1.5, rate_index=1),   # χc(1P)
    QuarkoniumState("2S", mass_GeV=3.686, tau_form_fm=2.5, rate_index=2),   # ψ(2S)
]

# Map our 3 "physics" states to the 5 TAMU output columns expected
# by primordial_module: [jpsi_1S, chic0_1P, chic1_1P, chic2_1P, psi_2S]
CHARM_OUTPUT_MAP = {
    "1S": ["jpsi_1S"],
    "1P": ["chic0_1P", "chic1_1P", "chic2_1P"],
    "2S": ["psi_2S"],
}

# ---------------------------------------------------------------------
#  Rate table: bilinear interpolation of Gamma(T, pT)
# ---------------------------------------------------------------------

class CharmRateTable:
    """
    Γ_ψ(T, pT) for charmonium, from TSV like 'charm_rates_pert.tsv'
    with columns:
        T  pT  Gamma_1S  Gamma_1P  Gamma_2S
    """

    def __init__(self, path: str):
        df = pd.read_csv(
            path, sep=r"\s+", header=None,
            names=["T", "pT", "G1S", "G1P", "G2S"],
        )
        df = df.sort_values(["T", "pT"]).reset_index(drop=True)

        Tvals = np.unique(df["T"].to_numpy())
        pTvals = np.unique(df["pT"].to_numpy())
        nT, nP = Tvals.size, pTvals.size
        if nT * nP != len(df):
            raise RuntimeError("Rate table is not on a regular T×pT grid.")

        # Reshape to (nT, nP)
        G1S = df["G1S"].to_numpy().reshape(nT, nP)
        G1P = df["G1P"].to_numpy().reshape(nT, nP)
        G2S = df["G2S"].to_numpy().reshape(nT, nP)
        self.T_grid = Tvals
        self.pT_grid = pTvals
        # gamma_table[state_index, iT, jP]
        self.gamma_table = np.stack([G1S, G1P, G2S], axis=0).astype(float)

    def _bilinear_single(self, state_idx: int, T: float, pT: float) -> float:
        """
        Bilinear interpolation for one (T,pT) point and one state index.
        """
        T_arr = self.T_grid
        p_arr = self.pT_grid
        G = self.gamma_table[state_idx]

        # Clamp to grid bounds
        Tq = float(np.clip(T, T_arr[0], T_arr[-1]))
        pTq = float(np.clip(pT, p_arr[0], p_arr[-1]))

        # Find neighboring indices
        i = np.searchsorted(T_arr, Tq) - 1
        j = np.searchsorted(p_arr, pTq) - 1
        i = int(np.clip(i, 0, len(T_arr) - 2))
        j = int(np.clip(j, 0, len(p_arr) - 2))

        T0, T1 = T_arr[i], T_arr[i + 1]
        p0, p1 = p_arr[j], p_arr[j + 1]
        g00 = G[i, j]
        g01 = G[i, j + 1]
        g10 = G[i + 1, j]
        g11 = G[i + 1, j + 1]

        # Local coordinates
        tx = 0.0 if T1 == T0 else (Tq - T0) / (T1 - T0)
        ty = 0.0 if p1 == p0 else (pTq - p0) / (p1 - p0)

        return (
            (1 - tx) * (1 - ty) * g00 +
            tx * (1 - ty) * g10 +
            (1 - tx) * ty * g01 +
            tx * ty * g11
        )

    def gamma(self, state_idx: int, T: float, pT: float) -> float:
        """Return Γ for one state index at (T, pT)."""
        return self._bilinear_single(state_idx, T, pT)


# ---------------------------------------------------------------------
#  Trajectory object (per aHydro trajectory file)
# ---------------------------------------------------------------------

@dataclass
class Trajectory:
    """
    One primordial quarkonium trajectory through the QGP.

    T_profile: temperature along the trajectory [GeV]
    t0_fm    : initial time [fm/c]
    dt_fm    : step size [fm/c]
    Tf       : final QGP temperature [GeV] – we stop when T<Tf

    Metadata (b, pt, y, phi, weight) are carried to the output.
    """
    T_profile: np.ndarray
    t0_fm: float
    dt_fm: float
    Tf: float

    b: float
    pt: float
    y: float
    phi: float = 0.0
    weight: float = 1.0

    def active_length(self) -> int:
        """
        Number of steps until T first drops below Tf (like maxIdx in C++).
        """
        T = self.T_profile
        mask = (T > self.Tf)
        if not np.any(mask):
            return 0
        # last index where T>Tf, +1 to include it
        last = int(np.where(mask)[0].max())
        return last + 1

    @property
    def nsteps(self) -> int:
        return self.T_profile.size


# ---------------------------------------------------------------------
#  Core solver: integrate dN/dτ = -Γ N along each trajectory
# ---------------------------------------------------------------------

def _integrate_single_trajectory(
    traj: Trajectory,
    states: Sequence[QuarkoniumState],
    rate_table: CharmRateTable,
) -> Dict[str, float]:
    """
    Integrate the primordial survival probabilities for all states
    along one trajectory, using trapezoidal rule and linear ramp
    before lab-frame formation time.

    Returns a dict with 5 keys: jpsi_1S, chic0_1P, chic1_1P, chic2_1P, psi_2S
    (χc states share the same survival).
    """

    T = traj.T_profile
    pt = float(traj.pt)
    y = float(traj.y)
    t0_fm = float(traj.t0_fm)
    dt_fm = float(traj.dt_fm)

    maxIdx = traj.active_length()
    if maxIdx == 0:
        # No QGP phase: survival=1 for all states
        out = {}
        for s_name, cols in CHARM_OUTPUT_MAP.items():
            for col in cols:
                out[col] = 1.0
        return out

    # Precompute the times in fm/c
    times_fm = t0_fm + dt_fm * np.arange(maxIdx, dtype=float)

    # Accumulated integrals ∫Γ dτ for the *3* physics states
    integrals = np.zeros(len(states), dtype=float)

    # Trapezoidal weights
    weights = np.ones(maxIdx, dtype=float)
    if maxIdx >= 2:
        weights[0] = 0.5
        weights[-1] = 0.5

    # Loop over time steps
    for i in range(maxIdx):
        Ti = T[i]
        if Ti <= traj.Tf:
            # below freeze-out temperature: no more QGP suppression
            continue

        t_fm = times_fm[i]
        w = weights[i]

        for k, st in enumerate(states):
            m = st.mass_GeV
            # lab gamma (we follow your newer code: ignore y in gamma)
            E = math.sqrt(m * m + pt * pt)
            gamma = E / m
            tau_form_lab_fm = st.tau_form_fm * gamma

            # linear ramp of the rate until the lab-frame formation time
            if tau_form_lab_fm > 0 and t_fm < tau_form_lab_fm:
                scale = t_fm / tau_form_lab_fm
            else:
                scale = 1.0

            Gamma = rate_table.gamma(st.rate_index, Ti, pt) * scale
            integrals[k] += w * dt_fm * Gamma

    # Convert integrals to survival probabilities
    S_phys = np.exp(-integrals)  # array of length 3

    # Map to the 5 TAMU-like columns
    out: Dict[str, float] = {}
    for st, S in zip(states, S_phys):
        for col in CHARM_OUTPUT_MAP[st.name]:
            out[col] = float(S)

    return out


@dataclass
class PrimordialResult:
    """Container for the full primordial survival table."""
    df: pd.DataFrame


def compute_primordial_survival(
    trajectories: Sequence[Trajectory],
    rate_table: CharmRateTable,
    states: Sequence[QuarkoniumState] = CHARM_STATES_DEFAULT,
) -> PrimordialResult:
    """
    High-level driver: loop over all trajectories and build
    a DataFrame with columns:
        b, pt, y, weight,
        jpsi_1S, chic0_1P, chic1_1P, chic2_1P, psi_2S
    ready to be passed to PrimordialAnalysis / BandEnsemble.
    """
    rows = []
    for traj in trajectories:
        surv = _integrate_single_trajectory(traj, states, rate_table)
        row = {
            "b": float(traj.b),
            "pt": float(traj.pt),
            "y": float(traj.y),
            "weight": float(traj.weight),
        }
        row.update(surv)
        rows.append(row)

    df = pd.DataFrame(rows)
    return PrimordialResult(df=df)
