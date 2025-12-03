# primordial_code/rates.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import numba as nb
    njit = nb.njit
except ImportError:  # fall back to no-op decorator
    def njit(*args, **kwargs):
        def wrap(fn): return fn
        return wrap

@dataclass
class RateTable:
    """Γ_k(T, p) for multiple quarkonium states on a regular T-p grid."""
    T_grid: np.ndarray             # shape (nT,)
    p_grid: np.ndarray             # shape (nP,)
    rates: np.ndarray              # shape (n_states, nT, nP)
    state_names: list[str]

    @classmethod
    def from_tsv(cls, path: str, state_names: list[str]) -> "RateTable":
        """
        Load rates from TSV with columns:
            T, pT, Γ_state0, Γ_state1, ...
        The file should cover a regular T × pT grid.
        """
        data = np.loadtxt(path, comments="#")
        T_raw = data[:, 0]
        p_raw = data[:, 1]
        Gamma = data[:, 2:]

        n_states = Gamma.shape[1]
        assert n_states == len(state_names), "state_names vs file columns mismatch"

        T_grid = np.unique(T_raw)
        p_grid = np.unique(p_raw)
        nT, nP = len(T_grid), len(p_grid)

        rates = np.zeros((n_states, nT, nP), dtype=float)

        # Fill the grid (robust, not assuming particular sort order)
        for iT, Tval in enumerate(T_grid):
            maskT = T_raw == Tval
            p_slice = p_raw[maskT]
            G_slice = Gamma[maskT, :]  # shape (nP, n_states)
            # sort by p
            order = np.argsort(p_slice)
            p_sorted = p_slice[order]
            G_sorted = G_slice[order, :]
            assert np.allclose(p_sorted, p_grid), "p-grid mismatch in rate file"
            for k in range(n_states):
                rates[k, iT, :] = G_sorted[:, k]

        return cls(T_grid=T_grid, p_grid=p_grid, rates=rates, state_names=state_names)


@njit
def _interp_rate(T_grid, p_grid, rate_grid, T_val, p_val):
    """
    Bilinear interpolation Γ(T,p) on regular grid.
    rate_grid: shape (nT, nP) for one state.
    """
    nT = T_grid.shape[0]
    nP = p_grid.shape[0]

    # Clip / locate indices for T
    if T_val <= T_grid[0]:
        iT = 0
    elif T_val >= T_grid[nT-1]:
        iT = nT - 2
    else:
        iT = np.searchsorted(T_grid, T_val) - 1

    # Clip / locate indices for p
    if p_val <= p_grid[0]:
        iP = 0
    elif p_val >= p_grid[nP-1]:
        iP = nP - 2
    else:
        iP = np.searchsorted(p_grid, p_val) - 1

    T1 = T_grid[iT];   T2 = T_grid[iT+1]
    P1 = p_grid[iP];   P2 = p_grid[iP+1]

    if T2 == T1:
        a = 0.0
    else:
        a = (T_val - T1) / (T2 - T1)

    if P2 == P1:
        b = 0.0
    else:
        b = (p_val - P1) / (P2 - P1)

    f11 = rate_grid[iT,   iP]
    f21 = rate_grid[iT+1, iP]
    f12 = rate_grid[iT,   iP+1]
    f22 = rate_grid[iT+1, iP+1]

    return ((1.0-a)*(1.0-b)*f11 +
            a*(1.0-b)*f21 +
            (1.0-a)*b*f12 +
            a*b*f22)
