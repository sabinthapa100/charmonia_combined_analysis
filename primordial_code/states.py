# primordial_code/states.py
from dataclasses import dataclass
import numpy as np

HBARC = 0.1973269804  # GeV*fm

@dataclass
class QuarkoniumState:
    name: str
    mass: float             # GeV
    formation_time_fm: float

    @property
    def formation_time_gevinv(self) -> float:
        """Formation time in 1/GeV (rest-frame)."""
        return self.formation_time_fm / HBARC

def default_charmonia_states():
    # You can tune masses and τ_form based on Du et al.
    return [
        QuarkoniumState("Jpsi_1S",  3.0969, 1.0),  # τ_form = 1 fm/c
        QuarkoniumState("chic_1P",  3.5107, 2.0),  # τ_form = 2 fm/c (all χc lumped)
        QuarkoniumState("psi2S_2S", 3.6861, 2.0),  # τ_form = 2 fm/c
    ]
