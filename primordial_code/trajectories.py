# primordial_code/trajectories.py
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class Trajectory:
    """
    Hydro trajectory sampled on a uniform time grid.

    t0, dt are in 1/GeV.
    T:    local temperature [GeV] at each step
    ux,uy: optional flow components (for rest-frame p; can be zeros initially)
    """
    T:  np.ndarray
    t0: float
    dt: float
    b:  float
    pt: float
    y:  float
    phi: float
    weight: float = 1.0
    Tf: float = 0.18          # freeze-out temp in GeV
    ux: Optional[np.ndarray] = None
    uy: Optional[np.ndarray] = None

    def effective_length(self) -> int:
        """
        Return maxIdx equivalent: last index with T > Tf.
        """
        T = self.T
        idx = len(T) - 1
        while idx >= 0 and T[idx] <= self.Tf:
            idx -= 1
        return max(idx+1, 0)

import tarfile
from dataclasses import dataclass

@dataclass
class TrajMeta:
    idx: int
    b: float
    pt: float
    y: float
    phi: float
    weight: float
    member_name: str    # name in the tar file

def build_traj_index(tgz_path: str) -> list[TrajMeta]:
    metas = []
    with tarfile.open(tgz_path, "r:gz") as tar:
        for idx, member in enumerate(tar):
            if not member.isfile():
                continue
            # e.g. "b_1.53/pt_2.0_y_0.5_phi_0.3_traj1234.txt"
            name = member.name

            # *** YOU will plug in the real parsing logic here ***:
            # Either:
            #  - parse b,pt,y,phi from the filename; or
            #  - open and read a header line with these numbers.
            # For now I'll assume a custom helper:
            b, pt, y, phi, weight = parse_metadata_from_name_or_header(tar, member)

            metas.append(TrajMeta(
                idx=idx,
                b=b, pt=pt, y=y,
                phi=phi, weight=weight,
                member_name=name,
            ))
    return metas