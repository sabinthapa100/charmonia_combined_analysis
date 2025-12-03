# cnm_combine.py
"""
Robust CNM combination module: nPDF × (eloss × Cronin pT broadening)
vs y, vs pT, vs centrality – closely mirroring the original notebook.

Key features
------------
* Handles both 5.02 and 8.16 TeV.
* Centrality bins, y-edges, pT-edges, y-windows can be given from outside.
* Uses the existing npdf_centrality + eloss_cronin_centrality modules.
* Returns clean dict-of-arrays results so you can:
    - save to CSV via helpers, and
    - reuse in later calculations (e.g. multiply with primordial bands).
* Fixes the Rb_hi bug that caused the UnboundLocalError.

Exports
-------
- CNMCombine
- DEFAULT_CENT_BINS, DEFAULT_Y_EDGES, DEFAULT_P_EDGES,
  DEFAULT_Y_WINDOWS, DEFAULT_PT_RANGE_AVG
- combine_two_bands_1d
- cnm_vs_y_to_dataframe, cnm_vs_pT_to_dataframe, cnm_vs_cent_to_dataframe
- demo_plots() (optional quick-look plotting)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
import sys

# ------------------------------------------------------------
# Paths / imports
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

NPDF_CODE_DIR = ROOT / "npdf_code"
ELOSS_CODE_DIR = ROOT / "eloss_code"

if str(NPDF_CODE_DIR) not in sys.path:
    sys.path.append(str(NPDF_CODE_DIR))
if str(ELOSS_CODE_DIR) not in sys.path:
    sys.path.append(str(ELOSS_CODE_DIR))

from npdf_data import NPDFSystem, RpAAnalysis  # type: ignore
from gluon_ratio import EPPS21Ratio, GluonEPPSProvider  # type: ignore
from glauber import OpticalGlauber, SystemSpec  # type: ignore
from npdf_centrality import (  # type: ignore
    compute_df49_by_centrality,
    make_centrality_weight_dict,
    bin_rpa_vs_y,
    bin_rpa_vs_pT,
    bin_rpa_vs_centrality,
)

from particle import Particle  # type: ignore
from coupling import alpha_s_provider  # type: ignore
import quenching_fast as QF  # type: ignore
from eloss_cronin_centrality import (  # type: ignore
    rpa_band_vs_y,
    rpa_band_vs_pT,
    rpa_band_vs_centrality,
)

# ------------------------------------------------------------
# nPDF input locations
# ------------------------------------------------------------
NPDF_INPUT_DIR = ROOT / "input" / "npdf"
P5_DIR = NPDF_INPUT_DIR / "pPb5TeV"
P8_DIR = NPDF_INPUT_DIR / "pPb8TeV"
EPPS_DIR = NPDF_INPUT_DIR / "nPDFs"

SQRTS = {"5.02": 5023.0, "8.16": 8160.0}
SIG_NN = {"5.02": 67.0, "8.16": 71.0}

# ------------------------------------------------------------
# Defaults (centrality, y, pT)
# ------------------------------------------------------------
DEFAULT_CENT_BINS: Sequence[Tuple[float, float]] = [
    (0, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 100),
]

# CMS-like y windows (you can override in notebook)
DEFAULT_Y_WINDOWS: Sequence[Tuple[float, float, str]] = [
    (-4.46, -2.96, "-4.46 < y < -2.96"),
    (-1.37, 0.43, "-1.37 < y < 0.43"),
    (2.03, 3.53, "2.03 < y < 3.53"),
]

# Binning grids
DEFAULT_PT_RANGE: Tuple[float, float] = (0.0, 20.0)
DEFAULT_PT_RANGE_AVG: Tuple[float, float] = (0.0, 15.0)
DEFAULT_PT_FLOOR_W: float = 1.0

DEFAULT_Y_EDGES: np.ndarray = np.arange(-5.0, 5.0 + 0.25, 0.5)
DEFAULT_P_EDGES: np.ndarray = np.arange(0.0, 15.0 + 2.5, 2.5)

# Weighting & centrality
DEFAULT_WEIGHT_MODE: str = "pp@local"
DEFAULT_Y_REF: float = 0.0
DEFAULT_CENT_EXP_C0: float = 0.25  # parameter in exp-weights for centrality

# eloss / broadening parameter bands
DEFAULT_Q0_PAIR: Tuple[float, float] = (0.05, 0.09)
DEFAULT_P0_SCALE_PAIR: Tuple[float, float] = (0.9, 1.1)

DEFAULT_NB_BSAMPLES: int = 5
DEFAULT_Y_SHIFT_FRACTION: float = 2.0


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def combine_two_bands_1d(
    R1_c,
    R1_lo,
    R1_hi,
    R2_c,
    R2_lo,
    R2_hi,
    eps: float = 1e-12,
):
    """
    Multiply two RpA bands: R_tot = R1 * R2 with standard
    quadrature error propagation on relative uncertainties.
    """
    R1_c = np.asarray(R1_c, float)
    R1_lo = np.asarray(R1_lo, float)
    R1_hi = np.asarray(R1_hi, float)

    R2_c = np.asarray(R2_c, float)
    R2_lo = np.asarray(R2_lo, float)
    R2_hi = np.asarray(R2_hi, float)

    Rc = np.full_like(R1_c, np.nan, dtype=float)
    Rlo = np.full_like(R1_c, np.nan, dtype=float)
    Rhi = np.full_like(R1_c, np.nan, dtype=float)

    mask = np.isfinite(R1_c) & np.isfinite(R2_c)
    if not np.any(mask):
        return Rc, Rlo, Rhi

    d1 = 0.5 * np.abs(R1_hi - R1_lo)
    d2 = 0.5 * np.abs(R2_hi - R2_lo)

    R_tot_c = R1_c * R2_c
    R1_safe = np.where(np.abs(R1_c) > eps, R1_c, 1.0)
    R2_safe = np.where(np.abs(R2_c) > eps, R2_c, 1.0)

    rel2 = (d1 / R1_safe) ** 2 + (d2 / R2_safe) ** 2
    d_tot = np.abs(R_tot_c) * np.sqrt(rel2)

    Rc[mask] = R_tot_c[mask]
    Rlo[mask] = R_tot_c[mask] - d_tot[mask]
    Rhi[mask] = R_tot_c[mask] + d_tot[mask]
    return Rc, Rlo, Rhi


def _tags_for_cent_bins(
    cent_bins: Sequence[Tuple[float, float]],
    include_mb: bool = True,
) -> Sequence[str]:
    tags = [f"{int(a)}-{int(b)}%" for (a, b) in cent_bins]
    if include_mb:
        tags.append("MB")
    return tags


# ------------------------------------------------------------
# Main container
# ------------------------------------------------------------
@dataclass
class CNMCombine:
    # configuration
    energy: str
    family: str
    particle_state: str

    sqrt_sNN: float
    sigma_nn_mb: float

    cent_bins: Sequence[Tuple[float, float]]
    y_edges: np.ndarray
    p_edges: np.ndarray
    y_windows: Sequence[Tuple[float, float, str]]
    pt_range_avg: Tuple[float, float]
    pt_floor_w: float

    weight_mode: str
    y_ref: float
    cent_c0: float

    q0_pair: Tuple[float, float]
    p0_scale_pair: Tuple[float, float]
    nb_bsamples: int
    y_shift_fraction: float

    # derived
    particle: Particle
    npdf_ctx: Dict[str, object]
    gl: OpticalGlauber
    qp_base: object

    # -------------------------
    # Constructor from defaults
    # -------------------------
    @classmethod
    def from_defaults(
        cls,
        energy: Literal["5.02", "8.16"] = "8.16",
        family: Literal["charmonia", "bottomonia"] = "charmonia",
        particle_state: str = "avg",
        cent_bins: Sequence[Tuple[float, float]] = None,
        y_edges: np.ndarray = None,
        p_edges: np.ndarray = None,
        y_windows: Sequence[Tuple[float, float, str]] = None,
        pt_range_avg: Tuple[float, float] = None,
        pt_floor_w: float = DEFAULT_PT_FLOOR_W,
        weight_mode: str = DEFAULT_WEIGHT_MODE,
        y_ref: float = DEFAULT_Y_REF,
        cent_c0: float = DEFAULT_CENT_EXP_C0,
        q0_pair: Tuple[float, float] = DEFAULT_Q0_PAIR,
        p0_scale_pair: Tuple[float, float] = DEFAULT_P0_SCALE_PAIR,
        nb_bsamples: int = DEFAULT_NB_BSAMPLES,
        y_shift_fraction: float = DEFAULT_Y_SHIFT_FRACTION,
    ) -> "CNMCombine":

        energy = str(energy)
        if energy not in SQRTS:
            raise ValueError("energy must be '5.02' or '8.16'")

        sqrt_sNN = SQRTS[energy]
        sigma_nn_mb = SIG_NN[energy]

        if cent_bins is None:
            cent_bins = DEFAULT_CENT_BINS
        if y_edges is None:
            y_edges = DEFAULT_Y_EDGES
        if p_edges is None:
            p_edges = DEFAULT_P_EDGES
        if y_windows is None:
            y_windows = DEFAULT_Y_WINDOWS
        if pt_range_avg is None:
            pt_range_avg = DEFAULT_PT_RANGE_AVG

        # quarkonium state (family + 'avg' or specific)
        particle = Particle(family=family, state=particle_state)

        # nPDF input dir
        input_dir = P5_DIR if energy == "5.02" else P8_DIR

        # ----------------------------------------------------
        # nPDF side: GluonEPPSProvider + RpAAnalysis
        # ----------------------------------------------------
        if family == "charmonia":
            m_state_for_np = "charmonium"
        elif family == "bottomonia":
            m_state_for_np = "bottomonium"
        else:
            # fallback: numeric mass if family name is non-standard
            m_state_for_np = particle.M_GeV

        epps_ratio = EPPS21Ratio(A=208, path=str(EPPS_DIR))
        gluon = GluonEPPSProvider(
            epps_ratio,
            sqrt_sNN_GeV=sqrt_sNN,
            m_state_GeV=m_state_for_np,
            y_sign_for_xA=-1,
        )

        gl_pA = OpticalGlauber(
            SystemSpec("pA", sqrt_sNN, A=208, sigma_nn_mb=sigma_nn_mb)
        )

        ana = RpAAnalysis()
        sys_npdf = NPDFSystem.from_folder(
            str(input_dir),
            kick="pp",
            name=f"p+Pb {energy} TeV",
        )

        base, r0, M = ana.compute_rpa_members(
            sys_npdf.df_pp,
            sys_npdf.df_pa,
            sys_npdf.df_errors,
            join="intersect",
            lowpt_policy="drop",
            pt_shift_min=pt_floor_w,
            shift_if_r_below=0.0,
        )

        df49_by_cent, K_by_cent, SA_all, Y_SHIFT = compute_df49_by_centrality(
            base,
            r0,
            M,
            gluon,
            gl_pA,
            cent_bins=cent_bins,
            nb_bsamples=nb_bsamples,
            y_shift_fraction=y_shift_fraction,
        )

        npdf_ctx = dict(
            df49_by_cent=df49_by_cent,
            df_pp=sys_npdf.df_pp,
            df_pa=sys_npdf.df_pa,
            gluon=gluon,
        )

        # ----------------------------------------------------
        # eloss + Cronin: QF.QuenchParams, alpha_s, etc.
        # ----------------------------------------------------
        alpha_s = alpha_s_provider(mode="running", LambdaQCD=0.25)
        gl_eloss = gl_pA
        Lmb = gl_eloss.leff_minbias_pA()

        device = "cpu"
        try:
            import torch  # type: ignore

            if QF._HAS_TORCH and torch.cuda.is_available():  # type: ignore
                device = "cuda"
        except Exception:
            pass

        qp_base = QF.QuenchParams(
            qhat0=0.075,
            lp_fm=1.5,
            LA_fm=Lmb,
            LB_fm=Lmb,
            lambdaQCD=0.25,
            roots_GeV=sqrt_sNN,
            alpha_of_mu=alpha_s,
            alpha_scale="mT",
            use_hard_cronin=True,
            mapping="exp",
            device=device,
        )

        return cls(
            energy=energy,
            family=family,
            particle_state=particle_state,
            sqrt_sNN=sqrt_sNN,
            sigma_nn_mb=sigma_nn_mb,
            cent_bins=cent_bins,
            y_edges=np.asarray(y_edges, float),
            p_edges=np.asarray(p_edges, float),
            y_windows=y_windows,
            pt_range_avg=pt_range_avg,
            pt_floor_w=pt_floor_w,
            weight_mode=weight_mode,
            y_ref=y_ref,
            cent_c0=cent_c0,
            q0_pair=q0_pair,
            p0_scale_pair=p0_scale_pair,
            nb_bsamples=nb_bsamples,
            y_shift_fraction=y_shift_fraction,
            particle=particle,
            npdf_ctx=npdf_ctx,
            gl=gl_eloss,
            qp_base=qp_base,
        )

    # --------------------------------------------------------
    # RpA vs rapidity
    # --------------------------------------------------------
    def cnm_vs_y(
        self,
        y_edges: Optional[np.ndarray] = None,
        pt_range_avg: Optional[Tuple[float, float]] = None,
        include_mb: bool = True,
        components: Sequence[str] = ("npdf", "eloss", "broad", "eloss_broad", "cnm"),
        mb_weight_mode: str = "exp",
        mb_c0: Optional[float] = None,
    ):
        if y_edges is None:
            y_edges = self.y_edges
        if pt_range_avg is None:
            pt_range_avg = self.pt_range_avg
        if mb_c0 is None:
            mb_c0 = self.cent_c0

        # MB weights for nPDF side
        wcent = (
            make_centrality_weight_dict(self.cent_bins, c0=mb_c0)
            if include_mb
            else None
        )

        # nPDF-only RpA vs y
        npdf_bins_y = bin_rpa_vs_y(
            self.npdf_ctx["df49_by_cent"],
            self.npdf_ctx["df_pp"],
            self.npdf_ctx["df_pa"],
            self.npdf_ctx["gluon"],
            cent_bins=self.cent_bins,
            y_edges=y_edges,
            pt_range_avg=pt_range_avg,
            weight_mode=self.weight_mode,
            y_ref=self.y_ref,
            pt_floor_w=self.pt_floor_w,
            wcent_dict=wcent,
            include_mb=include_mb,
        )

        y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])

        # eloss + broad + total vs y
        y_cent_eloss, bands_y, labels_y = rpa_band_vs_y(
            self.particle,
            self.sqrt_sNN,
            self.qp_base,
            self.gl,
            self.cent_bins,
            y_edges,
            pt_range_avg,
            components=("loss", "broad", "total"),
            q0_pair=self.q0_pair,
            p0_scale_pair=self.p0_scale_pair,
            Ny_bin=12,
            Npt_bin=24,
            weight_kind="pp",
            weight_ref_y="local",
            table_for_pp=None,
            mb_weight_mode=mb_weight_mode,
            mb_c0=mb_c0,
            mb_weights_custom=None,
        )

        if not np.allclose(y_cent, y_cent_eloss):
            raise RuntimeError("y grid mismatch between nPDF and eloss bands")

        RL_c, RL_lo, RL_hi = bands_y["loss"]
        RB_c, RB_lo, RB_hi = bands_y["broad"]
        RT_c, RT_lo, RT_hi = bands_y["total"]

        tags = _tags_for_cent_bins(self.cent_bins, include_mb=include_mb)

        cnm_y: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            comp: {} for comp in components
        }

        for tag in tags:
            # nPDF
            npdf_data = npdf_bins_y[tag]
            Rn_c = np.asarray(npdf_data["r_central"], float)
            Rn_lo = np.asarray(npdf_data["r_lo"], float)
            Rn_hi = np.asarray(npdf_data["r_hi"], float)

            # eloss
            Rloss_c = np.asarray(RL_c[tag], float)
            Rloss_lo = np.asarray(RL_lo[tag], float)
            Rloss_hi = np.asarray(RL_hi[tag], float)

            # broadening (BUG FIX: use RB_hi here)
            Rb_c = np.asarray(RB_c[tag], float)
            Rb_lo = np.asarray(RB_lo[tag], float)
            Rb_hi = np.asarray(RB_hi[tag], float)

            # total eloss×broad
            Rtot_c = np.asarray(RT_c[tag], float)
            Rtot_lo = np.asarray(RT_lo[tag], float)
            Rtot_hi = np.asarray(RT_hi[tag], float)

            # combined CNM = nPDF × total
            Rcnm_c, Rcnm_lo, Rcnm_hi = combine_two_bands_1d(
                Rn_c, Rn_lo, Rn_hi,
                Rtot_c, Rtot_lo, Rtot_hi,
            )

            if "npdf" in components:
                cnm_y["npdf"][tag] = (Rn_c, Rn_lo, Rn_hi)
            if "eloss" in components:
                cnm_y["eloss"][tag] = (Rloss_c, Rloss_lo, Rloss_hi)
            if "broad" in components:
                cnm_y["broad"][tag] = (Rb_c, Rb_lo, Rb_hi)
            if "eloss_broad" in components:
                cnm_y["eloss_broad"][tag] = (Rtot_c, Rtot_lo, Rtot_hi)
            if "cnm" in components:
                cnm_y["cnm"][tag] = (Rcnm_c, Rcnm_lo, Rcnm_hi)

        return y_cent, tags, cnm_y

    # --------------------------------------------------------
    # RpA vs pT
    # --------------------------------------------------------
    def cnm_vs_pT(
        self,
        y_window: Tuple[float, float] | Tuple[float, float, str],
        pt_edges: Optional[np.ndarray] = None,
        components: Sequence[str] = ("npdf", "eloss", "broad", "eloss_broad", "cnm"),
        include_mb: bool = True,
        mb_weight_mode: str = "exp",
        mb_c0: Optional[float] = None,
    ):
        if len(y_window) == 3:
            y0, y1, _ = y_window
        else:
            y0, y1 = y_window

        if pt_edges is None:
            pt_edges = self.p_edges
        if mb_c0 is None:
            mb_c0 = self.cent_c0

        wcent = (
            make_centrality_weight_dict(self.cent_bins, c0=mb_c0)
            if include_mb
            else None
        )

        # nPDF vs pT
        npdf_bins_pt = bin_rpa_vs_pT(
            self.npdf_ctx["df49_by_cent"],
            self.npdf_ctx["df_pp"],
            self.npdf_ctx["df_pa"],
            self.npdf_ctx["gluon"],
            cent_bins=self.cent_bins,
            pt_edges=pt_edges,
            y_window=(y0, y1),
            weight_mode=self.weight_mode,
            y_ref=self.y_ref,
            pt_floor_w=self.pt_floor_w,
            wcent_dict=wcent,
            include_mb=include_mb,
        )

        pT_cent = 0.5 * (pt_edges[:-1] + pt_edges[1:])

        # eloss + broad + total vs pT
        pT_cent_eloss, bands_pt, labels_pt = rpa_band_vs_pT(
            self.particle,
            self.sqrt_sNN,
            self.qp_base,
            self.gl,
            self.cent_bins,
            pt_edges,
            (y0, y1),
            components=("loss", "broad", "total"),
            q0_pair=self.q0_pair,
            p0_scale_pair=self.p0_scale_pair,
            Ny_bin=12,
            Npt_bin=24,
            weight_kind="pp",
            weight_ref_y="local",
            mb_weight_mode=mb_weight_mode,
            mb_c0=mb_c0,
            mb_weights_custom=None,
        )

        if not np.allclose(pT_cent, pT_cent_eloss):
            raise RuntimeError("pT grid mismatch between nPDF and eloss bands")

        RL_c, RL_lo, RL_hi = bands_pt["loss"]
        RB_c, RB_lo, RB_hi = bands_pt["broad"]
        RT_c, RT_lo, RT_hi = bands_pt["total"]

        tags = _tags_for_cent_bins(self.cent_bins, include_mb=include_mb)

        cnm_pt: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            comp: {} for comp in components
        }

        for tag in tags:
            d = npdf_bins_pt[tag]
            Rn_c = np.asarray(d["r_central"], float)
            Rn_lo = np.asarray(d["r_lo"], float)
            Rn_hi = np.asarray(d["r_hi"], float)

            Rloss_c = np.asarray(RL_c[tag], float)
            Rloss_lo = np.asarray(RL_lo[tag], float)
            Rloss_hi = np.asarray(RL_hi[tag], float)

            # BUG FIX here too: use RB_hi
            Rb_c = np.asarray(RB_c[tag], float)
            Rb_lo = np.asarray(RB_lo[tag], float)
            Rb_hi = np.asarray(RB_hi[tag], float)

            Rtot_c = np.asarray(RT_c[tag], float)
            Rtot_lo = np.asarray(RT_lo[tag], float)
            Rtot_hi = np.asarray(RT_hi[tag], float)

            Rcnm_c, Rcnm_lo, Rcnm_hi = combine_two_bands_1d(
                Rn_c, Rn_lo, Rn_hi,
                Rtot_c, Rtot_lo, Rtot_hi,
            )

            if "npdf" in components:
                cnm_pt["npdf"][tag] = (Rn_c, Rn_lo, Rn_hi)
            if "eloss" in components:
                cnm_pt["eloss"][tag] = (Rloss_c, Rloss_lo, Rloss_hi)
            if "broad" in components:
                cnm_pt["broad"][tag] = (Rb_c, Rb_lo, Rb_hi)
            if "eloss_broad" in components:
                cnm_pt["eloss_broad"][tag] = (Rtot_c, Rtot_lo, Rtot_hi)
            if "cnm" in components:
                cnm_pt["cnm"][tag] = (Rcnm_c, Rcnm_lo, Rcnm_hi)

        return pT_cent, tags, cnm_pt

    # --------------------------------------------------------
    # RpA vs centrality (with MB point)
    # --------------------------------------------------------
    def cnm_vs_centrality(
        self,
        y_window: Tuple[float, float] | Tuple[float, float, str],
        pt_range_avg: Optional[Tuple[float, float]] = None,
        components: Sequence[str] = ("npdf", "eloss", "broad", "eloss_broad", "cnm"),
        mb_weight_mode: str = "exp",
        mb_c0: Optional[float] = None,
    ):
        if len(y_window) == 3:
            y0, y1, _ = y_window
        else:
            y0, y1 = y_window

        if pt_range_avg is None:
            pt_range_avg = self.pt_range_avg
        if mb_c0 is None:
            mb_c0 = self.cent_c0

        # nPDF centrality-averaged result
        wcent = make_centrality_weight_dict(self.cent_bins, c0=mb_c0)
        width_weights = np.array(
            [wcent[f"{int(a)}-{int(b)}%"] for (a, b) in self.cent_bins],
            float,
        )

        npdf_cent = bin_rpa_vs_centrality(
            self.npdf_ctx["df49_by_cent"],
            self.npdf_ctx["df_pp"],
            self.npdf_ctx["df_pa"],
            self.npdf_ctx["gluon"],
            cent_bins=self.cent_bins,
            y_window=(y0, y1),
            pt_range_avg=pt_range_avg,
            weight_mode=self.weight_mode,
            y_ref=self.y_ref,
            pt_floor_w=self.pt_floor_w,
            width_weights=width_weights,
        )

        Rc_n = np.asarray(npdf_cent["r_central"], float)
        Rlo_n = np.asarray(npdf_cent["r_lo"], float)
        Rhi_n = np.asarray(npdf_cent["r_hi"], float)

        mb_n_c = float(npdf_cent["mb_r_central"])
        mb_n_lo = float(npdf_cent["mb_r_lo"])
        mb_n_hi = float(npdf_cent["mb_r_hi"])

        labels_cent = [f"{int(a)}-{int(b)}%" for (a, b) in self.cent_bins]

        # eloss + broadening centrality bands
        (
            labels_el,
            RL_c,
            RL_lo,
            RL_hi,
            RB_c,
            RB_lo,
            RB_hi,
            RT_c,
            RT_lo,
            RT_hi,
            RMB_loss,
            RMB_broad,
            RMB_tot,
        ) = rpa_band_vs_centrality(
            self.particle,
            self.sqrt_sNN,
            self.qp_base,
            self.gl,
            self.cent_bins,
            (y0, y1),
            pt_range_avg,
            q0_pair=self.q0_pair,
            p0_scale_pair=self.p0_scale_pair,
            Ny_bin=16,
            Npt_bin=32,
            weight_kind="pp",
            weight_ref_y="local",
            mb_weight_mode=mb_weight_mode,
            mb_c0=mb_c0,
            mb_weights_custom=None,
        )

        if labels_el != labels_cent:
            raise RuntimeError("Centrality label mismatch between nPDF and eloss bands")

        Rloss_c = np.array([RL_c[lab] for lab in labels_cent], float)
        Rloss_lo = np.array([RL_lo[lab] for lab in labels_cent], float)
        Rloss_hi = np.array([RL_hi[lab] for lab in labels_cent], float)

        Rb_c = np.array([RB_c[lab] for lab in labels_cent], float)
        Rb_lo = np.array([RB_lo[lab] for lab in labels_cent], float)
        Rb_hi = np.array([RB_hi[lab] for lab in labels_cent], float)

        Rtot_c = np.array([RT_c[lab] for lab in labels_cent], float)
        Rtot_lo = np.array([RT_lo[lab] for lab in labels_cent], float)
        Rtot_hi = np.array([RT_hi[lab] for lab in labels_cent], float)

        Rc_tot_MB, Rlo_tot_MB, Rhi_tot_MB = RMB_tot

        # combined CNM vs centrality
        Rcnm_c, Rcnm_lo, Rcnm_hi = combine_two_bands_1d(
            Rc_n, Rlo_n, Rhi_n,
            Rtot_c, Rtot_lo, Rtot_hi,
        )

        # MB CNM point
        Rcnm_MB_c, Rcnm_MB_lo, Rcnm_MB_hi = combine_two_bands_1d(
            np.array([mb_n_c]),
            np.array([mb_n_lo]),
            np.array([mb_n_hi]),
            np.array([Rc_tot_MB]),
            np.array([Rlo_tot_MB]),
            np.array([Rhi_tot_MB]),
        )
        Rcnm_MB_c = float(Rcnm_MB_c[0])
        Rcnm_MB_lo = float(Rcnm_MB_lo[0])
        Rcnm_MB_hi = float(Rcnm_MB_hi[0])

        # component → (centrality array, MB triple)
        cnm_cent: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]] = {}

        if "npdf" in components:
            cnm_cent["npdf"] = (Rc_n, Rlo_n, Rhi_n, mb_n_c, mb_n_lo, mb_n_hi)
        if "eloss" in components:
            Rc_MB_loss, Rlo_MB_loss, Rhi_MB_loss = RMB_loss
            cnm_cent["eloss"] = (Rloss_c, Rloss_lo, Rloss_hi,
                                 Rc_MB_loss, Rlo_MB_loss, Rhi_MB_loss)
        if "broad" in components:
            Rc_MB_broad, Rlo_MB_broad, Rhi_MB_broad = RMB_broad
            cnm_cent["broad"] = (Rb_c, Rb_lo, Rb_hi,
                                 Rc_MB_broad, Rlo_MB_broad, Rhi_MB_broad)
        if "eloss_broad" in components:
            cnm_cent["eloss_broad"] = (Rtot_c, Rtot_lo, Rtot_hi,
                                       Rc_tot_MB, Rlo_tot_MB, Rhi_tot_MB)
        if "cnm" in components:
            cnm_cent["cnm"] = (Rcnm_c, Rcnm_lo, Rcnm_hi,
                               Rcnm_MB_c, Rcnm_MB_lo, Rcnm_MB_hi)

        return cnm_cent


# ------------------------------------------------------------
# DataFrame converters (for CSV / plotting)
# ------------------------------------------------------------
def cnm_vs_y_to_dataframe(
    y_cent: np.ndarray,
    tags: Sequence[str],
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    component: str,
) -> pd.DataFrame:
    """
    Long-form DataFrame for a single component vs y across centralities.
    """
    rows = []
    comp_dict = result[component]

    for tag in tags:
        Rc, Rlo, Rhi = comp_dict[tag]
        for yi, Rc_i, Rlo_i, Rhi_i in zip(y_cent, Rc, Rlo, Rhi):
            rows.append(
                dict(
                    y_center=float(yi),
                    centrality=tag,
                    is_MB=(tag == "MB"),
                    R_central=float(Rc_i),
                    R_lo=float(Rlo_i),
                    R_hi=float(Rhi_i),
                )
            )

    return pd.DataFrame(rows)


def cnm_vs_pT_to_dataframe(
    pT_cent: np.ndarray,
    tags: Sequence[str],
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    component: str,
) -> pd.DataFrame:
    """
    Long-form DataFrame for a single component vs pT across centralities.
    """
    rows = []
    comp_dict = result[component]

    for tag in tags:
        Rc, Rlo, Rhi = comp_dict[tag]
        for pi, Rc_i, Rlo_i, Rhi_i in zip(pT_cent, Rc, Rlo, Rhi):
            rows.append(
                dict(
                    pT_center=float(pi),
                    centrality=tag,
                    is_MB=(tag == "MB"),
                    R_central=float(Rc_i),
                    R_lo=float(Rlo_i),
                    R_hi=float(Rhi_i),
                )
            )

    return pd.DataFrame(rows)


def cnm_vs_cent_to_dataframe(
    cent_bins: Sequence[Tuple[float, float]],
    result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]],
    component: str,
) -> pd.DataFrame:
    """
    Long-form DataFrame for a component vs centrality, including MB point.
    """
    Rc, Rlo, Rhi, Rc_MB, Rlo_MB, Rhi_MB = result[component]

    rows = []
    for (cL, cR), Rc_i, Rlo_i, Rhi_i in zip(cent_bins, Rc, Rlo, Rhi):
        lab = f"{int(cL)}-{int(cR)}%"
        rows.append(
            dict(
                cent_left=float(cL),
                cent_right=float(cR),
                cent_label=lab,
                is_MB=False,
                R_central=float(Rc_i),
                R_lo=float(Rlo_i),
                R_hi=float(Rhi_i),
            )
        )

    # MB entry
    rows.append(
        dict(
            cent_left=float(cent_bins[0][0]),
            cent_right=float(cent_bins[-1][1]),
            cent_label="MB",
            is_MB=True,
            R_central=float(Rc_MB),
            R_lo=float(Rlo_MB),
            R_hi=float(Rhi_MB),
        )
    )

    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Light demo plotting (optional)
# ----------------------------------------------------------------------


def _step_from_centers(x_cent: np.ndarray, vals: np.ndarray):
    x_cent = np.asarray(x_cent, float)
    vals = np.asarray(vals, float)
    assert x_cent.size == vals.size

    if x_cent.size > 1:
        dx = np.diff(x_cent)
        dx0 = dx[0]
        if not np.allclose(dx, dx0):
            raise ValueError("x_cent not uniformly spaced")
    else:
        dx0 = 1.0

    x_edges = np.concatenate(([x_cent[0] - 0.5 * dx0], x_cent + 0.5 * dx0))
    y_step = np.concatenate([vals, vals[-1:]])
    return x_edges, y_step


def demo_plots(
    energy: str = "8.16",
    family: str = "charmonia",
    particle_state: str = "avg",
    outdir: Optional[Path] = None,
):
    """
    Quick-look plots (vs y, vs pT, vs centrality) using this module.

    This is deliberately simpler than the publication-style notebook
    but is good for sanity checks.
    """
    import matplotlib.pyplot as plt

    if outdir is None:
        outdir = HERE / "output-cnm"
    outdir.mkdir(exist_ok=True, parents=True)

    comb = CNMCombine.from_defaults(
        energy=energy,
        family=family,
        particle_state=particle_state,
    )

    components = ("npdf", "eloss", "broad", "eloss_broad", "cnm")
    comp_colors = {
        "npdf": "tab:blue",
        "eloss": "tab:orange",
        "broad": "tab:green",
        "eloss_broad": "tab:purple",
        "cnm": "k",
    }

    # ---- RpA vs y ----
    y_cent, tags_y, cnm_y = comb.cnm_vs_y()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, tag in enumerate(tags_y):
        ax = axes[i]
        for comp in components:
            Rc, Rlo, Rhi = cnm_y[comp][tag]
            x_edges, y_c = _step_from_centers(y_cent, Rc)
            _, y_lo = _step_from_centers(y_cent, Rlo)
            _, y_hi = _step_from_centers(y_cent, Rhi)

            ax.step(
                x_edges, y_c, where="post",
                lw=1.6, color=comp_colors[comp],
                label=comp if i == 0 else None,
            )
            ax.fill_between(
                x_edges, y_lo, y_hi,
                step="post", alpha=0.2, color=comp_colors[comp],
            )

        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_title(tag)
        ax.set_xlabel("y")
        ax.set_ylabel(r"$R_{pA}$")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(f"CNM vs y, sqrt(sNN)={comb.sqrt_sNN/1000:.2f} TeV")
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(outdir / f"demo_RpA_CNM_vs_y_{energy.replace('.','p')}TeV.png", dpi=150)

    # ---- RpA vs pT (first y-window) ----
    y0, y1, label = comb.y_windows[0]
    pT_cent, tags_pt, cnm_pt = comb.cnm_vs_pT((y0, y1, label))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, tag in enumerate(tags_pt):
        ax = axes[i]
        for comp in components:
            Rc, Rlo, Rhi = cnm_pt[comp][tag]
            x_edges, y_c = _step_from_centers(pT_cent, Rc)
            _, y_lo = _step_from_centers(pT_cent, Rlo)
            _, y_hi = _step_from_centers(pT_cent, Rhi)

            ax.step(
                x_edges, y_c, where="post",
                lw=1.6, color=comp_colors[comp],
                label=comp if i == 0 else None,
            )
            ax.fill_between(
                x_edges, y_lo, y_hi,
                step="post", alpha=0.2, color=comp_colors[comp],
            )

        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_title(tag)
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel(r"$R_{pA}$")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(
        f"CNM vs pT, {label}, sqrt(sNN)={comb.sqrt_sNN/1000:.2f} TeV"
    )
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(
        outdir
        / f"demo_RpA_CNM_vs_pT_{label.replace(' ','_')}_{energy.replace('.','p')}TeV.png",
        dpi=150,
    )

    # ---- RpA vs centrality (all y-windows) ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (y0, y1, name) in zip(axes, comb.y_windows):
        cnm_cent = comb.cnm_vs_centrality((y0, y1, name))
        edges = [comb.cent_bins[0][0]] + [b for (_, b) in comb.cent_bins]
        x_edges = np.array(edges, float)

        for comp in components:
            Rc, Rlo, Rhi, Rc_MB, Rlo_MB, Rhi_MB = cnm_cent[comp]
            y_step = np.concatenate([Rc, Rc[-1:]])
            y_lo = np.concatenate([Rlo, Rlo[-1:]])
            y_hi = np.concatenate([Rhi, Rhi[-1:]])

            ax.step(
                x_edges, y_step, where="post",
                lw=1.8, color=comp_colors[comp],
                label=comp if ax is axes[0] else None,
            )
            ax.fill_between(
                x_edges, y_lo, y_hi,
                step="post", alpha=0.2, color=comp_colors[comp],
            )

            if comp == "cnm":
                x_mb = np.array([comb.cent_bins[0][0], comb.cent_bins[-1][1]], float)
                ax.hlines(Rc_MB, x_mb[0], x_mb[1],
                          colors="k", linestyles="--", lw=1.8)

        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("Centrality [%]")
        ax.set_ylabel(r"$R_{pA}$")
        ax.set_title(name)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(f"CNM vs centrality, sqrt(sNN)={comb.sqrt_sNN/1000:.2f} TeV")
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(
        outdir
        / f"demo_RpA_CNM_vs_centrality_{energy.replace('.','p')}TeV.png",
        dpi=150,
    )


if __name__ == "__main__":
    # Run quick sanity-check plots for 8.16 TeV charmonia
    demo_plots()