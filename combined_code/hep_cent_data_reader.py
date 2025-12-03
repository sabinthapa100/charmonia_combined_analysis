# -*- coding: utf-8 -*-
"""
ALICE p–Pb experimental data reader (5.02 & 8.16 TeV)
Reads UNZIPPED HEPData CSVs arranged as:

input/experimental_input/
  5TeVpPb/centrality/**/Table{4,5,10,11}.csv
  8TeVpPb/centrality/**/Table{5,6,7,12,13}.csv

Tidy columns:
['energy_TeV','table','rapidity','centrality','location','quantity',
 'xvar','x_low','x_high','x_cen','ncoll','value',
 'stat_up','stat_dn','sys_uncorr_up','sys_uncorr_dn',
 'sys_corr_up','sys_corr_dn','sys_common_up','sys_common_dn']
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict
import csv, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- small utils ----------
_NUM = re.compile(r'^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$')

def _to_float(x):
    if x is None: return None
    s = str(x).strip().replace("−","-")
    s = re.sub(r'\s*\([^)]+\)\s*$', '', s)  # drop trailing units "(GeV/c)"
    try: return float(s)
    except: return None

def _split_range(s: str) -> Tuple[Optional[float], Optional[float]]:
    if s is None: return (None, None)
    s = str(s).replace("–","-").replace("—","-").replace("−","-").replace("to","-")
    if "-" not in s: return (None, None)
    a,b = [t.strip() for t in s.split("-", 1)]
    return (_to_float(a), _to_float(b))

def _mid(a, b): return None if (a is None or b is None) else 0.5*(a+b)

# --- robust y_cms parser, handles "−" and "-" and double minus like "-4.46 - -2.96"
_YSPAN = re.compile(r'([\-−]?\d+(?:\.\d+)?)\s*[\-–]\s*([\-−]?\d+(?:\.\d+)?)')

def _ycms_center(rap: str | None) -> float | None:
    """Return center of a y range like '2.03 - 3.53' or '-4.46 - -2.96'."""
    if not rap:
        return None
    s = rap.replace("\u2212", "-")  # U+2212 → '-'
    # find signed floats anywhere in the string
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', s)
    if len(nums) < 2:
        return None
    a, b = float(nums[0]), float(nums[1])
    return 0.5 * (a + b)

@dataclass
class _Meta:
    table: str
    energy_TeV: Optional[float] = None
    rapidity: Optional[str] = None
    centrality: Optional[str] = None
    location: Optional[str] = None

# ---------- reader ----------
class HEPCentDataReader:
    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)

    # paths & which tables
    def _base(self, energy: str) -> Path:
        e = energy.lower()
        if e.startswith("8"): return self.root/"8TeVpPb"/"centrality"
        if e.startswith("5"): return self.root/"5TeVpPb"/"centrality"
        raise ValueError("energy must be '5TeV' or '8TeV' (or variants)")

    def _wanted(self, energy: str) -> set[str]:
        if energy.lower().startswith("8"):
            # 8.16 TeV: Ncoll, pT, double ratio
            return {"Table5.csv","Table6.csv","Table7.csv","Table12.csv","Table13.csv"}
        # 5.02 TeV: Ncoll (Tables 5 & 6) + pT (Tables 10 & 11)
        return {"Table4.csv","Table5.csv","Table6.csv","Table7.csv","Table10.csv","Table11.csv"}



    def _iter_csv(self, energy: str) -> Iterable[Path]:
        want = self._wanted(energy)
        for p in self._base(energy).rglob("Table*.csv"):
            if p.name in want:
                yield p

    # parse meta lines from a window preceding a block
    def _meta_from_context(self, ctx: List[str], table_name: str) -> _Meta:
        m = _Meta(table=table_name)
        for ln in ctx:
            if not ln.startswith("#:"): continue
            body = ln[2:].strip(); low = body.lower()
            if low.startswith("sqrt(s)/nucleon"):
                nums = re.findall(r'([0-9.]+)', body)
                if nums:
                    egev=float(nums[-1]); m.energy_TeV = egev/1000.0 if egev>1000 else egev
            elif low.startswith("yrap"):
                parts = [x.strip() for x in body.split(",")]
                m.rapidity = parts[-1] if parts else None
            elif low.startswith("centrality percentiles"):
                parts = [x.strip() for x in body.split(",")]
                m.centrality = parts[-1] if parts else None
            elif low.startswith("location:"):
                m.location = body.split(":",1)[1].strip()
        return m

    # normalize headers -> short canonical names
    def _canon(self, cols: List[str]) -> List[str]:
        out=[]
        for c in cols:
            x = (c or "").strip().replace('"','').replace("'","")
            x = x.replace("YRAP(RF=CM)","y")
            x = x.replace("pT range, GeV/c","pT").replace("pT range","pT").replace("PT","pT")
            x = x.replace("Y range","y")
            x = x.replace("sys, uncorrel","sys_uncorr").replace("sys, correl","sys_corr").replace("sys,corr","sys_corr")
            if "LOW" in x.upper() and "PT" in x.upper(): x="pT_LOW"
            if "HIGH" in x.upper() and "PT" in x.upper(): x="pT_HIGH"
            if x.lower().startswith("y") and "low" in x.lower(): x="y_LOW"
            if x.lower().startswith("y") and "high" in x.lower(): x="y_HIGH"
            if "NCOLL^" in x.upper(): x="NCOLL"         # 5 TeV NCOLL^MULT/V0A/ZNA -> NCOLL
            out.append(x)
        return out

    # is this a data header?
    def _is_header(self, s: str) -> bool:
        S = s.upper()
        return (("Q_PPB" in S) or ("R_PA" in S) or ("Q_PA" in S) or
                ("(SIGMA_PSI" in S and "/SIGMA_JPSI" in S))

    def _read_blocks(self, path: Path) -> List[Tuple[_Meta, pd.DataFrame]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()!=""]
        blocks: List[Tuple[_Meta, pd.DataFrame]] = []
        i = 0
        # rolling meta context (the last ~10 lines of '#:' are relevant)
        while i < len(lines):
            if self._is_header(lines[i]):
                # collect meta context above header
                ctx_start = max(0, i-12)
                meta = self._meta_from_context(lines[ctx_start:i], path.name)

                # gather rows until next '#:' or next header
                j = i+1
                while j < len(lines) and (not lines[j].startswith("#:")) and (not self._is_header(lines[j])):
                    j += 1
                raw = [lines[i]] + [ln for ln in lines[i+1:j]
                                    if not ln.startswith("#:")
                                    and not ln.startswith("The first uncertainty")]
                rows = list(csv.reader(raw))
                if not rows: i=j; continue
                hdr = self._canon(rows[0])
                body = [r for r in rows[1:] if len(r)==len(rows[0])]
                if not body: i=j; continue
                df = pd.DataFrame(body, columns=hdr)
                blocks.append((meta, df))
                i = j
            else:
                i += 1
        return blocks

    def _tidy(self, meta: _Meta, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # quantity column
        quantity=None; val=None
        for k in ["Q_pPb","R_pPb","Q_pA","R_pA"]:
            for c in df.columns:
                if (c==k) or (k in c):
                    quantity=k; val=c; break
            if val: break
        if val is None:
            for c in df.columns:
                cl=c.lower()
                if ("sigma_psi" in cl) and ("/sigma_jpsi" in cl):
                    quantity="double_ratio"; val=c; break
        if val is None: return None

        # x variable
        xvar=xmain=xlow=xhigh=None
        if "NCOLL" in df.columns: xvar,xmain="Ncoll","NCOLL"
        elif "pT" in df.columns:
            xvar,xmain="pT","pT"
            xlow = "pT_LOW" if "pT_LOW" in df.columns else None
            xhigh= "pT_HIGH" if "pT_HIGH" in df.columns else None
        elif "y" in df.columns:
            xvar,xmain="y","y"
            xlow="y_LOW" if "y_LOW" in df.columns else None
            xhigh="y_HIGH" if "y_HIGH" in df.columns else None

        # split string ranges if needed
        if xvar in {"pT","y"} and (xlow is None or xhigh is None):
            lows, highs = [], []
            for s in df[xmain].astype(str).tolist():
                lo,hi = _split_range(s); lows.append(lo); highs.append(hi)
            if any(v is not None for v in lows) and any(v is not None for v in highs):
                df[xvar+"_LOW"] = lows; df[xvar+"_HIGH"] = highs
                xlow, xhigh = xvar+"_LOW", xvar+"_HIGH"

        # uncertainties
        def pick(stem: str) -> Optional[str]:
            for c in df.columns:
                if c.lower().startswith(stem): return c
            return None
        stat_up = pick("stat +");          stat_dn = pick("stat -")
        su_up   = pick("sys_uncorr +") or pick("sys +")
        su_dn   = pick("sys_uncorr -") or pick("sys -")
        sc_up   = pick("sys_corr +");      sc_dn   = pick("sys_corr -")
        com_up  = pick("sys, common correl +"); com_dn = pick("sys, common correl -")

        # rows
        recs: List[Dict] = []
        for _, r in df.iterrows():
            xval = _to_float(r.get(xmain)) if xmain else None
            xl   = _to_float(r.get(xlow))  if xlow  else None
            xh   = _to_float(r.get(xhigh)) if xhigh else None
            recs.append(dict(
                energy_TeV = meta.energy_TeV,
                table      = meta.table,
                rapidity   = meta.rapidity,
                centrality = meta.centrality,
                location   = meta.location,
                quantity   = quantity,
                xvar       = xvar,
                x_low      = xl, x_high = xh,
                x_cen      = _mid(xl, xh) if xl is not None and xh is not None else xval,
                ncoll      = _to_float(r.get("NCOLL")),
                value      = _to_float(r.get(val)),
                stat_up    = _to_float(r.get(stat_up)) if stat_up else None,
                stat_dn    = _to_float(r.get(stat_dn)) if stat_dn else None,
                sys_uncorr_up = _to_float(r.get(su_up)) if su_up else None,
                sys_uncorr_dn = _to_float(r.get(su_dn)) if su_dn else None,
                sys_corr_up   = _to_float(r.get(sc_up)) if sc_up else None,
                sys_corr_dn   = _to_float(r.get(sc_dn)) if sc_dn else None,
                sys_common_up = _to_float(r.get(com_up)) if com_up else None,
                sys_common_dn = _to_float(r.get(com_dn)) if com_dn else None,
            ))
        out = pd.DataFrame.from_records(recs)
        # drop rows with no value
        return out[out["value"].notna()]

    # ---------- public API ----------
    def load_all(self, energy: str) -> pd.DataFrame:
        frames=[]
        for p in self._iter_csv(energy):
            for meta, raw in self._read_blocks(p):
                tidy = self._tidy(meta, raw)
                if tidy is not None and not tidy.empty:
                    frames.append(tidy)
        if not frames:
            raise RuntimeError(f"No usable blocks under {self._base(energy)}")
        return pd.concat(frames, ignore_index=True)

    # selectors
    def rpa_vs_ncoll(self, energy: str) -> pd.DataFrame:
        df = self.load_all(energy)
        keep = {"Q_pPb","R_pPb","Q_pA","R_pA"}    # exclude double_ratio
        df = df[(df["xvar"] == "Ncoll") & (df["quantity"].isin(keep)) & df["value"].notna()]
        return df.sort_values(["rapidity","ncoll"]).reset_index(drop=True)


    def rpa_vs_pt(self, energy: str, rapidity: Optional[str]=None,
                  centrality: Optional[str]=None) -> List[pd.DataFrame]:
        df = self.load_all(energy)
        df = df[(df["xvar"]=="pT") & df["value"].notna()]
        if rapidity:  df = df[df["rapidity"]==rapidity]
        if centrality: df = df[df["centrality"]==centrality]
        return [g.sort_values("x_cen").reset_index(drop=True)
                for _, g in df.groupby(["rapidity","centrality"])]

    def double_ratio_vs_ncoll(self, energy: str) -> pd.DataFrame:
        df = self.load_all(energy)
        return df[(df["xvar"]=="Ncoll") & (df["quantity"]=="double_ratio")]\
                 .sort_values(["rapidity","ncoll"]).reset_index(drop=True)

    # pT-averaged RpA vs y (two points per centrality: backward/forward)
    def rpa_vs_y_from_pt(self, energy: str, how: str = "width",
                        pt_min: float | None = None,
                        pt_max: float | None = None) -> pd.DataFrame:
        """
        pT-averaged RpA (or QpPb) vs y, per centrality.
        - Uses only pT-binned tables (backward/forward rapidities).
        - 'how="width"' -> average weighted by bin width (HIGH-LOW).
        - Returns two points per centrality (backward & forward) when present.
        """
        df = self.load_all(energy)
        df = df[(df["xvar"] == "pT") & df["value"].notna()].copy()
        if df.empty:
            return pd.DataFrame(columns=["energy_TeV","centrality","rapidity","y_cen","value"])

        # pT selection window
        if pt_min is not None:
            df = df[df["x_high"] > pt_min]
        if pt_max is not None:
            df = df[df["x_low"]  < pt_max]
        if df.empty:
            return pd.DataFrame(columns=["energy_TeV","centrality","rapidity","y_cen","value"])

        # weights
        df["width"] = (df["x_high"].astype(float) - df["x_low"].astype(float)).abs()

        def _agg(g):
            if how == "width":
                w = g["width"].to_numpy(float)
                v = g["value"].to_numpy(float)
                val = np.average(v, weights=w)
            else:
                val = float(np.mean(g["value"].to_numpy(float)))
            return pd.Series({
                "value": val,
                "y_cen": _ycms_center(g["rapidity"].iloc[0])
            })

        out = (df.groupby(["centrality","rapidity","energy_TeV"], as_index=False)
                .apply(_agg)
                .reset_index(drop=True))

        cols = ["energy_TeV","centrality","rapidity","y_cen","value"]
        return out[cols].sort_values(["centrality","rapidity"]).reset_index(drop=True)


# ---------- plotting helpers ----------
def _build_yerr(df: pd.DataFrame, mode="stat_plus_uncorr"):
    z = lambda col: np.nan_to_num(df.get(col, 0.0).astype(float).to_numpy(), nan=0.0)
    sup, sdn = z("stat_up"), np.abs(z("stat_dn"))
    uup, udn = z("sys_uncorr_up"), np.abs(z("sys_uncorr_dn"))
    cup, cdn = z("sys_corr_up"),   np.abs(z("sys_corr_dn"))
    cump, cumd = z("sys_common_up"), np.abs(z("sys_common_dn"))
    if mode=="stat_only":
        yup, ydn = sup, sdn
    elif mode=="all_uncorr_and_corr":
        yup = np.sqrt(sup**2 + uup**2 + cup**2 + cump**2)
        ydn = np.sqrt(sdn**2 + udn**2 + cdn**2 + cumd**2)
    else:  # default: stat ⊕ uncorrelated (match ALICE caption when "correlated added in quadrature" not desired)
        yup = np.sqrt(sup**2 + uup**2); ydn = np.sqrt(sdn**2 + udn**2)
    return np.vstack([ydn, yup])

def plot_series(ax, df: pd.DataFrame, *, x="x_cen", label=None,
                yerr_mode="stat_plus_uncorr", fmt="o", cap=3.0, show_xerr=True):
    df = df.sort_values(x)
    X = df[x].astype(float).to_numpy()
    Y = df["value"].astype(float).to_numpy()
    yerr = _build_yerr(df, mode=yerr_mode)
    xerr = None
    if show_xerr and df["x_low"].notna().any() and df["x_high"].notna().any():
        xl = df["x_low"].astype(float).to_numpy()
        xh = df["x_high"].astype(float).to_numpy()
        xerr = np.vstack([X-xl, xh-X])
    ax.errorbar(X, Y, yerr=yerr, xerr=xerr, fmt=fmt, capsize=cap, label=label)

def plot_pt_panels(groups: List[pd.DataFrame], energy_label: str, y_label: str):
    """Make 6 subpanels (one per centrality) for a given rapidity region."""
    # keep order typical in papers
    order = ["2-10%","10-20%","20-40%","40-60%","60-80%","80-90%","80-100%"]
    groups = sorted(groups, key=lambda g: order.index(g["centrality"].iloc[0]) if g["centrality"].iloc[0] in order else 999)
    n = len(groups)
    cols = 3
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.6*rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, g in zip(axes, groups):
        cent = g["centrality"].iloc[0]
        plot_series(ax, g, x="x_cen", label=None, fmt="s", yerr_mode="stat_plus_uncorr")
        ax.set_title(f"{y_label}, {cent}")
        ax.set_xlabel(r"$p_T$ [GeV/$c$]")
        ax.set_ylabel(energy_label)
        ax.grid(alpha=0.2)
    for ax in axes[n:]: ax.axis("off")
    fig.tight_layout()
    return fig, axes

def plot_y_panels(rpa5: pd.DataFrame | None,
                  rpa8: pd.DataFrame | None,
                  *,
                  xlim = (-5, 5),
                  title = r"$\langle R_{pA}\rangle_{p_T}$ vs $y_{\rm cms}$ (by centrality)",
                  merge_80100_into_8090: bool = True):
    # Optional: remap 80–100% → 80–90% to overlay on a single panel
    def _remap(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty: return df
        if merge_80100_into_8090:
            df = df.copy()
            df["centrality"] = df["centrality"].replace({"80-100%": "80-90%"})
        return df

    r5 = _remap(rpa5)
    r8 = _remap(rpa8)

    # find union of centralities, ordered nicely
    order = ["2-10%","10-20%","20-40%","40-60%","60-80%","80-90%"]
    cents = sorted(set(([] if r5 is None else r5["centrality"].dropna().unique())) |
                   set(([] if r8 is None else r8["centrality"].dropna().unique())),
                   key=lambda c: order.index(c) if c in order else 999)

    n = len(cents)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, np.array([ax])

    cols = 3
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2*rows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, cent in zip(axes, cents):
        if r8 is not None:
            a = r8[r8["centrality"] == cent]
            if not a.empty:
                ax.plot(a["y_cen"], a["value"], "s", label="8.16 TeV")
        if r5 is not None:
            b = r5[r5["centrality"] == cent]
            if not b.empty:
                ax.plot(b["y_cen"], b["value"], "o", label="5.02 TeV")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$y_{\rm cms}$")    # x-axis on every panel
        ax.set_title(cent)
        ax.grid(alpha=0.2)

    # turn off unused axes
    for ax in axes[n:]:
        ax.axis("off")

    axes[0].set_ylabel(r"$\langle R_{pA}\rangle_{p_T}$")  # common y label on first
    # global legend if any data present
    handles, labels = [], []
    for ax in axes[:n]:
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    if handles:
        # deduplicate labels
        seen = set(); H, L = [], []
        for h, l in zip(handles, labels):
            if l in seen: continue
            seen.add(l); H.append(h); L.append(l)
        fig.legend(H, L, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=(0,0,0.9,0.95))
    return fig, axes

