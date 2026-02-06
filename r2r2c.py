#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r2r2c.py — Fit a 2R2C grey-box model on tslow + tfast segments (full-forward).

What this script does
---------------------
You already estimated two dominant time-scales from filtered segments:
  - tslow_result.json → tau_slow (slow mode)
  - tfast_result.json → tau_fast (fast mode)

This script:
  1) Reads both segments_data.csv files (tslow + tfast).
  2) Builds a physically-consistent 2R2C model:
        Ci dTi/dt = (To-Ti)/Rao + (Tm-Ti)/Ria
        Cm dTm/dt = (Ti-Tm)/Ria
     where Ti = indoor air temperature, Tm = lumped "mass" temperature, To = outdoor temperature.
  3) Uses V (m³) to set Ci (air capacitance) = rho_air * cp_air * V (J/K).
  4) Enforces the two target time constants via the eigenvalues of A (physics-informed constraint),
     and searches the remaining free DOF (Ria) to best match *full-forward* simulation errors
     on both tslow and tfast segments.
  5) Refines the parameters with a quick local optimizer (keeps tau close to the targets).

Important detail: initial "mass temperature" Tm0 per segment
-----------------------------------------------------------
Setting Tm0 := Ti0 is often wrong right after e.g. heating stops or a window opens.
So for each segment, we estimate the best Tm0 (1 scalar) in closed form:
we simulate twice (Tm0=Ti0 and Tm0=Ti0+1°C) to get a sensitivity vector,
then solve a least-squares delta that minimizes the full-forward SSE.

Node-RED payload example
------------------------
{
  "name": "3R2C_test",
  "rootDir": "/home/nilsdebaer/scripts",
  "subProject": "r2r2c",
  "tslowSubProject": "tslow",
  "tfastSubProject": "tfast",
  "v": 140
}

Outputs
-------
<project>/<subProject>/r2r2c_result.json
<project>/<subProject>/r2r2c_predictions.csv   (optional, useful for debugging)

Authoring note
--------------
This script is deterministic and reproducible. It does NOT assume heating/solar gains,
because your segments are filtered for "no active sources".
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


# ───────────────────────── IO helpers ─────────────────────────

def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _resolve_project_root(root_dir: str, name: str) -> Path:
    """
    rootDir can be:
      - /home/.../scripts
      - /home/.../scripts/<name>
    We want: <proj> = /home/.../scripts/<name>
    """
    base = Path(root_dir)
    return base if base.name == name else (base / name)


@dataclass(frozen=True)
class Segment:
    dataset: str  # "tslow" or "tfast"
    segment_id: int
    time_utc: np.ndarray  # datetime64[ns, UTC]
    Tin: np.ndarray       # °C
    Tout: np.ndarray      # °C


def _read_segments_csv(
    csv_path: Path,
    *,
    dataset: str,
    segment_col: str = "segment_id",
    time_col: str = "time",
    tin_col: str = "binnentemperatuur",
    tout_col: str = "buitentemperatuur",
    min_len: int = 10,
) -> List[Segment]:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    missing = [c for c in [segment_col, time_col, tin_col, tout_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {missing}")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values([segment_col, time_col])

    out: List[Segment] = []
    for sid, g in df.groupby(segment_col, sort=False):
        t = g[time_col].to_numpy()
        Tin = pd.to_numeric(g[tin_col], errors="coerce").to_numpy(dtype=float)
        Tout = pd.to_numeric(g[tout_col], errors="coerce").to_numpy(dtype=float)

        m = np.isfinite(Tin) & np.isfinite(Tout)
        t = t[m]
        Tin = Tin[m]
        Tout = Tout[m]

        if len(Tin) >= min_len:
            out.append(Segment(dataset=dataset, segment_id=int(sid), time_utc=t, Tin=Tin, Tout=Tout))
    if not out:
        raise ValueError(f"No usable segments found in {csv_path}")
    return out


# ───────────────────────── 2R2C model ─────────────────────────

def air_capacitance_from_volume(V_m3: float, *, rho_air: float = 1.2, cp_air: float = 1005.0) -> float:
    """Ci [J/K] from room volume (rho*cp*V)."""
    return float(rho_air * cp_air * V_m3)


def build_continuous_matrices(Ria: float, Rao: float, Ci: float, Cm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time state-space:
      x = [Ti, Tm]
      u = To
      dx/dt = A x + B u
    """
    if min(Ria, Rao, Ci, Cm) <= 0:
        raise ValueError("All parameters must be > 0")

    a11 = -(1.0 / (Ci * Rao) + 1.0 / (Ci * Ria))
    a12 = 1.0 / (Ci * Ria)
    a21 = 1.0 / (Cm * Ria)
    a22 = -1.0 / (Cm * Ria)

    A = np.array([[a11, a12],
                  [a21, a22]], dtype=float)

    B = np.array([[1.0 / (Ci * Rao)],
                  [0.0]], dtype=float)
    return A, B


def discretize_exact(A: np.ndarray, B: np.ndarray, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact discretization via a single matrix exponential:
      [Ad Bd; 0 1] = exp( [A B; 0 0] * dt )
    """
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")

    M = np.block([
        [A, B],
        [np.zeros((1, 3), dtype=float)]
    ])
    Md = expm(M * dt_s)
    Ad = Md[:2, :2]
    Bd = Md[:2, 2:3]
    return Ad, Bd


def continuous_time_constants(A: np.ndarray) -> Tuple[float, float]:
    """
    Return (tau_fast, tau_slow) [seconds] from continuous-time eigenvalues.
    """
    eig = np.linalg.eigvals(A)
    taus: List[float] = []
    for lam in eig:
        lam_r = float(np.real(lam))
        if lam_r >= 0:
            taus.append(float("inf"))
        else:
            taus.append(-1.0 / lam_r)
    taus.sort()
    return float(taus[0]), float(taus[1])


def simulate_with_segment_Tm0_ls(
    seg: Segment,
    Ad: np.ndarray,
    Bd: np.ndarray,
    *,
    clamp_delta_C: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """
    Full-forward simulation for one segment, with Ti0 fixed to measured Tin[0],
    and Tm0 estimated (one scalar) by least squares in closed form.

    Returns:
      Tin_pred (len N), Tm0_est
    """
    Tin = seg.Tin
    Tout = seg.Tout
    n = len(Tin)
    Ti0 = float(Tin[0])

    # 1) base simulation with Tm0 = Ti0
    x = np.array([Ti0, Ti0], dtype=float)
    y_base = np.empty(n, dtype=float)
    y_base[0] = x[0]
    for k in range(n - 1):
        x = Ad @ x + Bd[:, 0] * float(Tout[k])
        y_base[k + 1] = x[0]

    # 2) sensitivity simulation with Tm0 = Ti0 + 1°C
    x = np.array([Ti0, Ti0 + 1.0], dtype=float)
    y_delta = np.empty(n, dtype=float)
    y_delta[0] = x[0]
    for k in range(n - 1):
        x = Ad @ x + Bd[:, 0] * float(Tout[k])
        y_delta[k + 1] = x[0]

    s = y_delta - y_base
    err = Tin - y_base

    # least squares on delta (exclude k=0 where s[0]=0 always)
    ss = float(np.dot(s[1:], s[1:]))
    if ss < 1e-12:
        delta = 0.0
    else:
        delta = float(np.dot(s[1:], err[1:]) / ss)

    delta = float(np.clip(delta, -clamp_delta_C, clamp_delta_C))
    Tm0 = Ti0 + delta
    y = y_base + delta * s
    return y, float(Tm0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ───────────────────────── physics-informed fit ─────────────────────────

@dataclass
class FitResult:
    Ria: float
    Rao: float
    Ci: float
    Cm: float
    tau_fast_s: float
    tau_slow_s: float
    rmse_slow: float
    r2_slow: float
    rmse_fast: float
    r2_fast: float
    rmse_all: float
    r2_all: float


def _evaluate_params(
    *,
    Ria: float,
    Rao: float,
    Ci: float,
    Cm: float,
    dt_s: float,
    segs_slow: Sequence[Segment],
    segs_fast: Sequence[Segment],
) -> FitResult:
    A, B = build_continuous_matrices(Ria, Rao, Ci, Cm)
    tau_fast_s, tau_slow_s = continuous_time_constants(A)
    Ad, Bd = discretize_exact(A, B, dt_s)

    # Collect predictions (exclude k=0 per segment for scoring: "1-step onward full-forward")
    preds_slow: List[np.ndarray] = []
    trues_slow: List[np.ndarray] = []
    preds_fast: List[np.ndarray] = []
    trues_fast: List[np.ndarray] = []

    for seg in segs_slow:
        y, _ = simulate_with_segment_Tm0_ls(seg, Ad, Bd)
        preds_slow.append(y[1:])
        trues_slow.append(seg.Tin[1:])

    for seg in segs_fast:
        y, _ = simulate_with_segment_Tm0_ls(seg, Ad, Bd)
        preds_fast.append(y[1:])
        trues_fast.append(seg.Tin[1:])

    yS = np.concatenate(trues_slow)
    pS = np.concatenate(preds_slow)
    yF = np.concatenate(trues_fast)
    pF = np.concatenate(preds_fast)

    yA = np.concatenate([yS, yF])
    pA = np.concatenate([pS, pF])

    return FitResult(
        Ria=float(Ria),
        Rao=float(Rao),
        Ci=float(Ci),
        Cm=float(Cm),
        tau_fast_s=float(tau_fast_s),
        tau_slow_s=float(tau_slow_s),
        rmse_slow=rmse(yS, pS),
        r2_slow=r2_score(yS, pS),
        rmse_fast=rmse(yF, pF),
        r2_fast=r2_score(yF, pF),
        rmse_all=rmse(yA, pA),
        r2_all=r2_score(yA, pA),
    )


def _solve_Rao_Cm_from_Ria_and_taus(
    *,
    Ria: float,
    Ci: float,
    tau_fast_s_target: float,
    tau_slow_s_target: float,
) -> List[Tuple[float, float]]:
    """
    Use the closed-form eigenvalue constraints to solve for Rao and Cm given Ria and Ci.

    For the 2R2C A-matrix:
      trace = λ1 + λ2
      det   = λ1 * λ2 = 1/(Ci*Cm*Ria*Rao)

    With λ1=-1/tau_fast, λ2=-1/tau_slow.
    Solving yields a quadratic in Rao; for each positive Rao, Cm follows from det.
    """
    if Ria <= 0 or Ci <= 0:
        return []

    lam1 = -1.0 / float(tau_fast_s_target)
    lam2 = -1.0 / float(tau_slow_s_target)
    S = lam1 + lam2                    # trace
    P = lam1 * lam2                    # determinant (>0)

    # K = -S - 1/(Ci*Ria)
    K = (-S) - (1.0 / (Ci * Ria))
    disc = K * K - 4.0 * P
    if disc <= 0:
        return []

    sqrt_disc = math.sqrt(disc)

    # Rao roots (both can be positive)
    # Rao = (K ± sqrt(disc)) / (2 * Ci * P)
    denom = 2.0 * Ci * P
    if denom == 0:
        return []

    Rao1 = (K + sqrt_disc) / denom
    Rao2 = (K - sqrt_disc) / denom

    sols: List[Tuple[float, float]] = []
    for Rao in (Rao1, Rao2):
        if Rao > 0:
            Cm = 1.0 / (Ci * Ria * Rao * P)
            if Cm > 0 and np.isfinite(Cm):
                sols.append((float(Rao), float(Cm)))
    return sols


def fit_2r2c_from_taus_and_segments(
    *,
    V_m3: float,
    dt_s: float,
    tau_fast_s_target: float,
    tau_slow_s_target: float,
    segs_slow: Sequence[Segment],
    segs_fast: Sequence[Segment],
    # search/priors
    Ria_bounds: Tuple[float, float] = (1e-3, 1.0),
    grid_points: int = 160,
    refine: bool = True,
    tau_penalty_weight: float = 0.01,
) -> FitResult:
    """
    Physics-informed fit:
      - Ci computed from V
      - enforce (tau_fast, tau_slow) approximately (hard in grid, soft in refine)
      - pick best on full-forward simulation error across both datasets
    """
    Ci = air_capacitance_from_volume(V_m3)

    # 1) Coarse grid over Ria (log-space)
    lo, hi = Ria_bounds
    if lo <= 0 or hi <= lo:
        raise ValueError("Invalid Ria_bounds")

    grid = np.logspace(math.log10(lo), math.log10(hi), grid_points)

    best: Optional[FitResult] = None
    for Ria in grid:
        sols = _solve_Rao_Cm_from_Ria_and_taus(
            Ria=float(Ria),
            Ci=Ci,
            tau_fast_s_target=tau_fast_s_target,
            tau_slow_s_target=tau_slow_s_target,
        )
        if not sols:
            continue

        # Evaluate both roots, pick the best.
        for Rao, Cm in sols:
            try:
                fr = _evaluate_params(
                    Ria=float(Ria), Rao=float(Rao), Ci=float(Ci), Cm=float(Cm),
                    dt_s=float(dt_s),
                    segs_slow=segs_slow,
                    segs_fast=segs_fast,
                )
            except Exception:
                continue

            if best is None or fr.rmse_all < best.rmse_all:
                best = fr

    if best is None:
        raise RuntimeError("No feasible parameter set found. Try widening Ria_bounds.")

    # 2) Local refinement (lets taus drift slightly, but penalizes it)
    if not refine:
        return best

    def objective_log10(x: np.ndarray) -> float:
        logRia, logRao, logCm = float(x[0]), float(x[1]), float(x[2])
        Ria = 10.0 ** logRia
        Rao = 10.0 ** logRao
        Cm = 10.0 ** logCm

        if min(Ria, Rao, Cm) <= 0:
            return 1e9

        # evaluate
        fr = _evaluate_params(
            Ria=Ria, Rao=Rao, Ci=Ci, Cm=Cm,
            dt_s=dt_s, segs_slow=segs_slow, segs_fast=segs_fast,
        )

        # soft penalty to keep taus close to your pre-fit taus
        pen = (math.log(fr.tau_fast_s / tau_fast_s_target) ** 2 +
               math.log(fr.tau_slow_s / tau_slow_s_target) ** 2)
        return fr.rmse_all + tau_penalty_weight * math.sqrt(pen)

    x0 = np.array([math.log10(best.Ria), math.log10(best.Rao), math.log10(best.Cm)], dtype=float)

    # bounds in log10-space (kept wide; refinement is gentle)
    bounds = [(-6, 2), (-6, 3), (2, 10)]
    res = minimize(objective_log10, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})

    if not res.success:
        # fall back to grid solution
        return best

    Ria = 10.0 ** float(res.x[0])
    Rao = 10.0 ** float(res.x[1])
    Cm = 10.0 ** float(res.x[2])

    return _evaluate_params(
        Ria=Ria, Rao=Rao, Ci=Ci, Cm=Cm,
        dt_s=dt_s, segs_slow=segs_slow, segs_fast=segs_fast,
    )


# ───────────────────────── main entrypoint ─────────────────────────

def run_r2r2c(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point called from Node-RED via pythonCall("r2r2c", data, ...).

    Expected keys in `data`:
      name, rootDir, subProject, tslowSubProject, tfastSubProject, v
    """
    name = str(data["name"])
    root_dir = str(data["rootDir"])
    subproject = str(data["subProject"])
    tslow_sub = str(data["tslowSubProject"])
    tfast_sub = str(data["tfastSubProject"])
    V_m3 = float(data["v"])

    proj = _resolve_project_root(root_dir, name)
    out_dir = proj / subproject
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs (paths)
    tslow_json = proj / tslow_sub / f"{tslow_sub}_result.json"
    tfast_json = proj / tfast_sub / f"{tfast_sub}_result.json"
    tslow_csv = proj / tslow_sub / "segments_data.csv"
    tfast_csv = proj / tfast_sub / "segments_data.csv"

    for p in (tslow_json, tfast_json, tslow_csv, tfast_csv):
        if not p.exists():
            raise FileNotFoundError(str(p))

    obj_slow = _load_json(tslow_json)
    obj_fast = _load_json(tfast_json)

    # Column names (default to your conventions; can be overridden by the json files)
    cols_slow = obj_slow.get("columns", {}) if isinstance(obj_slow, dict) else {}
    cols_fast = obj_fast.get("columns", {}) if isinstance(obj_fast, dict) else {}

    segment_col = str(cols_slow.get("segment_col", "segment_id"))
    time_col = str(cols_slow.get("time_col", "time"))
    tin_col = str(cols_slow.get("tin_col", "binnentemperatuur"))
    tout_col = str(cols_slow.get("tout_col", "buitentemperatuur"))

    # NOTE: we assume both subprojects use the same col names (your pipeline does)
    dt_s = float(obj_slow["data_stats"]["dt_seconds"])
    tau_slow_target = float(obj_slow["fit"]["tslow_seconds"])
    tau_fast_target = float(obj_fast["fit"]["tfast_seconds"])

    # Load segments
    segs_slow = _read_segments_csv(
        tslow_csv, dataset="tslow",
        segment_col=segment_col, time_col=time_col, tin_col=tin_col, tout_col=tout_col,
        min_len=10,
    )
    segs_fast = _read_segments_csv(
        tfast_csv, dataset="tfast",
        segment_col=segment_col, time_col=time_col, tin_col=tin_col, tout_col=tout_col,
        min_len=10,
    )

    # Fit
    fit = fit_2r2c_from_taus_and_segments(
        V_m3=V_m3,
        dt_s=dt_s,
        tau_fast_s_target=tau_fast_target,
        tau_slow_s_target=tau_slow_target,
        segs_slow=segs_slow,
        segs_fast=segs_fast,
        Ria_bounds=(1e-3, 1.0),
        grid_points=160,
        refine=True,
        tau_penalty_weight=0.01,
    )

    # Build prediction dump (debug-friendly)
    A, B = build_continuous_matrices(fit.Ria, fit.Rao, fit.Ci, fit.Cm)
    Ad, Bd = discretize_exact(A, B, dt_s)

    rows: List[Dict[str, Any]] = []
    for seg in list(segs_slow) + list(segs_fast):
        y, Tm0 = simulate_with_segment_Tm0_ls(seg, Ad, Bd)
        for t, Tin, Tout, Tin_pred in zip(seg.time_utc, seg.Tin, seg.Tout, y):
            rows.append({
                "dataset": seg.dataset,
                "segment_id": seg.segment_id,
                "time": pd.Timestamp(t).isoformat(),
                "Tin_meas": float(Tin),
                "Tin_pred": float(Tin_pred),
                "Tout": float(Tout),
                "Tm0_est": float(Tm0),
                "Tm0_minus_Ti0": float(Tm0 - float(seg.Tin[0])),
            })

    pred_path = out_dir / "r2r2c_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    # Result JSON
    result: Dict[str, Any] = {
        "name": name,
        "subProject": subproject,
        "sources": {
            "project_root": str(proj),
            "tslow_result_json": str(tslow_json),
            "tfast_result_json": str(tfast_json),
            "tslow_segments_csv": str(tslow_csv),
            "tfast_segments_csv": str(tfast_csv),
        },
        "inputs": {
            "V_m3": V_m3,
            "Ci_from_volume_J_per_K": air_capacitance_from_volume(V_m3),
            "dt_seconds": dt_s,
            "tau_targets_seconds": {
                "tau_fast": tau_fast_target,
                "tau_slow": tau_slow_target,
            },
            "columns": {
                "segment_col": segment_col,
                "time_col": time_col,
                "tin_col": tin_col,
                "tout_col": tout_col,
            },
        },
        "fit": {
            "Ria_K_per_W": fit.Ria,
            "Rao_K_per_W": fit.Rao,
            "Ci_J_per_K": fit.Ci,
            "Cm_J_per_K": fit.Cm,
            "taus_seconds": {"tau_fast": fit.tau_fast_s, "tau_slow": fit.tau_slow_s},
            "metrics_full_forward": {
                "rmse_slow_C": fit.rmse_slow,
                "r2_slow": fit.r2_slow,
                "rmse_fast_C": fit.rmse_fast,
                "r2_fast": fit.r2_fast,
                "rmse_all_C": fit.rmse_all,
                "r2_all": fit.r2_all,
            },
            "method": {
                "discretization": "exact matrix exponential",
                "Tm0_per_segment": "closed-form least-squares on full-forward residuals (Ti0 fixed)",
                "grid_search": {"parameter": "Ria", "space": "log", "points": 240},
                "refinement": {"optimizer": "L-BFGS-B", "tau_penalty_weight": 0.01},
            },
        },
        "paths": {
            "out_json": str(out_dir / "r2r2c_result.json"),
            "predictions_csv": str(pred_path),
        },
    }

    out_path = out_dir / "r2r2c_result.json"
    _dump_json(out_path, result)
    return result


def _main() -> None:
    """
    CLI mode:
      echo '{"name":"...","rootDir":"...","subProject":"...","tslowSubProject":"tslow","tfastSubProject":"tfast","v":140}' | ./r2r2c.py
    """
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        raise SystemExit(f"Expected JSON on stdin. Error: {e}")

    out = run_r2r2c(payload)
    sys.stdout.write(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    _main()
