#!/usr/bin/env python3
"""
fit_2r2c_disturbance.py

Fresh-start 2R2C fit on filtered training segments with a latent heat disturbance.

Model (continuous time)
-----------------------
State: x = [Ti, Tm, qd]^T
Inputs: u = [Tout]

Ci dTi/dt = (Tout - Ti)/Rao + (Tm - Ti)/Ria + qd
Cm dTm/dt = (Ti - Tm)/Ria
qd dot = 0  (random walk in discrete time)

We fit Ria, Rao, Cm, volume V, and sigma_qd (process noise of qd) by maximizing a
Kalman-filter log-likelihood. Ci is derived from the fitted room volume V.

Outputs
-------
- result JSON with fitted parameters, fit metrics, and segment filters
- optional CSV with filtered Ti, estimated qd, and segment ids
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from segment_filters import (
    FilterSpec,
    build_mask,
    filter_ranges_by_variation,
    infer_dt_seconds,
    prepare_dataframe,
    segment_ranges_from_mask,
)


@dataclass(frozen=True)
class Config:
    zip_path: Path
    time_col: str = "time"
    tin_col: str = "binnentemperatuur"
    tout_col: str = "buitentemperatuur"
    volume_m3: float = 140.0
    sigma_meas: float = 0.05
    sigma_proc_ti: float = 0.005
    sigma_proc_tm: float = 0.02
    sigma_proc_qd_init: float = 50.0
    require_cv_off: bool = True
    require_hp_off: bool = True
    max_solar: float | None = 20.0
    min_segment_len: int = 20
    min_tin_range: float = 0.3
    min_tout_range: float = 0.5
    maxiter: int = 80
    out_json: Path = Path("r2r2c_disturbance_result.json")
    out_csv: Path = Path("r2r2c_disturbance_filtered.csv")


@dataclass(frozen=True)
class Segment:
    segment_id: int
    tin: np.ndarray
    tout: np.ndarray
    times: np.ndarray


def _air_capacitance_from_volume(volume_m3: float) -> float:
    return float(1.2 * 1005.0 * volume_m3)


def _read_zip_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected single file in zip, got: {names}")
        with zf.open(names[0]) as f:
            return pd.read_csv(f)


def _build_filter_spec(cfg: Config) -> FilterSpec:
    return FilterSpec(
        time_col=cfg.time_col,
        tin_col=cfg.tin_col,
        tout_col=cfg.tout_col,
        require_cv_off=cfg.require_cv_off,
        require_hp_off=cfg.require_hp_off,
        max_solar=cfg.max_solar,
        min_segment_len=cfg.min_segment_len,
        min_tin_range=cfg.min_tin_range,
        min_tout_range=cfg.min_tout_range,
    )


def _build_continuous_matrices(Ria: float, Rao: float, Ci: float, Cm: float) -> Tuple[np.ndarray, np.ndarray]:
    if min(Ria, Rao, Ci, Cm) <= 0:
        raise ValueError("All parameters must be > 0")

    a11 = -(1.0 / (Ci * Rao) + 1.0 / (Ci * Ria))
    a12 = 1.0 / (Ci * Ria)
    a13 = 1.0 / Ci
    a21 = 1.0 / (Cm * Ria)
    a22 = -1.0 / (Cm * Ria)

    A = np.array(
        [
            [a11, a12, a13],
            [a21, a22, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    B = np.array([[1.0 / (Ci * Rao)], [0.0], [0.0]], dtype=float)
    return A, B


def _discretize(A: np.ndarray, B: np.ndarray, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt_s)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def _kalman_nll(
    params_log: np.ndarray,
    *,
    segments: Sequence[Segment],
    dt_s: float,
    cfg: Config,
) -> float:
    log_ria, log_rao, log_cm, log_volume, log_sigma_q = params_log
    Ria = 10.0 ** float(log_ria)
    Rao = 10.0 ** float(log_rao)
    Cm = 10.0 ** float(log_cm)
    volume_m3 = 10.0 ** float(log_volume)
    Ci = _air_capacitance_from_volume(volume_m3)
    sigma_q = 10.0 ** float(log_sigma_q)

    try:
        A, B = _build_continuous_matrices(Ria, Rao, Ci, Cm)
        Ad, Bd = _discretize(A, B, dt_s)
    except Exception:
        return 1e12

    H = np.array([[1.0, 0.0, 0.0]], dtype=float)

    Q = np.diag([
        cfg.sigma_proc_ti ** 2,
        cfg.sigma_proc_tm ** 2,
        sigma_q**2,
    ])
    R = np.array([[cfg.sigma_meas ** 2]], dtype=float)

    nll = 0.0
    for seg in segments:
        tin = seg.tin
        tout = seg.tout
        x = np.array([tin[0], tin[0], 0.0], dtype=float)
        P = np.diag([0.5**2, 0.5**2, cfg.sigma_proc_qd_init**2]).astype(float)

        for k in range(1, len(tin)):
            u = float(tout[k - 1])
            x_pred = Ad @ x + Bd[:, 0] * u
            P_pred = Ad @ P @ Ad.T + Q

            y = float(tin[k])
            y_pred = float((H @ x_pred)[0])
            S = float((H @ P_pred @ H.T + R)[0][0])
            if S <= 0:
                return 1e12

            innov = y - y_pred
            nll += 0.5 * (math.log(2.0 * math.pi * S) + (innov**2) / S)

            K = (P_pred @ H.T)[:, 0] / S
            x = x_pred + K * innov
            P = P_pred - np.outer(K, H @ P_pred)

    return float(nll)


def _run_kalman(
    *,
    Ria: float,
    Rao: float,
    Cm: float,
    sigma_q: float,
    Ci: float,
    segments: Sequence[Segment],
    dt_s: float,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A, B = _build_continuous_matrices(Ria, Rao, Ci, Cm)
    Ad, Bd = _discretize(A, B, dt_s)

    H = np.array([[1.0, 0.0, 0.0]], dtype=float)
    Q = np.diag([
        cfg.sigma_proc_ti ** 2,
        cfg.sigma_proc_tm ** 2,
        sigma_q**2,
    ])
    R = np.array([[cfg.sigma_meas ** 2]], dtype=float)

    xs_all: List[np.ndarray] = []
    tin_filt_all: List[float] = []
    times_all: List[np.ndarray] = []
    tout_all: List[np.ndarray] = []
    seg_ids_all: List[np.ndarray] = []

    for seg in segments:
        tin = seg.tin
        tout = seg.tout
        x = np.array([tin[0], tin[0], 0.0], dtype=float)
        P = np.diag([0.5**2, 0.5**2, cfg.sigma_proc_qd_init**2]).astype(float)

        xs = [x.copy()]
        for k in range(1, len(tin)):
            u = float(tout[k - 1])
            x_pred = Ad @ x + Bd[:, 0] * u
            P_pred = Ad @ P @ Ad.T + Q

            y = float(tin[k])
            y_pred = float((H @ x_pred)[0])
            S = float((H @ P_pred @ H.T + R)[0][0])
            K = (P_pred @ H.T)[:, 0] / S
            x = x_pred + K * (y - y_pred)
            P = P_pred - np.outer(K, H @ P_pred)
            xs.append(x.copy())

        xs_arr = np.array(xs)
        xs_all.append(xs_arr)
        tin_filt_all.append(xs_arr[:, 0])
        times_all.append(seg.times)
        tout_all.append(seg.tout)
        seg_ids_all.append(np.full(len(seg.tin), seg.segment_id, dtype=int))

    xs_concat = np.vstack(xs_all)
    tin_filt = np.concatenate(tin_filt_all)
    times = np.concatenate(times_all)
    tout = np.concatenate(tout_all)
    seg_ids = np.concatenate(seg_ids_all)
    return xs_concat, tin_filt, times, tout, seg_ids


def fit_model(cfg: Config) -> Dict[str, float]:
    df = _read_zip_csv(cfg.zip_path)
    spec = _build_filter_spec(cfg)
    df = prepare_dataframe(df, spec=spec)
    dt_s = infer_dt_seconds(df, time_col=cfg.time_col)
    mask = build_mask(df, spec=spec)
    ranges = segment_ranges_from_mask(df, time_col=cfg.time_col, mask=mask, dt_seconds=dt_s)
    ranges = filter_ranges_by_variation(
        df,
        ranges,
        tin_col=cfg.tin_col,
        tout_col=cfg.tout_col,
        min_segment_len=cfg.min_segment_len,
        min_tin_range=cfg.min_tin_range,
        min_tout_range=cfg.min_tout_range,
    )
    tin = df[cfg.tin_col].to_numpy(dtype=float)
    tout = df[cfg.tout_col].to_numpy(dtype=float)
    times = df[cfg.time_col].to_numpy()
    segments = [
        Segment(segment_id=idx, tin=tin[start:end], tout=tout[start:end], times=times[start:end])
        for idx, (start, end) in enumerate(ranges)
    ]

    x0 = np.array(
        [
            math.log10(0.05),
            math.log10(0.5),
            math.log10(8e6),
            math.log10(cfg.volume_m3),
            math.log10(50.0),
        ]
    )
    bounds = [(-6, 1), (-6, 2), (4, 9), (0, 4), (-3, 4)]

    nll = partial(_kalman_nll, segments=segments, dt_s=dt_s, cfg=cfg)
    res = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": cfg.maxiter},
    )

    if not res.success:
        print(f"Warning: optimization did not converge: {res.message}", file=sys.stderr)

    log_ria, log_rao, log_cm, log_volume, log_sigma_q = res.x
    Ria = 10.0 ** float(log_ria)
    Rao = 10.0 ** float(log_rao)
    Cm = 10.0 ** float(log_cm)
    volume_m3 = 10.0 ** float(log_volume)
    Ci = _air_capacitance_from_volume(volume_m3)
    sigma_q = 10.0 ** float(log_sigma_q)

    xs, tin_filt, times, tout, seg_ids = _run_kalman(
        Ria=Ria,
        Rao=Rao,
        Cm=Cm,
        sigma_q=sigma_q,
        Ci=Ci,
        segments=segments,
        dt_s=dt_s,
        cfg=cfg,
    )

    tin = np.concatenate([seg.tin for seg in segments])
    rmse = float(np.sqrt(np.mean((tin - tin_filt) ** 2)))

    out_rows = []
    for t, tin_meas, tin_pred, qd, tout_val, seg_id in zip(
        times, tin, tin_filt, xs[:, 2], tout, seg_ids
    ):
        out_rows.append(
            {
                "time": pd.Timestamp(t).isoformat(),
                "segment_id": int(seg_id),
                "Tin_meas": float(tin_meas),
                "Tin_filt": float(tin_pred),
                "Tout": float(tout_val),
                "Qdist_est_W": float(qd),
            }
        )

    pd.DataFrame(out_rows).to_csv(cfg.out_csv, index=False)

    result = {
        "Ria_K_per_W": Ria,
        "Rao_K_per_W": Rao,
        "volume_m3": volume_m3,
        "Ci_J_per_K": Ci,
        "Cm_J_per_K": Cm,
        "sigma_qd_W": sigma_q,
        "dt_seconds": dt_s,
        "rmse_C": rmse,
        "data_points": int(len(tin)),
        "segments": int(len(segments)),
        "segment_filters": {
            "require_cv_off": cfg.require_cv_off,
            "require_hp_off": cfg.require_hp_off,
            "max_solar": cfg.max_solar,
            "min_segment_len": cfg.min_segment_len,
            "min_tin_range": cfg.min_tin_range,
            "min_tout_range": cfg.min_tout_range,
        },
        "outputs": {
            "filtered_csv": str(cfg.out_csv),
        },
    }
    cfg.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fit 2R2C with latent heat disturbance.")
    parser.add_argument("--zip", dest="zip_path", default="old/train.zip")
    parser.add_argument("--volume", dest="volume_m3", type=float, default=140.0)
    parser.add_argument("--out-json", dest="out_json", default="r2r2c_disturbance_result.json")
    parser.add_argument("--out-csv", dest="out_csv", default="r2r2c_disturbance_filtered.csv")
    parser.add_argument("--time-col", dest="time_col", default="time")
    parser.add_argument("--tin-col", dest="tin_col", default="binnentemperatuur")
    parser.add_argument("--tout-col", dest="tout_col", default="buitentemperatuur")
    parser.add_argument("--sigma-meas", dest="sigma_meas", type=float, default=0.05)
    parser.add_argument("--sigma-proc-ti", dest="sigma_proc_ti", type=float, default=0.005)
    parser.add_argument("--sigma-proc-tm", dest="sigma_proc_tm", type=float, default=0.02)
    parser.add_argument("--sigma-proc-qd-init", dest="sigma_proc_qd_init", type=float, default=50.0)
    parser.add_argument("--allow-cv-on", dest="require_cv_off", action="store_false")
    parser.add_argument("--allow-hp-on", dest="require_hp_off", action="store_false")
    parser.add_argument("--max-solar", dest="max_solar", type=float, default=20.0)
    parser.add_argument("--min-segment-len", dest="min_segment_len", type=int, default=20)
    parser.add_argument("--min-tin-range", dest="min_tin_range", type=float, default=0.3)
    parser.add_argument("--min-tout-range", dest="min_tout_range", type=float, default=0.5)
    parser.add_argument("--maxiter", dest="maxiter", type=int, default=80)
    args = parser.parse_args()

    max_solar = None if args.max_solar is not None and args.max_solar < 0 else args.max_solar

    return Config(
        zip_path=Path(args.zip_path),
        time_col=args.time_col,
        tin_col=args.tin_col,
        tout_col=args.tout_col,
        volume_m3=args.volume_m3,
        sigma_meas=args.sigma_meas,
        sigma_proc_ti=args.sigma_proc_ti,
        sigma_proc_tm=args.sigma_proc_tm,
        sigma_proc_qd_init=args.sigma_proc_qd_init,
        require_cv_off=args.require_cv_off,
        require_hp_off=args.require_hp_off,
        max_solar=max_solar,
        min_segment_len=args.min_segment_len,
        min_tin_range=args.min_tin_range,
        min_tout_range=args.min_tout_range,
        maxiter=args.maxiter,
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
    )


def main() -> None:
    cfg = _parse_args()
    result = fit_model(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
