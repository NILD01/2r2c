#!/usr/bin/env python3
"""
fit_solar_disturbance.py

Fit solar gain parameters on the full dataset using fixed 2R2C parameters
and a latent heat disturbance (random-walk qd) for robustness.

Model (continuous time)
-----------------------
State: x = [Ti, Tm, qd]^T
Inputs: u = [Tout, Qi_solar, Qm_solar]

Ci dTi/dt = (Tout - Ti)/Rao + (Tm - Ti)/Ria + Qi_solar + qd
Cm dTm/dt = (Ti - Tm)/Ria + Qm_solar
qd dot = 0 (random walk in discrete time)

Solar model uses Erbs + Perez + horizon shading + IAM + shutters.
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from solar_model import (
    SolarParams,
    compute_q_trans,
    compute_solar_inputs,
    load_windows_config,
    precompute_windows,
    solar_lag,
    summarize_windows,
)


@dataclass(frozen=True)
class Config:
    zip_path: Path
    r2r2c_result: Path
    windows_config: Path
    time_col: str = "time"
    tin_col: str = "binnentemperatuur"
    tout_col: str = "buitentemperatuur"
    ghi_col: str = "zonnestraling"
    sun_az_col: str = "zonazimut"
    sun_elev_col: str = "zonhoogte"
    albedo: float = 0.2
    sigma_meas: float = 0.05
    sigma_proc_ti: float = 0.005
    sigma_proc_tm: float = 0.02
    sigma_proc_qd_init: float = 50.0
    maxiter: int = 120
    out_json: Path = Path("solar_disturbance_result.json")
    out_csv: Path = Path("solar_disturbance_filtered.csv")


@dataclass(frozen=True)
class PreparedData:
    tin: np.ndarray
    tout: np.ndarray
    dt_s: float
    times: np.ndarray
    df: pd.DataFrame


def _read_zip_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected single file in zip, got: {names}")
        with zf.open(names[0]) as f:
            return pd.read_csv(f)


def _prepare_series(df: pd.DataFrame, cfg: Config) -> PreparedData:
    required = [cfg.time_col, cfg.tin_col, cfg.tout_col, cfg.ghi_col, cfg.sun_az_col, cfg.sun_elev_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    df = df[required].copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    df[cfg.tin_col] = pd.to_numeric(df[cfg.tin_col], errors="coerce")
    df[cfg.tout_col] = pd.to_numeric(df[cfg.tout_col], errors="coerce")
    df[cfg.ghi_col] = pd.to_numeric(df[cfg.ghi_col], errors="coerce")
    df[cfg.sun_az_col] = pd.to_numeric(df[cfg.sun_az_col], errors="coerce")
    df[cfg.sun_elev_col] = pd.to_numeric(df[cfg.sun_elev_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col, cfg.tin_col, cfg.tout_col])
    df = df.sort_values(cfg.time_col)

    if len(df) < 10:
        raise ValueError("Not enough valid samples after cleaning.")

    dt_seconds = float(df[cfg.time_col].diff().dt.total_seconds().dropna().median())
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("Could not infer a positive dt from time column.")

    tin = df[cfg.tin_col].to_numpy(dtype=float)
    tout = df[cfg.tout_col].to_numpy(dtype=float)
    times = df[cfg.time_col].to_numpy()
    return PreparedData(tin=tin, tout=tout, dt_s=dt_seconds, times=times, df=df)


def _build_continuous_matrices(
    Ria: float,
    Rao: float,
    Ci: float,
    Cm: float,
) -> Tuple[np.ndarray, np.ndarray]:
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

    B = np.array(
        [
            [1.0 / (Ci * Rao), 1.0 / Ci, 0.0],
            [0.0, 0.0, 1.0 / Cm],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return A, B


def _discretize(A: np.ndarray, B: np.ndarray, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt_s)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _unpack_theta(theta: np.ndarray) -> Tuple[SolarParams, float]:
    log_k, logit_f, log_tau, logit_alpha, logit_b0, q_bias, log_sigma_q = [float(v) for v in theta]
    k_gain = math.exp(log_k)
    f_air = _sigmoid(logit_f)
    tau_s = math.exp(log_tau)
    alpha_sh = _sigmoid(logit_alpha)
    b0 = 0.2 * _sigmoid(logit_b0)
    sigma_q = math.exp(log_sigma_q)
    return (
        SolarParams(
            k_gain=k_gain,
            f_air=f_air,
            tau_solar_s=tau_s,
            alpha_shutter=alpha_sh,
            b0_iam=b0,
            q_bias_W=q_bias,
        ),
        sigma_q,
    )


def _kalman_nll(
    theta: np.ndarray,
    *,
    tin: np.ndarray,
    tout: np.ndarray,
    dt_s: float,
    cfg: Config,
    A_base: np.ndarray,
    pre_windows: List,
) -> float:
    params, sigma_q = _unpack_theta(theta)

    q_trans = compute_q_trans(pre_windows, b0_iam=params.b0_iam, alpha_shutter=params.alpha_shutter)
    q_filt = q_trans if params.tau_solar_s <= 1e-9 else solar_lag(q_trans, dt_s=dt_s, tau_s=params.tau_solar_s)

    q_solar = params.k_gain * q_filt
    qi = params.f_air * q_solar + params.q_bias_W
    qm = (1.0 - params.f_air) * q_solar

    A, B = A_base
    Ad, Bd = _discretize(A, B, dt_s)

    H = np.array([[1.0, 0.0, 0.0]], dtype=float)
    Q = np.diag([cfg.sigma_proc_ti**2, cfg.sigma_proc_tm**2, sigma_q**2])
    R = np.array([[cfg.sigma_meas**2]], dtype=float)

    x = np.array([tin[0], tin[0], 0.0], dtype=float)
    P = np.diag([0.5**2, 0.5**2, cfg.sigma_proc_qd_init**2]).astype(float)

    nll = 0.0
    for k in range(1, len(tin)):
        u = np.array([float(tout[k - 1]), float(qi[k - 1]), float(qm[k - 1])], dtype=float)
        x_pred = Ad @ x + Bd @ u
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
    tin: np.ndarray,
    tout: np.ndarray,
    qi: np.ndarray,
    qm: np.ndarray,
    dt_s: float,
    cfg: Config,
    A: np.ndarray,
    B: np.ndarray,
    sigma_q: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ad, Bd = _discretize(A, B, dt_s)
    H = np.array([[1.0, 0.0, 0.0]], dtype=float)
    Q = np.diag([cfg.sigma_proc_ti**2, cfg.sigma_proc_tm**2, sigma_q**2])
    R = np.array([[cfg.sigma_meas**2]], dtype=float)

    x = np.array([tin[0], tin[0], 0.0], dtype=float)
    P = np.diag([0.5**2, 0.5**2, cfg.sigma_proc_qd_init**2]).astype(float)

    ti_filt = np.zeros_like(tin, dtype=float)
    tm_filt = np.zeros_like(tin, dtype=float)
    qd_filt = np.zeros_like(tin, dtype=float)
    ti_filt[0] = x[0]
    tm_filt[0] = x[1]
    qd_filt[0] = x[2]

    for k in range(1, len(tin)):
        u = np.array([float(tout[k - 1]), float(qi[k - 1]), float(qm[k - 1])], dtype=float)
        x_pred = Ad @ x + Bd @ u
        P_pred = Ad @ P @ Ad.T + Q

        y = float(tin[k])
        y_pred = float((H @ x_pred)[0])
        S = float((H @ P_pred @ H.T + R)[0][0])
        if S <= 0:
            S = 1e-6
        innov = y - y_pred
        K = (P_pred @ H.T)[:, 0] / S
        x = x_pred + K * innov
        P = P_pred - np.outer(K, H @ P_pred)

        ti_filt[k] = x[0]
        tm_filt[k] = x[1]
        qd_filt[k] = x[2]

    return ti_filt, tm_filt, qd_filt


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (np.sum(err**2) / denom)) if denom > 0 else float("nan")
    return {"rmse_C": rmse, "mae_C": mae, "bias_C": bias, "r2": r2}


def _load_r2r2c(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fit_solar(cfg: Config) -> Dict[str, float]:
    df = _read_zip_csv(cfg.zip_path)
    prepared = _prepare_series(df, cfg)

    r2 = _load_r2r2c(cfg.r2r2c_result)
    Ria = float(r2["Ria_K_per_W"])
    Rao = float(r2["Rao_K_per_W"])
    Ci = float(r2["Ci_J_per_K"])
    Cm = float(r2["Cm_J_per_K"])

    windows = load_windows_config(cfg.windows_config)
    pre_windows, dbg = precompute_windows(
        prepared.df,
        windows,
        time_col=cfg.time_col,
        ghi_col=cfg.ghi_col,
        sun_az_col=cfg.sun_az_col,
        sun_elev_col=cfg.sun_elev_col,
        albedo=cfg.albedo,
    )

    A, B = _build_continuous_matrices(Ria, Rao, Ci, Cm)

    theta0 = np.array(
        [
            math.log(0.15),
            0.0,
            math.log(1800.0),
            0.0,
            0.0,
            0.0,
            math.log(float(r2.get("sigma_qd_W", 10.0))),
        ],
        dtype=float,
    )

    objective = lambda th: _kalman_nll(
        th,
        tin=prepared.tin,
        tout=prepared.tout,
        dt_s=prepared.dt_s,
        cfg=cfg,
        A_base=(A, B),
        pre_windows=pre_windows,
    )

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": cfg.maxiter},
    )

    params, sigma_q = _unpack_theta(res.x)
    solar_inputs = compute_solar_inputs(
        prepared.df,
        windows,
        params,
        time_col=cfg.time_col,
        ghi_col=cfg.ghi_col,
        sun_az_col=cfg.sun_az_col,
        sun_elev_col=cfg.sun_elev_col,
        albedo=cfg.albedo,
        dt_s=prepared.dt_s,
    )

    ti_filt, tm_filt, qd_filt = _run_kalman(
        tin=prepared.tin,
        tout=prepared.tout,
        qi=solar_inputs.qi,
        qm=solar_inputs.qm,
        dt_s=prepared.dt_s,
        cfg=cfg,
        A=A,
        B=B,
        sigma_q=sigma_q,
    )

    metrics = _metrics(prepared.tin, ti_filt)

    out_rows = []
    for t, tin, tout, ti_k, tm_k, qd_k, qi, qm, qt, qf in zip(
        prepared.times,
        prepared.tin,
        prepared.tout,
        ti_filt,
        tm_filt,
        qd_filt,
        solar_inputs.qi,
        solar_inputs.qm,
        solar_inputs.q_trans,
        solar_inputs.q_filt,
    ):
        out_rows.append(
            {
                "time": pd.Timestamp(t).isoformat(),
                "Tin_meas": float(tin),
                "Tin_filt": float(ti_k),
                "Tm_filt": float(tm_k),
                "Tout": float(tout),
                "qd_W": float(qd_k),
                "Qi_solar_W": float(qi),
                "Qm_solar_W": float(qm),
                "Q_trans": float(qt),
                "Q_trans_filt": float(qf),
            }
        )

    pd.DataFrame(out_rows).to_csv(cfg.out_csv, index=False)

    result = {
        "k_gain": params.k_gain,
        "f_air": params.f_air,
        "tau_solar_s": params.tau_solar_s,
        "alpha_shutter": params.alpha_shutter,
        "b0_iam": params.b0_iam,
        "q_bias_W": params.q_bias_W,
        "sigma_qd_W": sigma_q,
        "dt_seconds": prepared.dt_s,
        "data_points": int(len(prepared.tin)),
        "metrics": metrics,
        "optimizer": {
            "success": bool(res.success),
            "nll": float(res.fun),
            "message": str(res.message),
            "nit": int(res.nit),
        },
        "r2r2c": {
            "Ria_K_per_W": Ria,
            "Rao_K_per_W": Rao,
            "Ci_J_per_K": Ci,
            "Cm_J_per_K": Cm,
        },
        "solar_columns": {
            "ghi_col": cfg.ghi_col,
            "sun_az_col": cfg.sun_az_col,
            "sun_elev_col": cfg.sun_elev_col,
        },
        "windows": summarize_windows(windows),
        "outputs": {"filtered_csv": str(cfg.out_csv)},
    }

    cfg.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fit solar parameters with disturbance.")
    parser.add_argument("--zip", dest="zip_path", default="old/train.zip")
    parser.add_argument("--r2r2c", dest="r2r2c_result", default="r2r2c_disturbance_result.json")
    parser.add_argument("--windows", dest="windows_config", default="windows_config.json")
    parser.add_argument("--out-json", dest="out_json", default="solar_disturbance_result.json")
    parser.add_argument("--out-csv", dest="out_csv", default="solar_disturbance_filtered.csv")
    parser.add_argument("--time-col", dest="time_col", default="time")
    parser.add_argument("--tin-col", dest="tin_col", default="binnentemperatuur")
    parser.add_argument("--tout-col", dest="tout_col", default="buitentemperatuur")
    parser.add_argument("--ghi-col", dest="ghi_col", default="zonnestraling")
    parser.add_argument("--sun-az-col", dest="sun_az_col", default="zonazimut")
    parser.add_argument("--sun-elev-col", dest="sun_elev_col", default="zonhoogte")
    parser.add_argument("--albedo", type=float, default=0.2)
    parser.add_argument("--sigma-meas", type=float, default=0.05)
    parser.add_argument("--sigma-proc-ti", type=float, default=0.005)
    parser.add_argument("--sigma-proc-tm", type=float, default=0.02)
    parser.add_argument("--sigma-proc-qd-init", type=float, default=50.0)
    parser.add_argument("--maxiter", type=int, default=120)

    args = parser.parse_args()
    return Config(
        zip_path=Path(args.zip_path),
        r2r2c_result=Path(args.r2r2c_result),
        windows_config=Path(args.windows_config),
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
        time_col=args.time_col,
        tin_col=args.tin_col,
        tout_col=args.tout_col,
        ghi_col=args.ghi_col,
        sun_az_col=args.sun_az_col,
        sun_elev_col=args.sun_elev_col,
        albedo=args.albedo,
        sigma_meas=args.sigma_meas,
        sigma_proc_ti=args.sigma_proc_ti,
        sigma_proc_tm=args.sigma_proc_tm,
        sigma_proc_qd_init=args.sigma_proc_qd_init,
        maxiter=args.maxiter,
    )


def main() -> None:
    cfg = _parse_args()
    result = fit_solar(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
