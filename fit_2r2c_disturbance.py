#!/usr/bin/env python3
"""
fit_2r2c_disturbance.py

Fresh-start 2R2C fit on the full training dataset with a latent heat disturbance.

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
- result JSON with fitted parameters and fit metrics
- optional CSV with filtered Ti and estimated qd
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
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


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
    maxiter: int = 80
    n_restarts: int = 6
    prior_sigma_log: float = 2.0
    stride: int = 1
    progress_every: int = 10
    out_json: Path = Path("r2r2c_disturbance_result.json")
    out_csv: Path = Path("r2r2c_disturbance_filtered.csv")


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


def _prepare_series(df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    for col in (cfg.time_col, cfg.tin_col, cfg.tout_col):
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r}. Available: {list(df.columns)}")

    df = df[[cfg.time_col, cfg.tin_col, cfg.tout_col]].copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    df[cfg.tin_col] = pd.to_numeric(df[cfg.tin_col], errors="coerce")
    df[cfg.tout_col] = pd.to_numeric(df[cfg.tout_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col, cfg.tin_col, cfg.tout_col])
    df = df.sort_values(cfg.time_col)
    if cfg.stride < 1:
        raise ValueError("Stride must be >= 1.")
    if cfg.stride > 1:
        df = df.iloc[:: cfg.stride].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Not enough valid samples after cleaning.")

    dt_seconds = float(df[cfg.time_col].diff().dt.total_seconds().dropna().median())
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("Could not infer a positive dt from time column.")

    tin = df[cfg.tin_col].to_numpy(dtype=float)
    tout = df[cfg.tout_col].to_numpy(dtype=float)
    times = df[cfg.time_col].to_numpy()
    return tin, tout, dt_seconds, times


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
    tin: np.ndarray,
    tout: np.ndarray,
    dt_s: float,
    cfg: Config,
    prior_mu_log: np.ndarray,
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

    x = np.array([tin[0], tin[0], 0.0], dtype=float)
    P = np.diag([0.5**2, 0.5**2, cfg.sigma_proc_qd_init**2]).astype(float)

    nll = 0.0
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

    if cfg.prior_sigma_log > 0:
        sigma = float(cfg.prior_sigma_log)
        penalty = 0.5 * float(np.sum(((params_log - prior_mu_log) / sigma) ** 2))
        nll += penalty

    return float(nll)


def _run_kalman(
    *,
    Ria: float,
    Rao: float,
    Cm: float,
    sigma_q: float,
    Ci: float,
    tin: np.ndarray,
    tout: np.ndarray,
    dt_s: float,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    A, B = _build_continuous_matrices(Ria, Rao, Ci, Cm)
    Ad, Bd = _discretize(A, B, dt_s)

    H = np.array([[1.0, 0.0, 0.0]], dtype=float)
    Q = np.diag([
        cfg.sigma_proc_ti ** 2,
        cfg.sigma_proc_tm ** 2,
        sigma_q**2,
    ])
    R = np.array([[cfg.sigma_meas ** 2]], dtype=float)

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

    return np.array(xs), np.array([x[0] for x in xs])


def fit_model(cfg: Config) -> Dict[str, float]:
    df = _read_zip_csv(cfg.zip_path)
    tin, tout, dt_s, times = _prepare_series(df, cfg)

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

    rng = np.random.default_rng(0)
    initial_guesses = [x0]
    for _ in range(max(cfg.n_restarts - 1, 0)):
        guess = np.array([rng.uniform(low, high) for low, high in bounds], dtype=float)
        initial_guesses.append(guess)

    nll = partial(_kalman_nll, tin=tin, tout=tout, dt_s=dt_s, cfg=cfg, prior_mu_log=x0)
    best_nll = math.inf
    best_res = None
    for idx, guess in enumerate(initial_guesses, start=1):
        state = {"iter": 0, "last_nll": None}

        def nll_wrapped(params: np.ndarray) -> float:
            value = nll(params)
            state["last_nll"] = value
            return value

        def callback(_xk: np.ndarray) -> None:
            state["iter"] += 1
            if cfg.progress_every <= 0:
                return
            if state["iter"] % cfg.progress_every != 0:
                return
            current_nll = state["last_nll"]
            nonlocal best_nll
            if current_nll is not None:
                best_nll = min(best_nll, current_nll)
            progress = min(state["iter"] / max(cfg.maxiter, 1), 1.0)
            print(
                f"[restart {idx}/{len(initial_guesses)}] "
                f"iter {state['iter']}/{cfg.maxiter} "
                f"({progress:.0%}) "
                f"nll={current_nll:.2f} "
                f"best={best_nll:.2f}",
                flush=True,
            )

        start = time.perf_counter()
        res = minimize(
            nll_wrapped,
            guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": cfg.maxiter},
            callback=callback,
        )
        elapsed = time.perf_counter() - start
        print(
            f"[restart {idx}/{len(initial_guesses)}] done in {elapsed:.1f}s "
            f"nll={res.fun:.2f} iters={res.nit} evals={res.nfev}",
            flush=True,
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    if best_res is None:
        raise RuntimeError("Optimization failed to produce a result.")
    if not best_res.success:
        print(f"Warning: optimization did not converge: {best_res.message}", file=sys.stderr)

    log_ria, log_rao, log_cm, log_volume, log_sigma_q = best_res.x
    Ria = 10.0 ** float(log_ria)
    Rao = 10.0 ** float(log_rao)
    Cm = 10.0 ** float(log_cm)
    volume_m3 = 10.0 ** float(log_volume)
    Ci = _air_capacitance_from_volume(volume_m3)
    sigma_q = 10.0 ** float(log_sigma_q)

    xs, tin_filt = _run_kalman(
        Ria=Ria,
        Rao=Rao,
        Cm=Cm,
        sigma_q=sigma_q,
        Ci=Ci,
        tin=tin,
        tout=tout,
        dt_s=dt_s,
        cfg=cfg,
    )

    rmse = float(np.sqrt(np.mean((tin - tin_filt) ** 2)))

    out_rows = []
    for t, tin_meas, tin_pred, qd in zip(times, tin, tin_filt, xs[:, 2]):
        out_rows.append(
            {
                "time": pd.Timestamp(t).isoformat(),
                "Tin_meas": float(tin_meas),
                "Tin_filt": float(tin_pred),
                "Tout": float(tout[len(out_rows)]),
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
        "nll": float(best_res.fun),
        "data_points": int(len(tin)),
        "stride": int(cfg.stride),
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
    parser.add_argument("--maxiter", dest="maxiter", type=int, default=80)
    parser.add_argument("--n-restarts", dest="n_restarts", type=int, default=6)
    parser.add_argument("--prior-sigma-log", dest="prior_sigma_log", type=float, default=2.0)
    parser.add_argument("--stride", dest="stride", type=int, default=1)
    parser.add_argument("--progress-every", dest="progress_every", type=int, default=10)
    args = parser.parse_args()

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
        maxiter=args.maxiter,
        n_restarts=args.n_restarts,
        prior_sigma_log=args.prior_sigma_log,
        stride=args.stride,
        progress_every=args.progress_every,
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
    )


def main() -> None:
    cfg = _parse_args()
    result = fit_model(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
