#!/usr/bin/env python3
"""
evaluate_openloop.py

Evaluate open-loop predictions for a fitted 2R2C model on the full dataset.

This script loads the fitted parameters (from fit_2r2c_disturbance.py),
simulates Ti/Tm forward without measurement updates, and reports metrics.
Optional extra gain inputs can be supplied (e.g., solar) via --gain flags.

Example:
  python evaluate_openloop.py \
    --zip old/train.zip \
    --result r2r2c_disturbance_result.json \
    --gain zonnestraling:0.0
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm


@dataclass(frozen=True)
class GainSpec:
    column: str
    scale: float


@dataclass(frozen=True)
class Config:
    zip_path: Path
    result_json: Path
    out_json: Path
    out_csv: Path
    time_col: str
    tin_col: str
    tout_col: str
    gains: Tuple[GainSpec, ...]


def _read_zip_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected single file in zip, got: {names}")
        with zf.open(names[0]) as f:
            return pd.read_csv(f)


def _prepare_series(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    required = [cfg.time_col, cfg.tin_col, cfg.tout_col]
    required.extend(g.column for g in cfg.gains)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    use_cols = [cfg.time_col, cfg.tin_col, cfg.tout_col] + [g.column for g in cfg.gains]
    df = df[use_cols].copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    df[cfg.tin_col] = pd.to_numeric(df[cfg.tin_col], errors="coerce")
    df[cfg.tout_col] = pd.to_numeric(df[cfg.tout_col], errors="coerce")
    for g in cfg.gains:
        df[g.column] = pd.to_numeric(df[g.column], errors="coerce")
    df = df.dropna(subset=[cfg.time_col, cfg.tin_col, cfg.tout_col])
    df = df.sort_values(cfg.time_col)

    if len(df) < 10:
        raise ValueError("Not enough valid samples after cleaning.")

    dt_seconds = float(df[cfg.time_col].diff().dt.total_seconds().dropna().median())
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("Could not infer a positive dt from time column.")

    tin = df[cfg.tin_col].to_numpy(dtype=float)
    tout = df[cfg.tout_col].to_numpy(dtype=float)
    gains = df[[g.column for g in cfg.gains]].to_numpy(dtype=float) if cfg.gains else np.zeros((len(df), 0))
    times = df[cfg.time_col].to_numpy()
    return tin, tout, gains, dt_seconds, times


def _build_continuous_matrices(
    Ria: float,
    Rao: float,
    Ci: float,
    Cm: float,
    gain_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if min(Ria, Rao, Ci, Cm) <= 0:
        raise ValueError("All parameters must be > 0")

    a11 = -(1.0 / (Ci * Rao) + 1.0 / (Ci * Ria))
    a12 = 1.0 / (Ci * Ria)
    a21 = 1.0 / (Cm * Ria)
    a22 = -1.0 / (Cm * Ria)

    A = np.array(
        [
            [a11, a12],
            [a21, a22],
        ],
        dtype=float,
    )

    # Inputs: Tout, then optional gains in W.
    B = np.zeros((2, 1 + gain_count), dtype=float)
    B[0, 0] = 1.0 / (Ci * Rao)
    if gain_count:
        B[0, 1:] = 1.0 / Ci
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


def _load_fit_params(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (np.sum(err**2) / denom)) if denom > 0 else float("nan")
    return {"rmse_C": rmse, "mae_C": mae, "bias_C": bias, "r2": r2}


def evaluate_open_loop(cfg: Config) -> Dict[str, float]:
    df = _read_zip_csv(cfg.zip_path)
    tin, tout, gains, dt_s, times = _prepare_series(df, cfg)

    fit = _load_fit_params(cfg.result_json)
    Ria = float(fit["Ria_K_per_W"])
    Rao = float(fit["Rao_K_per_W"])
    Ci = float(fit["Ci_J_per_K"])
    Cm = float(fit["Cm_J_per_K"])

    A, B = _build_continuous_matrices(Ria, Rao, Ci, Cm, gain_count=gains.shape[1])
    Ad, Bd = _discretize(A, B, dt_s)

    x = np.array([tin[0], tin[0]], dtype=float)
    tin_pred = np.zeros_like(tin)
    tin_pred[0] = x[0]

    scales = np.array([g.scale for g in cfg.gains], dtype=float)
    for k in range(1, len(tin)):
        u = np.zeros(B.shape[1], dtype=float)
        u[0] = float(tout[k - 1])
        if gains.shape[1]:
            u[1:] = gains[k - 1, :] * scales
        x = Ad @ x + Bd @ u
        tin_pred[k] = x[0]

    metrics = _metrics(tin, tin_pred)
    out_rows = []
    for t, tin_meas, tin_sim, tout_val in zip(times, tin, tin_pred, tout):
        out_rows.append(
            {
                "time": pd.Timestamp(t).isoformat(),
                "Tin_meas": float(tin_meas),
                "Tin_openloop": float(tin_sim),
                "Tout": float(tout_val),
                "error_C": float(tin_sim - tin_meas),
            }
        )
    pd.DataFrame(out_rows).to_csv(cfg.out_csv, index=False)

    result = {
        "dt_seconds": dt_s,
        "data_points": int(len(tin)),
        "metrics": metrics,
        "parameters": {
            "Ria_K_per_W": Ria,
            "Rao_K_per_W": Rao,
            "Ci_J_per_K": Ci,
            "Cm_J_per_K": Cm,
        },
        "gains": [{"column": g.column, "scale": g.scale} for g in cfg.gains],
        "outputs": {"openloop_csv": str(cfg.out_csv)},
    }
    cfg.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_gain(raw: str) -> GainSpec:
    if ":" in raw:
        col, scale = raw.split(":", 1)
        return GainSpec(column=col, scale=float(scale))
    return GainSpec(column=raw, scale=1.0)


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Evaluate open-loop 2R2C predictions.")
    parser.add_argument("--zip", dest="zip_path", default="old/train.zip")
    parser.add_argument("--result", dest="result_json", default="r2r2c_disturbance_result.json")
    parser.add_argument("--out-json", dest="out_json", default="r2r2c_openloop_eval.json")
    parser.add_argument("--out-csv", dest="out_csv", default="r2r2c_openloop_eval.csv")
    parser.add_argument("--time-col", dest="time_col", default="time")
    parser.add_argument("--tin-col", dest="tin_col", default="binnentemperatuur")
    parser.add_argument("--tout-col", dest="tout_col", default="buitentemperatuur")
    parser.add_argument(
        "--gain",
        dest="gains",
        action="append",
        default=[],
        help="Optional gain column with scale, e.g. 'zonnestraling:0.05'",
    )
    args = parser.parse_args()

    gains = tuple(_parse_gain(raw) for raw in args.gains)
    return Config(
        zip_path=Path(args.zip_path),
        result_json=Path(args.result_json),
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
        time_col=args.time_col,
        tin_col=args.tin_col,
        tout_col=args.tout_col,
        gains=gains,
    )


def main() -> None:
    cfg = _parse_args()
    result = evaluate_open_loop(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
