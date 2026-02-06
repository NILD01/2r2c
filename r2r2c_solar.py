#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r2r2c_solar.py — Full-forward indoor temperature prediction with:
  - fixed 2R2C parameters from r2r2c_result.json
  - fixed solar gain parameters from solar_result.json
  - ONLINE Kalman warm-up on the last N "past" samples to estimate the hidden mass state (Tm)

Pipeline (predict mode)
-----------------------
1) df_all = past + future
2) Q_trans(t) via solar.py window model (Erbs + Perez + shading + IAM + shutters)
3) Solar dynamics + split:
      Qf(t)    = 1st-order lag of Q_trans with tau_solar
      Q_solar  = k_gain * Qf + q_bias
      Qi(t)    = f_air * Q_solar
      Qm(t)    = (1-f_air) * (k_gain * Qf)     (bias to air only)
4) Kalman warm-up on past:
      x=[Ti, Tm],  y=Ti_measured,  u=[Tout, Qi, Qm]
5) Open-loop simulate future → Tin_pred

Optional: eval mode over segments_data.csv (segment_id groups)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm


# ──────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────

def _write_stdout_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False, allow_nan=False))


def _df_to_records_no_nan(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df2 = df.replace({np.inf: None, -np.inf: None})
    df2 = df2.astype(object).where(pd.notnull(df2), None)
    return df2.to_dict(orient="records")


# ──────────────────────────────────────────────────────────────
# Paths / config
# ──────────────────────────────────────────────────────────────

def _resolve_project_root(root_dir: str, name: Optional[str]) -> Path:
    base = Path(root_dir).expanduser()
    if base.is_dir() and (base / "data").exists():
        return base
    if name:
        cand = base / name
        if cand.is_dir():
            return cand
    if base.is_dir():
        return base
    raise FileNotFoundError(f"Cannot resolve project root from rootDir={root_dir!r}, name={name!r}")


def _find_first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist:\n" + "\n".join(str(p) for p in paths))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────
# Solar gain computation (reuse solar.py implementation)
# ──────────────────────────────────────────────────────────────

def _import_solar_module(project_root: Path) -> Any:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        import solar as solar_mod  # type: ignore
        return solar_mod
    except Exception as e:
        raise ImportError(
            "Failed to import solar.py. Make sure solar.py is in your project root and is importable.\n"
            f"project_root={project_root}\n"
            f"error={type(e).__name__}: {e}"
        ) from e


def _solar_lag(q_trans: np.ndarray, *, dt_s: float, tau_s: float) -> np.ndarray:
    if len(q_trans) == 0:
        return np.zeros(0, dtype=float)
    tau = float(tau_s)
    if tau <= 1e-9:
        return q_trans.astype(float)

    alpha = 1.0 - math.exp(-float(dt_s) / tau)
    qf = np.empty_like(q_trans, dtype=float)
    qf[0] = float(q_trans[0])
    for k in range(1, len(q_trans)):
        qf[k] = qf[k - 1] + alpha * (float(q_trans[k]) - qf[k - 1])
    return qf


def _compute_solar_inputs(
    solar_mod: Any,
    df: pd.DataFrame,
    windows: List[Dict[str, Any]],
    solar_params: Dict[str, float],
    *,
    time_col: str = "time",
    ghi_col: str = "zonnestraling",
    sun_az_col: str = "zonazimut",
    sun_elev_col: str = "zonhoogte",
    albedo: float = 0.2,
    dt_s: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre, _dbg = solar_mod._precompute_windows(
        df,
        windows,
        time_col=time_col,
        ghi_col=ghi_col,
        sun_az_col=sun_az_col,
        sun_elev_col=sun_elev_col,
        albedo=float(albedo),
    )
    q_trans = solar_mod._compute_q_trans(
        pre,
        b0_iam=float(solar_params["b0_iam"]),
        alpha_shutter=float(solar_params["alpha_shutter"]),
    )

    q_filt = _solar_lag(q_trans, dt_s=float(dt_s), tau_s=float(solar_params["tau_solar_s"]))

    k_gain = float(solar_params["k_gain"])
    f_air = float(solar_params["f_air"])
    q_bias = float(solar_params.get("q_bias_W", 0.0))

    q_solar = k_gain * q_filt
    Qi = f_air * q_solar + q_bias
    Qm = (1.0 - f_air) * q_solar
    return q_trans, q_filt, Qi, Qm


# ──────────────────────────────────────────────────────────────
# 2R2C + Kalman warm-up
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class R2R2CParams:
    Ria: float
    Rao: float
    Ci: float
    Cm: float
    dt_s: float


def _discretize_2r2c(params: R2R2CParams) -> Tuple[np.ndarray, np.ndarray]:
    Ria, Rao, Ci, Cm, dt = params.Ria, params.Rao, params.Ci, params.Cm, params.dt_s

    A = np.array(
        [
            [-(1.0 / Rao + 1.0 / Ria) / Ci, (1.0 / Ria) / Ci],
            [(1.0 / Ria) / Cm, -(1.0 / Ria) / Cm],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [(1.0 / Rao) / Ci, 1.0 / Ci, 0.0],
            [0.0, 0.0, 1.0 / Cm],
        ],
        dtype=float,
    )

    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


@dataclass(frozen=True)
class KalmanCfg:
    sigma_meas_C: float = 0.02
    sigma_proc_Ti_C: float = 0.005
    sigma_proc_Tm_C: float = 0.05


def _kalman_filter_trace(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Tout: np.ndarray,
    Qi: np.ndarray,
    Qm: np.ndarray,
    Ti_obs: np.ndarray,
    warmup_n: int,
    cfg: KalmanCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_obs = len(Ti_obs)
    warmup_n = int(max(1, min(warmup_n, n_obs)))

    H = np.array([[1.0, 0.0]], dtype=float)
    I = np.eye(2)

    R = np.array([[float(cfg.sigma_meas_C) ** 2]], dtype=float)
    Q = np.diag([float(cfg.sigma_proc_Ti_C) ** 2, float(cfg.sigma_proc_Tm_C) ** 2]).astype(float)

    x = np.array([float(Ti_obs[0]), float(Ti_obs[0])], dtype=float)
    P = np.diag([0.1**2, 5.0**2]).astype(float)

    Ti_filt = np.empty(warmup_n, dtype=float)

    # update at k=0
    y0 = np.array([float(Ti_obs[0])], dtype=float)
    S = H @ P @ H.T + R
    K = (P @ H.T) / S
    x = x + (K @ (y0 - H @ x)).ravel()
    P = (I - K @ H) @ P
    x[0] = float(Ti_obs[0])
    Ti_filt[0] = x[0]

    for k in range(warmup_n - 1):
        u = np.array([float(Tout[k]), float(Qi[k]), float(Qm[k])], dtype=float)
        x = Ad @ x + Bd @ u
        P = Ad @ P @ Ad.T + Q

        y = np.array([float(Ti_obs[k + 1])], dtype=float)
        S = H @ P @ H.T + R
        K = (P @ H.T) / S
        x = x + (K @ (y - H @ x)).ravel()
        P = (I - K @ H) @ P

        x[0] = float(Ti_obs[k + 1])
        Ti_filt[k + 1] = x[0]

    return x, P, Ti_filt


def _simulate_forward(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Tout: np.ndarray,
    Qi: np.ndarray,
    Qm: np.ndarray,
    x0: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    n = len(Tout)
    x = np.array(x0, dtype=float).copy()
    Ti_pred = np.full(n, np.nan, dtype=float)
    Ti_pred[start_idx - 1] = float(x[0])

    for k in range(start_idx - 1, n - 1):
        u = np.array([float(Tout[k]), float(Qi[k]), float(Qm[k])], dtype=float)
        x = Ad @ x + Bd @ u
        Ti_pred[k + 1] = float(x[0])
    return Ti_pred


# ──────────────────────────────────────────────────────────────
# Payload handling
# ──────────────────────────────────────────────────────────────

def _payload_to_df(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
    if "past" not in payload or "future" not in payload:
        raise KeyError("payload must contain keys 'past' and 'future'.")

    past = payload["past"]
    future = payload["future"]
    if not isinstance(past, list) or not isinstance(future, list):
        raise TypeError("payload['past'] and payload['future'] must be lists of objects.")

    n_past = len(past)
    df = pd.DataFrame(list(past) + list(future))

    if "time" not in df.columns:
        raise KeyError("Expected a 'time' column.")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        bad = df[df["time"].isna()].head(3).to_dict(orient="records")
        raise ValueError(f"Some 'time' values could not be parsed. Examples: {bad}")

    return df, n_past


def _parse_kalman_cfg(payload: Dict[str, Any]) -> KalmanCfg:
    k = payload.get("kalman", {}) if isinstance(payload, dict) else {}
    if not isinstance(k, dict):
        return KalmanCfg()
    return KalmanCfg(
        sigma_meas_C=float(k.get("sigma_meas_C", KalmanCfg.sigma_meas_C)),
        sigma_proc_Ti_C=float(k.get("sigma_proc_Ti_C", KalmanCfg.sigma_proc_Ti_C)),
        sigma_proc_Tm_C=float(k.get("sigma_proc_Tm_C", KalmanCfg.sigma_proc_Tm_C)),
    )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def run_r2r2c_solar(data: Dict[str, Any]) -> Dict[str, Any]:
    name = str(data.get("name", "") or "")
    root_dir = str(data.get("rootDir", "") or "")
    subproject = data.get("subProject") or data.get("subproject") or "r2r2c_solar"
    r2_sub = str(data.get("r2r2c_SubProject", "r2r2c"))
    solar_sub = str(data.get("solar_SubProject", "solar"))
    windows = data.get("windows", [])
    payload = data.get("payload", {}) or {}

    project_root = _resolve_project_root(root_dir, name if name else None)

    r2_path = _find_first_existing(
        [
            project_root / r2_sub / "r2r2c_result.json",
            project_root / r2_sub / "results" / "r2r2c_result.json",
            project_root / "r2r2c_result.json",
        ]
    )
    solar_path = _find_first_existing(
        [
            project_root / solar_sub / "solar_result.json",
            project_root / solar_sub / "results" / "solar_result.json",
            project_root / "solar_result.json",
        ]
    )

    r2 = _load_json(r2_path)
    sol = _load_json(solar_path)

    tin_col = "binnentemperatuur"
    tout_col = "buitentemperatuur"

    fit = r2["fit"]
    params = R2R2CParams(
        Ria=float(fit["Ria_K_per_W"]),
        Rao=float(fit["Rao_K_per_W"]),
        Ci=float(fit["Ci_J_per_K"]),
        Cm=float(fit["Cm_J_per_K"]),
        dt_s=float(r2.get("inputs", {}).get("dt_seconds", 60.0)),
    )
    solar_params = sol["solar_fit"]["params"]
    albedo = float(sol.get("inputs", {}).get("albedo", 0.2))

    mode = str(payload.get("mode", "predict"))

    out: Dict[str, Any] = {
        "ok": True,
        "name": name,
        "project_root": str(project_root),
        "subProject": subproject,
        "mode": mode,
        "sources": {
            "r2r2c_result": str(r2_path),
            "solar_result": str(solar_path),
        },
        "params": {
            "r2r2c": {
                "Ria_K_per_W": params.Ria,
                "Rao_K_per_W": params.Rao,
                "Ci_J_per_K": params.Ci,
                "Cm_J_per_K": params.Cm,
                "dt_s": params.dt_s,
            },
            "solar": dict(solar_params),
            "albedo": albedo,
        },
    }

    solar_mod = _import_solar_module(project_root)
    Ad, Bd = _discretize_2r2c(params)
    kcfg = _parse_kalman_cfg(payload)
    out["kalman"] = {"cfg": kcfg.__dict__}

    if mode.lower() == "predict":
        df_all, n_past = _payload_to_df(payload)

        missing = [c for c in [tout_col, "zonnestraling", "zonazimut", "zonhoogte"] if c not in df_all.columns]
        if missing:
            raise KeyError(f"Missing required columns in payload: {missing}")
        if tin_col not in df_all.columns:
            raise KeyError(f"Missing required column: {tin_col!r}")

        q_trans, q_filt, Qi, Qm = _compute_solar_inputs(
            solar_mod,
            df_all,
            windows,
            solar_params,
            time_col="time",
            ghi_col="zonnestraling",
            sun_az_col="zonazimut",
            sun_elev_col="zonhoogte",
            albedo=albedo,
            dt_s=params.dt_s,
        )

        Tout = df_all[tout_col].to_numpy(dtype=float)
        Ti_obs = df_all.iloc[:n_past][tin_col].to_numpy(dtype=float)

        if np.isnan(Ti_obs).any():
            raise ValueError("NaN in past binnentemperatuur. Warm-up needs observed Tin in 'past'.")

        x_end, P_end, Ti_filt = _kalman_filter_trace(
            Ad,
            Bd,
            Tout=Tout[:n_past],
            Qi=Qi[:n_past],
            Qm=Qm[:n_past],
            Ti_obs=Ti_obs,
            warmup_n=n_past,
            cfg=kcfg,
        )

        Ti_pred_future = _simulate_forward(
            Ad,
            Bd,
            Tout=Tout,
            Qi=Qi,
            Qm=Qm,
            x0=x_end,
            start_idx=n_past,
        )

        df_out = df_all.copy()
        df_out["Tin_obs"] = df_out[tin_col].astype(float, errors="ignore")
        df_out["Tin_filt"] = np.nan
        df_out.loc[: n_past - 1, "Tin_filt"] = Ti_filt
        df_out["Tin_pred"] = np.nan
        df_out.loc[: n_past - 1, "Tin_pred"] = Ti_obs
        df_out.loc[n_past:, "Tin_pred"] = Ti_pred_future[n_past:]

        df_out["Qtrans_proxy_W"] = q_trans
        df_out["Qtrans_filt_W"] = q_filt
        df_out["Qi_W"] = Qi
        df_out["Qm_W"] = Qm

        warmup_fit_rmse = float(np.sqrt(np.mean((Ti_obs - Ti_filt) ** 2)))

        out["kalman"].update(
            {
                "warmup_n": int(n_past),
                "warmup_fit_rmse_C": warmup_fit_rmse,
                "x_end": {"Ti": float(x_end[0]), "Tm": float(x_end[1])},
                "P_end": P_end.tolist(),
            }
        )

        return_arrays = bool(payload.get("return_arrays", False))
        out["results"] = {"future": _df_to_records_no_nan(df_out.iloc[n_past:][["time", "Tin_pred"]])}
        if return_arrays:
            out["results"]["past"] = _df_to_records_no_nan(df_out.iloc[:n_past][["time", "Tin_obs", "Tin_filt"]])
            out["results"]["debug_full"] = _df_to_records_no_nan(
                df_out[["time", "Tin_obs", tout_col, "Tin_filt", "Tin_pred", "Qtrans_proxy_W", "Qtrans_filt_W", "Qi_W", "Qm_W"]]
            )

        out_dir = project_root / str(subproject)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "predict_r2r2c_solar_debug.csv"
        df_out.to_csv(csv_path, index=False)
        out["artifacts"] = {"debug_csv": str(csv_path)}
        return out

    if mode.lower() == "eval":
        warmup_n = int(payload.get("warmup_n", 60))
        seg_path = _find_first_existing(
            [
                project_root / str(subproject) / "segments_data.csv",
                project_root / "segments_data.csv",
                project_root / "data" / "segments_data.csv",
            ]
        )
        df = pd.read_csv(seg_path)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

        q_trans_full, _q_filt_full, _Qi_full, _Qm_full = _compute_solar_inputs(
            solar_mod,
            df,
            windows,
            solar_params,
            time_col="time",
            ghi_col="zonnestraling",
            sun_az_col="zonazimut",
            sun_elev_col="zonhoogte",
            albedo=albedo,
            dt_s=params.dt_s,
        )

        seg_col = "segment_id"
        if seg_col not in df.columns:
            raise KeyError(f"segments_data.csv must contain '{seg_col}'")
        if tin_col not in df.columns or tout_col not in df.columns:
            raise KeyError(f"segments_data.csv must contain '{tin_col}' and '{tout_col}'")

        tau_s = float(solar_params["tau_solar_s"])
        k_gain = float(solar_params["k_gain"])
        f_air = float(solar_params["f_air"])
        q_bias = float(solar_params.get("q_bias_W", 0.0))

        per_seg = []
        rmses = []
        r2s = []

        for sid, g in df.groupby(seg_col):
            idx = g.index.to_numpy()
            Tin = g[tin_col].to_numpy(dtype=float)
            Tout = g[tout_col].to_numpy(dtype=float)
            if len(Tin) < warmup_n + 10:
                continue

            qt = q_trans_full[idx]
            qf = _solar_lag(qt, dt_s=params.dt_s, tau_s=tau_s)
            qsolar = k_gain * qf
            Qi_s = f_air * qsolar + q_bias
            Qm_s = (1.0 - f_air) * qsolar

            x_end, _P_end, _Ti_filt = _kalman_filter_trace(
                Ad,
                Bd,
                Tout=Tout[:warmup_n],
                Qi=Qi_s[:warmup_n],
                Qm=Qm_s[:warmup_n],
                Ti_obs=Tin[:warmup_n],
                warmup_n=warmup_n,
                cfg=kcfg,
            )

            Ti_pred = _simulate_forward(
                Ad,
                Bd,
                Tout=Tout,
                Qi=Qi_s,
                Qm=Qm_s,
                x0=x_end,
                start_idx=warmup_n,
            )

            y_true = Tin[warmup_n:]
            y_pred = Ti_pred[warmup_n:]
            err = y_true - y_pred
            rmse = float(np.sqrt(np.mean(err**2)))

            ss_res = float(np.sum(err**2))
            ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2)) + 1e-12
            r2 = float(1.0 - ss_res / ss_tot)

            per_seg.append({"segment_id": int(sid), "n": int(len(Tin)), "rmse_C": rmse, "r2_tail": r2})
            rmses.append(rmse)
            r2s.append(r2)

        out["sources"]["segments_data"] = str(seg_path)
        out["eval"] = {
            "warmup_n": warmup_n,
            "segments_scored": int(len(per_seg)),
            "rmse_C_mean": float(np.mean(rmses)) if rmses else None,
            "r2_tail_mean": float(np.mean(r2s)) if r2s else None,
            "per_segment": per_seg[:200],
            "note": "r2_tail can be very negative when the segment tail is almost flat; RMSE is more meaningful then.",
        }
        return out

    raise ValueError(f"Unknown mode={mode!r}. Supported: 'predict', 'eval'.")


if __name__ == "__main__":
    try:
        data = json.loads(sys.stdin.read())
        out = run_r2r2c_solar(data)
        _write_stdout_json(out)
    except Exception as e:
        err = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(err, ensure_ascii=False))
        raise
