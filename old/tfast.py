#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# tfast.py
#
# Doel:
#   Bepaal tfast (snelle tijdconstante) uit segments_data.csv
#   met "warmtepomp OFF" afkoelsegmenten.
#
# Methode:
#   Fit discrete 2e-orde ARX(2,2) op Tin met Tout als exogene input:
#     Tin[k+1] = a1*Tin[k] + a2*Tin[k-1] + b0*Tout[k] + b1*Tout[k-1] + c
#
#   Poles uit r^2 - a1*r - a2 = 0  => r1,r2 in (0,1)
#   Tijdconstanten: tau = -dt / ln(r)
#
#   tfast = "dominante snelle pool" = grootste tau van de twee (maar nog steeds << tslow).
#
# Output:
#   schrijft <subproject_dir>/tfast_result.json
# ─────────────────────────────────────────────────────────────


def _as_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _resolve_subproject_dir(root_dir: str, name: str, subproject: str) -> Path:
    base = Path(root_dir)

    # A) rootDir = .../<name>
    cand_a = base / subproject
    if cand_a.exists():
        return cand_a

    # B) rootDir = .../ (parent)
    cand_b = base / name / subproject
    if cand_b.exists():
        return cand_b

    # Fallback: maak B aan
    return cand_b


def _read_segments_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"segments CSV niet gevonden: {csv_path}")
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if df.empty:
        raise ValueError(f"segments CSV is leeg: {csv_path}")
    return df


def _ensure_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Ontbrekende kolommen in {context}: {missing}. Beschikbaar: {list(df.columns)}"
        )


def _infer_dt_seconds(df: pd.DataFrame, segment_col: str, time_col: str) -> float:
    if time_col not in df.columns:
        return 300.0

    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df2 = df.loc[~t.isna(), [segment_col, time_col]].copy()
    df2[time_col] = pd.to_datetime(df2[time_col], utc=True, errors="coerce")
    df2 = df2.sort_values([segment_col, time_col])

    dts: List[float] = []
    for _, g in df2.groupby(segment_col, sort=False):
        if len(g) < 4:
            continue
        dt = g[time_col].diff().dt.total_seconds().dropna()
        if not dt.empty:
            dts.append(float(dt.median()))

    return float(np.median(dts)) if dts else 300.0


def _build_arx_segment_mats(
    tin: np.ndarray,
    tout: np.ndarray,
    max_steps: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bouw X,y voor één segment (ARX2):
      y = Tin[k+1]
      X = [Tin[k], Tin[k-1], Tout[k], Tout[k-1], 1]
    met k = 1..n-2

    max_steps: gebruik enkel eerste N samples (vanaf start segment) om "snelle zone" te isoleren.
    """
    if max_steps is not None and max_steps > 0:
        tin = tin[:max_steps]
        tout = tout[:max_steps]

    n = len(tin)
    if n < 4:
        return np.zeros((0, 5), dtype=float), np.zeros((0,), dtype=float)

    rows: List[List[float]] = []
    ys: List[float] = []

    for k in range(1, n - 1):
        y = tin[k + 1]
        x = [tin[k], tin[k - 1], tout[k], tout[k - 1], 1.0]
        if np.isfinite(y) and np.all(np.isfinite(x)):
            rows.append(x)
            ys.append(float(y))

    if not rows:
        return np.zeros((0, 5), dtype=float), np.zeros((0,), dtype=float)

    return np.asarray(rows, dtype=float), np.asarray(ys, dtype=float)


def _fit_arx2_from_Xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Least squares fit, return beta + RMSE + R² (one-step).
    beta = [a1, a2, b0, b1, c]
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    res = y - yhat
    rmse = float(np.sqrt(np.mean(res**2)))
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return beta.astype(float), rmse, r2


def _roots_and_taus(beta: np.ndarray, dt_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roots uit r^2 - a1*r - a2 = 0.
    Return (roots_sorted_desc, taus_seconds_sorted_desc)
    """
    a1 = float(beta[0])
    a2 = float(beta[1])

    roots = np.roots([1.0, -a1, -a2])
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots > 0.0) & (roots < 1.0)]
    if roots.size != 2:
        raise ValueError(f"Ongeldige roots voor stabiel 2e-orde model: {roots}")

    roots = np.sort(roots)[::-1]  # r1>r2
    taus = np.array([-dt_seconds / math.log(r) for r in roots], dtype=float)
    return roots, taus


def run_tfast(data: Dict[str, Any]) -> Dict[str, Any]:
    name = str(data.get("name") or "")
    root_dir = str(data.get("rootDir") or "")
    subproject = str(data.get("subProject") or "")

    if not name:
        raise ValueError("data.name ontbreekt")
    if not root_dir:
        raise ValueError("data.rootDir ontbreekt")
    if not subproject:
        raise ValueError("data.subProject ontbreekt")

    segment_col = str(data.get("segment_col") or "segment_id")
    time_col = str(data.get("time_col") or "time")
    tin_col = str(data.get("tin_col") or "binnentemperatuur")
    tout_col = str(data.get("tout_col") or "buitentemperatuur")

    csv_name = str(data.get("csv_name") or "segments_data.csv")
    out_name = str(data.get("out_name") or "tfast_result.json")

    # Snelle zone: default eerste 24 stappen = 2 uur bij 5-min data.
    max_steps = data.get("max_steps_per_segment")
    if max_steps is None:
        max_steps_i: Optional[int] = 24
    else:
        max_steps_i = _as_int(max_steps, 24)
        if max_steps_i <= 0:
            max_steps_i = None

    bootstrap_n = _as_int(data.get("bootstrap"), 5000)
    seed = _as_int(data.get("seed"), 0)

    # Optioneel: sanity filter dat deze segmenten echt "off" zijn
    off_col = str(data.get("off_col") or "warmtepomp_mode_off")
    require_off = bool(data.get("require_off", False))

    sub_dir = _resolve_subproject_dir(root_dir, name, subproject)
    sub_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sub_dir / csv_name
    if not csv_path.exists():
        alt = sub_dir / "data" / csv_name
        if alt.exists():
            csv_path = alt

    df = _read_segments_csv(csv_path)

    # segment col fallback
    if segment_col not in df.columns:
        for alt_seg in ("segment", "segmentId", "seg_id", "segmentID"):
            if alt_seg in df.columns:
                segment_col = alt_seg
                break

    _ensure_columns(df, [segment_col, tin_col, tout_col], "segments_data.csv")

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.sort_values([segment_col, time_col])
    else:
        df = df.sort_values([segment_col])

    if require_off and off_col in df.columns:
        df = df[df[off_col] >= 0.5].copy()

    dt_seconds = _infer_dt_seconds(df, segment_col=segment_col, time_col=time_col)

    # Precompute per-segment X,y voor snelle zone
    seg_ids = [sid for sid in df[segment_col].dropna().unique().tolist()]
    seg_mats: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
    n_rows_total = 0

    for sid, g in df.groupby(segment_col, sort=False):
        tin = pd.to_numeric(g[tin_col], errors="coerce").to_numpy(dtype=float, copy=False)
        tout = pd.to_numeric(g[tout_col], errors="coerce").to_numpy(dtype=float, copy=False)
        Xs, ys = _build_arx_segment_mats(tin, tout, max_steps=max_steps_i)
        if Xs.shape[0] >= 3:
            seg_mats[sid] = (Xs, ys)
            n_rows_total += int(Xs.shape[0])

    if len(seg_mats) < 3 or n_rows_total < 30:
        raise ValueError(
            f"Te weinig bruikbare data voor ARX2 (segments={len(seg_mats)}, rows={n_rows_total}). "
            f"Verhoog max_steps_per_segment of controleer kolommen/NaN."
        )

    # Global fit op alle segmenten samen
    X_all = np.concatenate([seg_mats[sid][0] for sid in seg_mats.keys()], axis=0)
    y_all = np.concatenate([seg_mats[sid][1] for sid in seg_mats.keys()], axis=0)

    beta, rmse, r2 = _fit_arx2_from_Xy(X_all, y_all)
    roots, taus = _roots_and_taus(beta, dt_seconds)

    # Dominante snelle mode = grootste tau (maar nog steeds "fast" t.o.v. tslow)
    tfast_s = float(taus[0])
    tfast_h = float(tfast_s / 3600.0)

    tfast2_s = float(taus[1])
    tfast2_h = float(tfast2_s / 3600.0)

    warnings: List[str] = []
    if tfast_h > 12:
        warnings.append(
            f"tfast lijkt onverwacht groot ({tfast_h:.2f} h). "
            f"Overweeg max_steps_per_segment te verlagen (bv. 12–18) om enkel de snelle zone te fitten."
        )

    # Bootstrap CI (segment-resample)
    ci_tfast_h: Optional[List[float]] = None
    ci_tfast2_h: Optional[List[float]] = None

    if bootstrap_n > 0 and len(seg_mats) >= 3:
        rng = np.random.default_rng(seed)
        keys = list(seg_mats.keys())

        boot1: List[float] = []
        boot2: List[float] = []

        for _ in range(int(bootstrap_n)):
            sample = rng.choice(keys, size=len(keys), replace=True)
            Xb = np.concatenate([seg_mats[sid][0] for sid in sample], axis=0)
            yb = np.concatenate([seg_mats[sid][1] for sid in sample], axis=0)

            try:
                bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
                rr, tt = _roots_and_taus(bb.astype(float), dt_seconds)
                boot1.append(float(tt[0] / 3600.0))
                boot2.append(float(tt[1] / 3600.0))
            except Exception:
                continue

        if len(boot1) >= max(50, int(0.1 * bootstrap_n)):
            ci_tfast_h = [
                float(np.percentile(boot1, 2.5)),
                float(np.percentile(boot1, 97.5)),
            ]
            ci_tfast2_h = [
                float(np.percentile(boot2, 2.5)),
                float(np.percentile(boot2, 97.5)),
            ]
        else:
            warnings.append("Bootstrap CI kon niet robuust berekend worden (te weinig geldige samples).")

    result: Dict[str, Any] = {
        "name": name,
        "subProject": subproject,
        "paths": {
            "subproject_dir": str(sub_dir),
            "csv_path": str(csv_path),
        },
        "columns": {
            "segment_col": segment_col,
            "time_col": time_col if time_col in df.columns else None,
            "tin_col": tin_col,
            "tout_col": tout_col,
        },
        "data_stats": {
            "n_rows": int(len(df)),
            "n_segments": int(len(seg_mats)),
            "dt_seconds": float(dt_seconds),
            "max_steps_per_segment": int(max_steps_i) if max_steps_i is not None else None,
            "n_arx_rows_used": int(X_all.shape[0]),
        },
        "fit": {
            "beta": {
                "a1": float(beta[0]),
                "a2": float(beta[1]),
                "b0": float(beta[2]),
                "b1": float(beta[3]),
                "c": float(beta[4]),
            },
            "roots": [float(roots[0]), float(roots[1])],
            "tfast_seconds": float(tfast_s),
            "tfast_hours": float(tfast_h),
            "tfast2_seconds": float(tfast2_s),
            "tfast2_hours": float(tfast2_h),
            "rmse_1step_C": float(rmse),
            "r2_1step": float(r2),
            "bootstrap_ci_tfast_hours": ci_tfast_h,
            "bootstrap_ci_tfast2_hours": ci_tfast2_h,
        },
        "warnings": warnings,
    }

    out_path = sub_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    result["paths"]["out_json"] = str(out_path)
    return result


if __name__ == "__main__":
    example = {
        "name": "3R2C_test",
        "rootDir": "/home/nilsdebaer/scripts",
        "subProject": "tfast",
        "bootstrap": 2000,
        "seed": 0,
        # "max_steps_per_segment": 18,
    }
    print(json.dumps(run_tfast(example), indent=2, ensure_ascii=False))
