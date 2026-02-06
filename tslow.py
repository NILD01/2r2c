#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# tslow.py (v2)
#
# Fix t.o.v. vorige versie:
#   - Correcte discretisatie met exogene input Tout(t):
#       Tin[k+1] = a*Tin[k] + (1-a)*Tout[k]
#     => (Tin[k+1] - Tout[k]) = a*(Tin[k] - Tout[k])
#   - Dus: x = Tin[k] - Tout[k]
#          y = Tin[k+1] - Tout[k]   (NIET Tout[k+1])
#
#   De vorige implementatie gebruikte y = (Tin-Tout)[k+1] = Tin[k+1]-Tout[k+1],
#   wat een extra term (Tout[k]-Tout[k+1]) introduceert en alpha kan vertekenen.
#
#   - Bootstrap/per-segment: ongeldige alpha (<=0 of >=1) wordt nu SKIPPED
#     i.p.v. geclipped naar 1-1e-12 (wat astronomische tau's gaf).
#
# Aanroep via python_daemon:
#   cmd="tslow"
#   data = { "name": ..., "rootDir": ..., "subProject": ... , ...optioneel... }
#
# Bestanden:
#   leest:  <rootDir>/<name>/<subProject>/segments_data.csv  (of <rootDir>/<subProject>/...)
#   schrijft: .../tslow_result.json
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
        raise KeyError(f"Ontbrekende kolommen in {context}: {missing}. Beschikbaar: {list(df.columns)}")


def _infer_dt_seconds(df: pd.DataFrame, segment_col: str, time_col: str) -> float:
    if time_col not in df.columns:
        return 300.0

    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df2 = df.loc[~t.isna(), [segment_col, time_col]].copy()
    df2[time_col] = pd.to_datetime(df2[time_col], utc=True, errors="coerce")
    df2 = df2.sort_values([segment_col, time_col])

    dts: List[float] = []
    for _, g in df2.groupby(segment_col, sort=False):
        if len(g) < 3:
            continue
        dt = g[time_col].diff().dt.total_seconds().dropna()
        if not dt.empty:
            dts.append(float(dt.median()))

    return float(np.median(dts)) if dts else 300.0


def _alpha_from_pairs(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError("Onvoldoende variatie in x om alpha te schatten.")
    return float(np.dot(x, y) / denom)


def _tau_from_alpha(alpha: float, dt_seconds: float) -> float:
    if not np.isfinite(alpha):
        raise ValueError("alpha is niet eindig.")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha buiten (0,1): {alpha}")
    return float(-dt_seconds / math.log(alpha))


def _prepare_pairs_tout_k(
    df: pd.DataFrame,
    segment_col: str,
    time_col: str,
    tin_col: str,
    tout_col: str,
    min_abs_x: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Correcte paren voor variërende Tout:
      x = Tin[k] - Tout[k]
      y = Tin[k+1] - Tout[k]
    """
    _ensure_columns(df, [segment_col, tin_col, tout_col], "segments_data.csv")

    df2 = df.copy()
    if time_col in df2.columns:
        df2[time_col] = pd.to_datetime(df2[time_col], utc=True, errors="coerce")
        df2 = df2.sort_values([segment_col, time_col])
    else:
        df2 = df2.sort_values([segment_col])

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    n_pairs_total = 0
    n_pairs_used = 0

    for _, g in df2.groupby(segment_col, sort=False):
        tin = pd.to_numeric(g[tin_col], errors="coerce").to_numpy(dtype=float, copy=False)
        tout = pd.to_numeric(g[tout_col], errors="coerce").to_numpy(dtype=float, copy=False)
        if len(tin) < 2:
            continue

        x = tin[:-1] - tout[:-1]
        y = tin[1:] - tout[:-1]   # <-- Tout[k]
        n_pairs_total += len(x)

        m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) >= float(min_abs_x))
        xk = x[m]
        yk = y[m]
        if xk.size:
            xs.append(xk)
            ys.append(yk)
            n_pairs_used += int(xk.size)

    if not xs:
        raise ValueError("Geen bruikbare paren. Verlaag min_abs_x of controleer NaN/kolommen.")

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)

    return X, Y, {
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_used": int(n_pairs_used),
        "min_abs_x": float(min_abs_x),
        "pair_definition": "x = Tin[k]-Tout[k], y = Tin[k+1]-Tout[k]"
    }


def _fit_metrics(x: np.ndarray, y: np.ndarray, alpha: float) -> Dict[str, float]:
    yhat = alpha * x
    res = y - yhat
    rmse = float(np.sqrt(np.mean(res**2)))
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y - float(np.mean(y)))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "r2": r2}


def _per_segment_tau_hours(
    df: pd.DataFrame,
    segment_col: str,
    time_col: str,
    tin_col: str,
    tout_col: str,
    dt_seconds: float,
    min_abs_x: float,
) -> List[float]:
    df2 = df.copy()
    if time_col in df2.columns:
        df2[time_col] = pd.to_datetime(df2[time_col], utc=True, errors="coerce")
        df2 = df2.sort_values([segment_col, time_col])
    else:
        df2 = df2.sort_values([segment_col])

    taus_h: List[float] = []
    for _, g in df2.groupby(segment_col, sort=False):
        tin = pd.to_numeric(g[tin_col], errors="coerce").to_numpy(dtype=float, copy=False)
        tout = pd.to_numeric(g[tout_col], errors="coerce").to_numpy(dtype=float, copy=False)
        if len(tin) < 3:
            continue

        x = tin[:-1] - tout[:-1]
        y = tin[1:] - tout[:-1]
        m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) >= float(min_abs_x))
        xk = x[m]
        yk = y[m]
        if xk.size < 10:
            continue

        try:
            a = _alpha_from_pairs(xk, yk)
            if not (0.0 < a < 1.0):
                continue
            tau_s = _tau_from_alpha(a, dt_seconds)
            taus_h.append(float(tau_s / 3600.0))
        except Exception:
            continue

    return taus_h


def run_tslow(data: Dict[str, Any]) -> Dict[str, Any]:
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
    out_name = str(data.get("out_name") or "tslow_result.json")
    min_abs_x = _as_float(data.get("min_abs_x"), 0.05)
    bootstrap_n = _as_int(data.get("bootstrap"), 5000)  # default 5000 om stabiele CI te krijgen
    seed = _as_int(data.get("seed"), 0)

    sub_dir = _resolve_subproject_dir(root_dir, name, subproject)
    sub_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sub_dir / csv_name
    if not csv_path.exists():
        alt = sub_dir / "data" / csv_name
        if alt.exists():
            csv_path = alt

    df = _read_segments_csv(csv_path)

    # Segment kolom fallback
    if segment_col not in df.columns:
        for alt_seg in ("segment", "segmentId", "seg_id", "segmentID"):
            if alt_seg in df.columns:
                segment_col = alt_seg
                break

    dt_seconds = _infer_dt_seconds(df, segment_col=segment_col, time_col=time_col)

    X, Y, diag_pairs = _prepare_pairs_tout_k(
        df,
        segment_col=segment_col,
        time_col=time_col,
        tin_col=tin_col,
        tout_col=tout_col,
        min_abs_x=min_abs_x,
    )

    alpha = _alpha_from_pairs(X, Y)
    warnings: List[str] = []
    if not (0.0 < alpha < 1.0):
        warnings.append(f"Globale alpha buiten (0,1): {alpha} (data bevat waarschijnlijk verstoringen).")

    # tau
    tau_seconds: Optional[float]
    try:
        tau_seconds = _tau_from_alpha(alpha, dt_seconds)
    except Exception as e:
        tau_seconds = None
        warnings.append(f"Kon tslow niet berekenen: {type(e).__name__}: {e}")

    metrics = _fit_metrics(X, Y, alpha)

    _ensure_columns(df, [segment_col, tin_col, tout_col], "segments_data.csv")
    n_rows = int(len(df))
    n_segments = int(df[segment_col].nunique(dropna=True))

    taus_h = _per_segment_tau_hours(
        df,
        segment_col=segment_col,
        time_col=time_col,
        tin_col=tin_col,
        tout_col=tout_col,
        dt_seconds=dt_seconds,
        min_abs_x=min_abs_x,
    )

    per_seg_stats: Dict[str, Any] = {}
    if taus_h:
        per_seg_stats = {
            "n_segments_used": int(len(taus_h)),
            "median_hours": float(np.median(taus_h)),
            "p10_hours": float(np.percentile(taus_h, 10)),
            "p90_hours": float(np.percentile(taus_h, 90)),
            "min_hours": float(np.min(taus_h)),
            "max_hours": float(np.max(taus_h)),
        }
    else:
        warnings.append("Per-segment tau's konden niet berekend worden (te korte/te vlakke segmenten).")

    # Bootstrap CI: resample segmenten, maar SKIP ongeldige alpha's
    ci_hours: Optional[List[float]] = None
    if bootstrap_n > 0 and n_segments >= 3:
        rng = np.random.default_rng(seed)
        df2 = df.copy()
        if time_col in df2.columns:
            df2[time_col] = pd.to_datetime(df2[time_col], utc=True, errors="coerce")
            df2 = df2.sort_values([segment_col, time_col])
        else:
            df2 = df2.sort_values([segment_col])

        seg_ids = [sid for sid in df2[segment_col].dropna().unique().tolist()]
        seg_arrays: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        for sid, g in df2.groupby(segment_col, sort=False):
            seg_arrays[sid] = (
                pd.to_numeric(g[tin_col], errors="coerce").to_numpy(dtype=float, copy=False),
                pd.to_numeric(g[tout_col], errors="coerce").to_numpy(dtype=float, copy=False),
            )

        boot: List[float] = []
        for _ in range(int(bootstrap_n)):
            sample = rng.choice(seg_ids, size=len(seg_ids), replace=True)
            xs: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            for sid in sample:
                tin, tout = seg_arrays[sid]
                if len(tin) < 2:
                    continue
                x = tin[:-1] - tout[:-1]
                y = tin[1:] - tout[:-1]
                m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) >= float(min_abs_x))
                xk = x[m]
                yk = y[m]
                if xk.size:
                    xs.append(xk)
                    ys.append(yk)

            if not xs:
                continue

            Xb = np.concatenate(xs)
            Yb = np.concatenate(ys)
            try:
                ab = _alpha_from_pairs(Xb, Yb)
                if not (0.0 < ab < 1.0):
                    continue
                tb = _tau_from_alpha(ab, dt_seconds)
                boot.append(float(tb / 3600.0))
            except Exception:
                continue

        if len(boot) >= max(50, int(0.1 * bootstrap_n)):
            ci_hours = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        else:
            warnings.append("Bootstrap CI kon niet robuust berekend worden (te weinig geldige bootstrap-samples).")
    elif bootstrap_n > 0:
        warnings.append("Bootstrap CI overgeslagen (te weinig segmenten).")

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
            "n_rows": n_rows,
            "n_segments": n_segments,
            "dt_seconds": float(dt_seconds),
            **diag_pairs,
        },
        "fit": {
            "alpha": float(alpha),
            "tslow_seconds": float(tau_seconds) if tau_seconds is not None else None,
            "tslow_hours": float(tau_seconds / 3600.0) if tau_seconds is not None else None,
            "tslow_days": float(tau_seconds / 86400.0) if tau_seconds is not None else None,
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
            "bootstrap_ci_hours": ci_hours,
            "per_segment_tau_hours": per_seg_stats,
        },
        "warnings": warnings,
    }

    out_path = sub_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    result["paths"]["out_json"] = str(out_path)
    return result


if __name__ == "__main__":
    # dev test (geen argparse)
    example = {
        "name": "3R2C_test",
        "rootDir": "/home/nilsdebaer/scripts",
        "subProject": "T_slow",
        "bootstrap": 0,
    }
    print(json.dumps(run_tslow(example), indent=2, ensure_ascii=False))
