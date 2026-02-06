#!/usr/bin/env python3
"""
Reusable solar gain model (Erbs + Perez + horizon + IAM + shutters).

This module provides:
- window precomputation of POA beam/diffuse terms
- transmitted solar power proxy Q_trans
- 1st-order lag and split to air/mass gains (Qi/Qm)

Designed to be shared by fit_solar_disturbance.py and evaluate_openloop.py.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SolarParams:
    k_gain: float
    f_air: float
    tau_solar_s: float
    alpha_shutter: float
    b0_iam: float
    q_bias_W: float = 0.0


@dataclass(frozen=True)
class WindowPrecomp:
    name: str
    area_g: float
    pb: np.ndarray
    pd: np.ndarray
    cos_theta: np.ndarray
    shutter_pos: np.ndarray


@dataclass(frozen=True)
class SolarInputs:
    q_trans: np.ndarray
    q_filt: np.ndarray
    qi: np.ndarray
    qm: np.ndarray


_PEREZ_COEFFS = np.array(
    [
        [-0.008, 0.588, -0.062, -0.060, 0.072, -0.022],
        [0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
        [0.330, 0.487, -0.221, 0.055, -0.064, -0.026],
        [0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
        [0.873, -0.392, -0.362, 0.226, -0.462, 0.001],
        [1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
        [1.060, -1.600, -0.359, 0.264, -1.127, 0.131],
        [0.678, -0.327, -0.250, 0.156, -1.377, 0.251],
    ],
    dtype=float,
)

_EPS_BINS = [1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2]


def load_windows_config(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "windows" in data:
        return list(data["windows"])
    if isinstance(data, list):
        return data
    raise ValueError("Expected JSON list or object with 'windows'.")


def _solar_extraterrestrial_normal(doy: int) -> float:
    I_sc = 1367.0
    gamma = 2.0 * math.pi * (doy - 1) / 365.0
    e0 = (
        1.00011
        + 0.034221 * math.cos(gamma)
        + 0.00128 * math.sin(gamma)
        + 0.000719 * math.cos(2 * gamma)
        + 0.000077 * math.sin(2 * gamma)
    )
    return I_sc * e0


def _kasten_young_airmass(zenith_deg: float) -> float:
    z = float(zenith_deg)
    if z >= 90.0:
        return float("nan")
    cosz = math.cos(math.radians(z))
    return 1.0 / (cosz + 0.50572 * ((96.07995 - z) ** (-1.6364)))


def _erbs_dni_dhi(ghi: float, zenith_deg: float, doy: int) -> Tuple[float, float, float]:
    ghi = float(ghi)
    z = float(zenith_deg)
    cosz = math.cos(math.radians(z))
    i0n = _solar_extraterrestrial_normal(doy)
    i0h = max(i0n * cosz, 1e-6)

    if ghi <= 0.0 or cosz <= 0.065:
        return 0.0, 0.0, i0n

    kt = max(0.0, min(2.0, ghi / i0h))
    if kt <= 0.22:
        df = 1.0 - 0.09 * kt
    elif kt <= 0.8:
        df = 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4
    else:
        df = 0.165

    df = max(0.0, min(1.0, df))
    dhi = df * ghi
    dni = (ghi - dhi) / max(cosz, 1e-6)
    return max(dni, 0.0), max(dhi, 0.0), i0n


def _iam_ashrae(cos_theta: np.ndarray, b0: float) -> np.ndarray:
    ct = np.clip(cos_theta, 1e-6, 1.0)
    iam = 1.0 - b0 * (1.0 / ct - 1.0)
    return np.clip(iam, 0.0, 1.0)


def _horizon_elev_min_for_az(
    az_deg: float,
    horizon: List[Dict[str, Any]],
) -> Optional[float]:
    az = float(az_deg) % 360.0
    for seg in horizon:
        a0 = seg.get("az_min", None)
        a1 = seg.get("az_max", None)
        if a0 is None or a1 is None:
            continue
        a0 = float(a0) % 360.0
        a1 = float(a1) % 360.0

        if a0 <= a1:
            inside = (az >= a0) and (az < a1)
            t = (az - a0) / (a1 - a0) if a1 > a0 else 0.0
        else:
            inside = (az >= a0) or (az < a1)
            t = 0.0

        if not inside:
            continue

        em = seg.get("elev_min_deg", None)
        if em is None:
            return None
        if isinstance(em, dict) and "lin" in em:
            e0, e1 = em["lin"]
            return float(e0) + t * (float(e1) - float(e0))
        return float(em)
    return None


def _perez_poa_components(
    *,
    dhi: float,
    dni: float,
    i0n: float,
    zenith_deg: float,
    sun_az_deg: float,
    sun_elev_deg: float,
    tilt_deg: float,
    surf_az_deg: float,
    albedo: float,
    horizon: Optional[List[Dict[str, Any]]],
    horizon_block_elsewhere: bool,
) -> Tuple[float, float, float, float]:
    z = float(zenith_deg)
    cosz = math.cos(math.radians(z))
    if cosz <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    beta = math.radians(float(tilt_deg))
    az_sun = math.radians(float(sun_az_deg))
    az_surf = math.radians(float(surf_az_deg))
    elev = float(sun_elev_deg)

    cos_theta = (
        math.sin(math.radians(elev)) * math.cos(beta)
        + math.cos(math.radians(elev)) * math.sin(beta) * math.cos(az_sun - az_surf)
    )
    cos_theta = max(cos_theta, 0.0)

    cos_theta_beam = cos_theta
    if horizon:
        elev_min = _horizon_elev_min_for_az(sun_az_deg, horizon)
        if elev_min is None:
            if horizon_block_elsewhere:
                cos_theta_beam = 0.0
        else:
            if elev < elev_min:
                cos_theta_beam = 0.0

    poa_beam = float(dni) * cos_theta_beam

    z_rad = math.radians(z)
    if dhi <= 1e-6:
        epsilon = 999.0
        delta = 0.0
    else:
        epsilon = ((dhi + dni) / dhi + 1.041 * (z_rad**3)) / (1.0 + 1.041 * (z_rad**3))
        am = _kasten_young_airmass(z)
        delta = (dhi * am / max(i0n, 1e-6)) if math.isfinite(am) else 0.0

    bin_idx = 0
    for b in _EPS_BINS:
        if epsilon < b:
            break
        bin_idx += 1
    bin_idx = min(bin_idx, 7)
    f11, f12, f13, f21, f22, f23 = _PEREZ_COEFFS[bin_idx]

    f1 = f11 + f12 * delta + f13 * z_rad
    f2 = f21 + f22 * delta + f23 * z_rad
    f1 = max(0.0, min(1.0, f1))

    a = max(cos_theta_beam, 0.0)
    b = max(cosz, 0.087)
    cos_ratio = a / b

    poa_diffuse_sky = dhi * (
        (1.0 - f1) * (1.0 + math.cos(beta)) / 2.0
        + f1 * cos_ratio
        + f2 * math.sin(beta)
    )
    poa_diffuse_sky = max(poa_diffuse_sky, 0.0)

    ghi = dhi + dni * cosz
    poa_ground = albedo * ghi * (1.0 - math.cos(beta)) / 2.0
    poa_ground = max(poa_ground, 0.0)

    return float(poa_beam), float(poa_diffuse_sky), float(poa_ground), float(cos_theta)


def _normalize_shutter(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if np.nanpercentile(values, 95) > 1.5:
        values = values / 100.0
    return np.clip(values, 0.0, 1.0)


def precompute_windows(
    df: pd.DataFrame,
    windows: List[Dict[str, Any]],
    *,
    time_col: str,
    ghi_col: str,
    sun_az_col: str,
    sun_elev_col: str,
    albedo: float,
) -> Tuple[List[WindowPrecomp], Dict[str, np.ndarray]]:
    t = pd.to_datetime(df[time_col], utc=True)
    doy = t.dt.dayofyear.to_numpy()
    sun_elev = pd.to_numeric(df[sun_elev_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    zen = 90.0 - sun_elev
    sun_az = pd.to_numeric(df[sun_az_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ghi = pd.to_numeric(df[ghi_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    n = len(df)
    dni = np.zeros(n, dtype=float)
    dhi = np.zeros(n, dtype=float)
    i0n = np.zeros(n, dtype=float)

    for d in np.unique(doy):
        idx = doy == d
        for i in np.where(idx)[0]:
            dni[i], dhi[i], i0n[i] = _erbs_dni_dhi(float(ghi[i]), float(zen[i]), int(d))

    pre: List[WindowPrecomp] = []
    for w in windows:
        name = str(w.get("name", "window"))
        area = float(w.get("area_m2", 0.0))
        gval = float(w.get("g_value", 0.6))
        area_g = area * gval

        tilt = float(w.get("tilt_deg", 90.0))
        surf_az = float(w.get("azimuth_deg", 180.0))
        horizon = w.get("horizon", None)
        horizon_block_elsewhere = bool(w.get("horizon_block_elsewhere", False))

        shutter_feature = w.get("shutter_feature", None)
        shutter_pos = np.zeros(n, dtype=float)
        if shutter_feature:
            candidates = [str(shutter_feature), "Rolluik woonkamer", "RollerShutter_Livingroom"]
            col = next((c for c in candidates if c in df.columns), None)
            if col is not None:
                shutter_pos = _normalize_shutter(df[col])

        pb = np.zeros(n, dtype=float)
        pdif = np.zeros(n, dtype=float)
        ctheta = np.zeros(n, dtype=float)

        for i in range(n):
            if ghi[i] <= 0.0 or sun_elev[i] <= 0.0:
                continue
            b, d, gnd, ct = _perez_poa_components(
                dhi=float(dhi[i]),
                dni=float(dni[i]),
                i0n=float(i0n[i]),
                zenith_deg=float(zen[i]),
                sun_az_deg=float(sun_az[i]),
                sun_elev_deg=float(sun_elev[i]),
                tilt_deg=tilt,
                surf_az_deg=surf_az,
                albedo=float(albedo),
                horizon=horizon if isinstance(horizon, list) else None,
                horizon_block_elsewhere=horizon_block_elsewhere,
            )
            pb[i] = b
            pdif[i] = d + gnd
            ctheta[i] = ct

        pre.append(
            WindowPrecomp(
                name=name,
                area_g=area_g,
                pb=pb,
                pd=pdif,
                cos_theta=ctheta,
                shutter_pos=shutter_pos,
            )
        )

    dbg = {"dni": dni, "dhi": dhi, "ghi": ghi, "sun_elev": sun_elev, "sun_az": sun_az, "doy": doy}
    return pre, dbg


def compute_q_trans(
    pre: List[WindowPrecomp],
    *,
    b0_iam: float,
    alpha_shutter: float,
) -> np.ndarray:
    if not pre:
        return np.zeros(0, dtype=float)

    iam_diff = float(_iam_ashrae(np.array([math.cos(math.radians(59.0))]), b0=float(b0_iam))[0])

    q = np.zeros_like(pre[0].pb, dtype=float)
    for w in pre:
        iam_beam = _iam_ashrae(w.cos_theta, b0=float(b0_iam))
        sh_factor = 1.0 - float(alpha_shutter) * (1.0 - w.shutter_pos)
        sh_factor = np.clip(sh_factor, 0.0, 1.0)
        q += w.area_g * (w.pb * iam_beam + w.pd * iam_diff) * sh_factor
    return q


def solar_lag(q_trans: np.ndarray, *, dt_s: float, tau_s: float) -> np.ndarray:
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


def compute_solar_inputs(
    df: pd.DataFrame,
    windows: List[Dict[str, Any]],
    params: SolarParams,
    *,
    time_col: str,
    ghi_col: str,
    sun_az_col: str,
    sun_elev_col: str,
    albedo: float,
    dt_s: float,
) -> SolarInputs:
    pre, _dbg = precompute_windows(
        df,
        windows,
        time_col=time_col,
        ghi_col=ghi_col,
        sun_az_col=sun_az_col,
        sun_elev_col=sun_elev_col,
        albedo=albedo,
    )
    q_trans = compute_q_trans(pre, b0_iam=params.b0_iam, alpha_shutter=params.alpha_shutter)
    q_filt = solar_lag(q_trans, dt_s=dt_s, tau_s=params.tau_solar_s)

    q_solar = params.k_gain * q_filt
    qi = params.f_air * q_solar + float(params.q_bias_W)
    qm = (1.0 - params.f_air) * q_solar
    return SolarInputs(q_trans=q_trans, q_filt=q_filt, qi=qi, qm=qm)


def read_solar_params(path: Path) -> SolarParams:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SolarParams(
        k_gain=float(data["k_gain"]),
        f_air=float(data["f_air"]),
        tau_solar_s=float(data["tau_solar_s"]),
        alpha_shutter=float(data["alpha_shutter"]),
        b0_iam=float(data["b0_iam"]),
        q_bias_W=float(data.get("q_bias_W", 0.0)),
    )


def summarize_windows(windows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for w in windows:
        summary.append(
            {
                "name": w.get("name"),
                "area_m2": w.get("area_m2"),
                "g_value": w.get("g_value"),
                "tilt_deg": w.get("tilt_deg"),
                "azimuth_deg": w.get("azimuth_deg"),
                "shutter_feature": w.get("shutter_feature"),
                "horizon": w.get("horizon"),
            }
        )
    return summary
