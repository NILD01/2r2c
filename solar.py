#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solar.py — Fit zonnewinsten (solar gains) op binnentemperatuur met vaste 2R2C parameters.

Doel
----
Je hebt al een 2R2C (Ti/Tm) model gefit (Ria, Rao, Ci, Cm) op segmenten zonder externe warmtebronnen.
In dit script voegen we een zonne-input toe (op basis van GHI + zonpositie + raam-geometrie + horizon/shading),
en fitten we een *effectieve* koppeling van zon → warmte-inbreng zodat full-forward Tin goed klopt.

Waarom "effectief"?
-------------------
Omdat jouw 2R2C-fit Ci/Cm hier (bewust) vaststaan, en Ci in jouw r2r2c_result.json typisch de luchtcapaciteit
benadert (Ci_from_volume). In realiteit gaat een groot deel van zonnestraling eerst in oppervlakken (vloer, muren,
meubels) en wordt pas later (en gedempt) zichtbaar in Tin. Daarom fitten we een schaalfactor (k_gain) en optionele
parameters (split naar air/mass, shutter-factor, IAM) om de zonne-input realistisch te projecteren op jouw 2R2C.

State-of-the-art keuzes (pragmatisch, robuust)
----------------------------------------------
1) GHI → DNI/DHI via Erbs (met clearness index Kt, extraterrestrial irradiance) (Erbs 1982).
2) Transpositie naar vlak van het raam via Perez 1990 diffuse model (sky diffuse) + ground-reflected.
3) Beam + circumsolar shading via horizon-profiel (azimuth-elev_min) per raam.
4) Angle-of-incidence modifier (IAM) via ASHRAE-form (b0 parameter).
5) Fit gebeurt full-forward, segment-per-segment, met analytische LS-oplossing voor Tm0 (warm-up),
   en globale optimalisatie voor solar parameters (L-BFGS-B).

Input
-----
Van Node-RED pythonCall("solar", data, msg, node) verwacht dit script typisch:
{
  "name": "3R2C_test",
  "rootDir": "/home/.../scripts",
  "subProject": "solar",
  "tslowSubProject": "r2r2c",             # of: "r2r2cSubProject"
  "windows": [ ... ]                     # raamdefs zoals in je function node
}

Bestanden
---------
- segments_data.csv:    <project_root>/<subProject>/segments_data.csv
- r2r2c_result.json:    <project_root>/<r2r2cSubProject>/r2r2c_result.json  (of *_result.json)

Output
------
- solar_result.json:        <project_root>/<subProject>/solar_result.json
- solar_predictions.csv:    <project_root>/<subProject>/solar_predictions.csv

Opmerking: dit script maakt géén aanspraak op perfecte fysica tot op SHGC-niveau;
het is bedoeld als zeer accurate en reproduceerbare *grey-box* solar fit op jouw data.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────
# Utilities / IO
# ─────────────────────────────────────────────────────────────

def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _resolve_project_root(root_dir: str, name: str) -> Path:
    """
    rootDir kan zijn:
      - /home/.../scripts
      - /home/.../scripts/<name>
    We willen altijd: /home/.../scripts/<name>
    """
    base = Path(root_dir)
    return base if base.name == name else (base / name)


def _first_existing(candidates: Iterable[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    # fallback: return first (for better error message)
    cand = list(candidates)
    if not cand:
        raise FileNotFoundError("No candidates provided")
    raise FileNotFoundError(str(cand[0]))


def _read_segments_csv(csv_path: Path, *, time_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if time_col not in df.columns:
        raise KeyError(f"Missing '{time_col}' in {csv_path}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# Solar decomposition + transposition (Erbs + Perez)
# ─────────────────────────────────────────────────────────────

def _solar_extraterrestrial_normal(doy: int) -> float:
    """
    Extraterrestrial normal irradiance I0n [W/m²] (approx).
    """
    I_sc = 1367.0
    gamma = 2.0 * math.pi * (doy - 1) / 365.0
    E0 = (
        1.00011
        + 0.034221 * math.cos(gamma)
        + 0.00128 * math.sin(gamma)
        + 0.000719 * math.cos(2 * gamma)
        + 0.000077 * math.sin(2 * gamma)
    )
    return I_sc * E0


def _kasten_young_airmass(zenith_deg: float) -> float:
    """
    Relative airmass (Kasten & Young).
    """
    z = float(zenith_deg)
    if z >= 90.0:
        return float("nan")
    cosz = math.cos(math.radians(z))
    return 1.0 / (cosz + 0.50572 * ((96.07995 - z) ** (-1.6364)))


def _erbs_dni_dhi(ghi: float, zenith_deg: float, doy: int) -> Tuple[float, float, float]:
    """
    Estimate DNI and DHI from GHI using Erbs correlation.
    Returns: dni, dhi, I0n
    """
    ghi = float(ghi)
    z = float(zenith_deg)
    cosz = math.cos(math.radians(z))
    I0n = _solar_extraterrestrial_normal(doy)
    I0h = max(I0n * cosz, 1e-6)

    if ghi <= 0.0 or cosz <= 0.065:
        return 0.0, 0.0, I0n

    Kt = max(0.0, min(2.0, ghi / I0h))

    if Kt <= 0.22:
        DF = 1.0 - 0.09 * Kt
    elif Kt <= 0.8:
        DF = 0.9511 - 0.1604 * Kt + 4.388 * Kt**2 - 16.638 * Kt**3 + 12.336 * Kt**4
    else:
        DF = 0.165

    DF = max(0.0, min(1.0, DF))
    dhi = DF * ghi
    dni = (ghi - dhi) / max(cosz, 1e-6)
    dni = max(dni, 0.0)
    dhi = max(dhi, 0.0)
    return dni, dhi, I0n


# Perez 1990 coefficients (8 bins).
# (Deze tabel komt overeen met de veelgebruikte implementaties in pvlib/PVsyst-achtige transpositie.)
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


def _iam_ashrae(cos_theta: np.ndarray, b0: float) -> np.ndarray:
    """
    ASHRAE-like IAM: IAM = 1 - b0 * (1/cos - 1)
    b0 in [0..0.2] typical.
    """
    ct = np.clip(cos_theta, 1e-6, 1.0)
    iam = 1.0 - b0 * (1.0 / ct - 1.0)
    return np.clip(iam, 0.0, 1.0)


def _horizon_elev_min_for_az(
    az_deg: float,
    horizon: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Zoek de elev_min_deg voor een gegeven azimuth in de horizon-segmentlijst.
    Segment kan elev_min_deg als float hebben, of {"lin":[e0,e1]} voor lineair over az-range.
    Returns None als az niet in horizon-segmenten valt.
    """
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
            # wrap-around (we nemen t=0 als eenvoudige fallback)
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
    I0n: float,
    zenith_deg: float,
    sun_az_deg: float,
    sun_elev_deg: float,
    tilt_deg: float,
    surf_az_deg: float,
    albedo: float,
    horizon: Optional[List[Dict[str, Any]]],
    horizon_block_elsewhere: bool,
) -> Tuple[float, float, float, float]:
    """
    Compute POA components for ONE timestep and ONE surface:
      returns: (poa_beam, poa_diffuse_sky, poa_ground, cos_theta)

    Beam & circumsolar shading:
      if horizon blocks sun for this az/elev (or if az outside horizon and horizon_block_elsewhere=True),
      we set beam and circumsolar part to 0 via cos_theta_beam=0.
    """
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

    # Horizon shading on beam (+ circumsolar)
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

    # Perez diffuse sky
    z_rad = math.radians(z)
    if dhi <= 1e-6:
        epsilon = 999.0
        delta = 0.0
    else:
        epsilon = ((dhi + dni) / dhi + 1.041 * (z_rad**3)) / (1.0 + 1.041 * (z_rad**3))
        am = _kasten_young_airmass(z)
        delta = (dhi * am / max(I0n, 1e-6)) if math.isfinite(am) else 0.0

    # bin selection
    bin_idx = 0
    for b in _EPS_BINS:
        if epsilon < b:
            break
        bin_idx += 1
    bin_idx = min(bin_idx, 7)
    F11, F12, F13, F21, F22, F23 = _PEREZ_COEFFS[bin_idx]

    F1 = F11 + F12 * delta + F13 * z_rad
    F2 = F21 + F22 * delta + F23 * z_rad
    F1 = max(0.0, min(1.0, F1))

    # circumsolar term uses cos_theta_beam/cosz
    a = max(cos_theta_beam, 0.0)
    b = max(cosz, 0.087)  # common numeric stabilization
    cos_ratio = a / b

    poa_diffuse_sky = dhi * (
        (1.0 - F1) * (1.0 + math.cos(beta)) / 2.0
        + F1 * cos_ratio
        + F2 * math.sin(beta)
    )
    poa_diffuse_sky = max(poa_diffuse_sky, 0.0)

    # ground-reflected
    ghi = dhi + dni * cosz
    poa_ground = albedo * ghi * (1.0 - math.cos(beta)) / 2.0
    poa_ground = max(poa_ground, 0.0)

    return float(poa_beam), float(poa_diffuse_sky), float(poa_ground), float(cos_theta)


# ─────────────────────────────────────────────────────────────
# 2R2C discretization + simulation
# ─────────────────────────────────────────────────────────────

def _discretize_2r2c(Ria: float, Rao: float, Ci: float, Cm: float, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous model:
      Ci dTi/dt = (Tout - Ti)/Rao + (Tm - Ti)/Ria + Qi
      Cm dTm/dt = (Ti - Tm)/Ria + Qm

    Discretization via exact matrix exponential with input integral.
    Inputs u = [Tout, Qi, Qm]
    """
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

    n, m = 2, 3
    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * float(dt_s))
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def _simulate_segment_with_tm0_ls(
    *,
    Ti_obs: np.ndarray,
    Tout: np.ndarray,
    q_trans: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    dt_s: float,
    k_gain: float,
    f_air: float,
    tau_solar_s: float,
    q_bias_W: float,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Full-forward simulation for one segment.
    Ti0 is fixed to observed Ti_obs[0].
    Tm0 is chosen by closed-form least squares (linear in Tm0).

    Returns: Ti_pred, Tm_pred, tm0, qf_used
    """
    n = int(len(Ti_obs))
    if n <= 1:
        return Ti_obs.copy(), np.full_like(Ti_obs, np.nan), float("nan"), q_trans.copy()

    # optional low-pass filter for solar feature inside the segment
    if tau_solar_s <= 1e-9:
        qf = q_trans.astype(float, copy=True)
    else:
        alpha = float(dt_s) / (float(tau_solar_s) + float(dt_s))
        qf = np.empty_like(q_trans, dtype=float)
        qf[0] = float(q_trans[0])
        for k in range(1, n):
            qf[k] = qf[k - 1] + alpha * (float(q_trans[k]) - qf[k - 1])

    Qi = f_air * k_gain * qf + q_bias_W
    Qm = (1.0 - f_air) * k_gain * qf

    Ti0 = float(Ti_obs[0])

    def _sim(tm0: float) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array([Ti0, float(tm0)], dtype=float)
        Ti_pred = np.empty(n, dtype=float)
        Tm_pred = np.empty(n, dtype=float)
        Ti_pred[0] = Ti0
        Tm_pred[0] = float(tm0)
        for k in range(n - 1):
            u = np.array([float(Tout[k]), float(Qi[k]), float(Qm[k])], dtype=float)
            x = Ad.dot(x) + Bd.dot(u)
            Ti_pred[k + 1] = x[0]
            Tm_pred[k + 1] = x[1]
        return Ti_pred, Tm_pred

    Ti_base, _ = _sim(0.0)
    Ti_one, _ = _sim(1.0)
    coeff = Ti_one - Ti_base
    y = Ti_obs - Ti_base

    # solve tm0 = argmin || y - coeff*tm0 ||² (excluding k=0 where coeff=0)
    num = float(np.dot(coeff[1:], y[1:]))
    den = float(np.dot(coeff[1:], coeff[1:])) + 1e-12
    tm0 = num / den
    tm0 = float(np.clip(tm0, Ti0 - 10.0, Ti0 + 10.0))

    Ti_pred, Tm_pred = _sim(tm0)
    return Ti_pred, Tm_pred, tm0, qf


def _rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return rmse, r2


# ─────────────────────────────────────────────────────────────
# Solar feature precomputation (per window)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WindowPrecomp:
    name: str
    area_g: float              # area * g_value
    pb: np.ndarray             # POA beam (before IAM)
    pd: np.ndarray             # POA diffuse+ground (before IAM)
    cos_theta: np.ndarray      # cos(incidence)
    shutter_pos: np.ndarray    # 0..1


def _normalize_shutter(series: pd.Series) -> np.ndarray:
    """
    Robust mapping van shutter-feature naar 0..1.
    Ondersteunt:
      - 0..100 (percentage)
      - 0..1
    """
    v = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # heuristic: if values mostly >1, assume percentage
    if np.nanpercentile(v, 95) > 1.5:
        v = v / 100.0
    return np.clip(v, 0.0, 1.0)


def _precompute_windows(
    df: pd.DataFrame,
    windows: List[Dict[str, Any]],
    *,
    time_col: str,
    ghi_col: str,
    sun_az_col: str,
    sun_elev_col: str,
    albedo: float,
) -> Tuple[List[WindowPrecomp], Dict[str, np.ndarray]]:
    """
    Precompute per window:
      - POA beam (pb), POA diffuse+ground (pd), cos(theta), shutter_pos

    Returns also solar decomposition arrays for debugging (dni,dhi).
    """
    t = pd.to_datetime(df[time_col], utc=True)
    doy = t.dt.dayofyear.to_numpy()
    sun_elev = pd.to_numeric(df[sun_elev_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    zen = 90.0 - sun_elev
    sun_az = pd.to_numeric(df[sun_az_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ghi = pd.to_numeric(df[ghi_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    n = len(df)
    dni = np.zeros(n, dtype=float)
    dhi = np.zeros(n, dtype=float)
    I0n = np.zeros(n, dtype=float)

    # compute Erbs per day (vectorized enough)
    for d in np.unique(doy):
        idx = doy == d
        for i in np.where(idx)[0]:
            dni[i], dhi[i], I0n[i] = _erbs_dni_dhi(float(ghi[i]), float(zen[i]), int(d))

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

        # shutter position (optional)
        shutter_feature = w.get("shutter_feature", None)
        shutter_pos = np.zeros(n, dtype=float)
        if shutter_feature:
            # allow some practical fallbacks
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
                I0n=float(I0n[i]),
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

        pre.append(WindowPrecomp(name=name, area_g=area_g, pb=pb, pd=pdif, cos_theta=ctheta, shutter_pos=shutter_pos))

    dbg = {"dni": dni, "dhi": dhi, "ghi": ghi, "sun_elev": sun_elev, "sun_az": sun_az, "doy": doy}
    return pre, dbg


def _compute_q_trans(
    pre: List[WindowPrecomp],
    *,
    b0_iam: float,
    alpha_shutter: float,
) -> np.ndarray:
    """
    Compute effective transmitted solar "power proxy" [W]:
      Q_trans = sum(area*g * (pb*IAM_beam + pd*IAM_diff) * shutter_factor)
    """
    if not pre:
        return np.zeros(0, dtype=float)

    # diffuse IAM at a representative angle (≈59° is common)
    iam_diff = float(_iam_ashrae(np.array([math.cos(math.radians(59.0))]), b0=float(b0_iam))[0])

    q = np.zeros_like(pre[0].pb, dtype=float)
    for w in pre:
        iam_beam = _iam_ashrae(w.cos_theta, b0=float(b0_iam))
        ##sh_factor = 1.0 - float(alpha_shutter) * w.shutter_pos
        sh_factor = 1.0 - float(alpha_shutter) * (1.0 - w.shutter_pos)
        sh_factor = np.clip(sh_factor, 0.0, 1.0)

        q += w.area_g * (w.pb * iam_beam + w.pd * iam_diff) * sh_factor
    return q


# ─────────────────────────────────────────────────────────────
# Fit loop
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitParams:
    k_gain: float
    f_air: float
    tau_solar_s: float
    alpha_shutter: float
    b0_iam: float
    q_bias_W: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _unpack_theta(theta: np.ndarray) -> FitParams:
    """
    Optimization vector in unconstrained space → physical parameters.
      theta = [log(k_gain), logit(f_air), log(tau), logit(alpha_shutter), logit(b0/0.2), q_bias]
    """
    log_k, logit_f, log_tau, logit_alpha, logit_b0, q_bias = [float(v) for v in theta]
    k_gain = float(math.exp(log_k))
    f_air = float(_sigmoid(logit_f))
    tau_s = float(math.exp(log_tau))
    alpha_sh = float(_sigmoid(logit_alpha))
    b0 = float(0.2 * _sigmoid(logit_b0))  # 0..0.2 typical
    return FitParams(k_gain=k_gain, f_air=f_air, tau_solar_s=tau_s, alpha_shutter=alpha_sh, b0_iam=b0, q_bias_W=float(q_bias))


def _fit_solar(
    *,
    df: pd.DataFrame,
    pre: List[WindowPrecomp],
    r2: Dict[str, Any],
    columns: Dict[str, str],
    out_dir: Path,
    fit_seed: int = 42,
    val_fraction: float = 0.2,
) -> Dict[str, Any]:
    # extract RC
    fit = r2["fit"]
    Ria = float(fit["Ria_K_per_W"])
    Rao = float(fit["Rao_K_per_W"])
    Ci = float(fit["Ci_J_per_K"])
    Cm = float(fit["Cm_J_per_K"])
    dt_s = float(r2.get("inputs", {}).get("dt_seconds", 60.0))

    seg_col = columns["segment_col"]
    time_col = columns["time_col"]
    tin_col = columns["tin_col"]
    tout_col = columns["tout_col"]

    # group segments
    df = df.sort_values([seg_col, time_col]).reset_index(drop=True)

    seg_ids = df[seg_col].dropna().unique().tolist()
    seg_ids = [int(s) for s in seg_ids]

    rng = np.random.default_rng(int(fit_seed))
    perm = rng.permutation(seg_ids)
    n_val = max(1, int(round(val_fraction * len(perm)))) if len(perm) >= 5 else 1
    val_set = set(int(x) for x in perm[:n_val])
    train_set = set(int(x) for x in perm[n_val:]) if len(perm) > n_val else set(int(x) for x in perm)

    # cached indices per segment
    segments: List[Dict[str, Any]] = []
    for sid, g in df.groupby(seg_col, sort=False):
        sid_i = int(sid)
        Tin = pd.to_numeric(g[tin_col], errors="coerce").to_numpy(dtype=float)
        Tout = pd.to_numeric(g[tout_col], errors="coerce").to_numpy(dtype=float)
        pos = g.index.to_numpy(dtype=int)
        m = np.isfinite(Tin) & np.isfinite(Tout)
        Tin = Tin[m]
        Tout = Tout[m]
        pos = pos[m]
        if len(Tin) >= 10:
            segments.append({"sid": sid_i, "pos": pos, "Tin": Tin, "Tout": Tout})

    Ad, Bd = _discretize_2r2c(Ria, Rao, Ci, Cm, dt_s)

    # objective (train only)
    def _eval_on(theta: np.ndarray, sids: set[int]) -> Tuple[float, float]:
        p = _unpack_theta(theta)
        q_trans_all = _compute_q_trans(pre, b0_iam=p.b0_iam, alpha_shutter=p.alpha_shutter)

        ys: List[np.ndarray] = []
        ps: List[np.ndarray] = []
        for seg in segments:
            if seg["sid"] not in sids:
                continue
            pos = seg["pos"]
            Ti_obs = seg["Tin"]
            Tout = seg["Tout"]
            qt = q_trans_all[pos]
            Ti_pred, _, _, _ = _simulate_segment_with_tm0_ls(
                Ti_obs=Ti_obs,
                Tout=Tout,
                q_trans=qt,
                Ad=Ad,
                Bd=Bd,
                dt_s=dt_s,
                k_gain=p.k_gain,
                f_air=p.f_air,
                tau_solar_s=p.tau_solar_s,
                q_bias_W=p.q_bias_W,
            )
            ys.append(Ti_obs)
            ps.append(Ti_pred)

        y = np.concatenate(ys) if ys else np.zeros(0, dtype=float)
        pr = np.concatenate(ps) if ps else np.zeros(0, dtype=float)
        return _rmse_r2(y, pr)

    def _objective(theta: np.ndarray) -> float:
        rmse, _ = _eval_on(theta, train_set)

        # kleine regularisatie om extreme waarden te vermijden (maar fit blijft gedreven door data)
        p = _unpack_theta(theta)
        reg = 0.0
        reg += 1e-6 * (p.q_bias_W ** 2)
        reg += 1e-4 * ((p.b0_iam - 0.10) / 0.05) ** 2
        reg += 1e-4 * (math.log(max(p.tau_solar_s, 1e-9) / 600.0) ** 2)  # zachte voorkeur rond 10 min
        return float(rmse + reg)

    # init: kleine effectieve solar gain (Ci is klein), f_air ~ 0.1, tau ~ 30 min
    theta0 = np.array(
        [
            math.log(0.01),     # log(k_gain)
            -2.0,               # logit(f_air)  (~0.12)
            math.log(1800.0),   # log(tau_solar_s)
            2.0,                # logit(alpha_shutter) (~0.88)
            0.0,                # logit(b0/0.2) -> b0 ~0.1
            0.0,                # q_bias_W
        ],
        dtype=float,
    )

    res = minimize(_objective, theta0, method="L-BFGS-B", options={"maxiter": 80})

    p_best = _unpack_theta(res.x)

    # metrics
    rmse_train, r2_train = _eval_on(res.x, train_set)
    rmse_val, r2_val = _eval_on(res.x, val_set)
    rmse_all, r2_all = _eval_on(res.x, set([s["sid"] for s in segments]))

    # predictions (all)
    q_trans_all = _compute_q_trans(pre, b0_iam=p_best.b0_iam, alpha_shutter=p_best.alpha_shutter)

    rows: List[Dict[str, Any]] = []
    tm0_map: Dict[int, float] = {}
    for seg in segments:
        sid = int(seg["sid"])
        pos = seg["pos"]
        Ti_obs = seg["Tin"]
        Tout = seg["Tout"]
        qt = q_trans_all[pos]

        Ti_pred, Tm_pred, tm0, qf = _simulate_segment_with_tm0_ls(
            Ti_obs=Ti_obs,
            Tout=Tout,
            q_trans=qt,
            Ad=Ad,
            Bd=Bd,
            dt_s=dt_s,
            k_gain=p_best.k_gain,
            f_air=p_best.f_air,
            tau_solar_s=p_best.tau_solar_s,
            q_bias_W=p_best.q_bias_W,
        )
        tm0_map[sid] = float(tm0)

        # times for this segment (from df, using pos indices)
        seg_times = df.loc[pos, time_col].to_list()
        for i in range(len(Ti_obs)):
            rows.append(
                {
                    "segment_id": sid,
                    "time": seg_times[i],
                    "Tin_obs_C": float(Ti_obs[i]),
                    "Tin_pred_C": float(Ti_pred[i]),
                    "Tm_pred_C": float(Tm_pred[i]) if np.isfinite(Tm_pred[i]) else None,
                    "Tout_C": float(Tout[i]),
                    "Qtrans_proxy_W": float(qt[i]),
                    "Qtrans_filt_W": float(qf[i]),
                }
            )

    pred_df = pd.DataFrame(rows)
    pred_csv = out_dir / "solar_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    out_json = out_dir / "solar_result.json"
    result: Dict[str, Any] = {
        "name": r2.get("name"),
        "subProject": "solar",
        "sources": {
            "segments_csv": str(out_dir / "segments_data.csv"),
            "r2r2c_result_json": str(out_dir.parent / str(r2.get("subProject", "r2r2c")) / "r2r2c_result.json"),
        },
        "inputs": {
            "columns": columns,
            "albedo": 0.2,
            "val_fraction": float(val_fraction),
            "fit_seed": int(fit_seed),
            "dt_seconds": float(dt_s),
        },
        "solar_fit": {
            "params": {
                "k_gain": p_best.k_gain,
                "f_air": p_best.f_air,
                "tau_solar_s": p_best.tau_solar_s,
                "alpha_shutter": p_best.alpha_shutter,
                "b0_iam": p_best.b0_iam,
                "q_bias_W": p_best.q_bias_W,
            },
            "tm0_per_segment": tm0_map,
            "metrics_full_forward": {
                "rmse_train_C": rmse_train,
                "r2_train": r2_train,
                "rmse_val_C": rmse_val,
                "r2_val": r2_val,
                "rmse_all_C": rmse_all,
                "r2_all": r2_all,
            },
            "optimizer": {
                "method": "L-BFGS-B",
                "success": bool(res.success),
                "status": int(res.status),
                "message": str(res.message),
                "nit": int(getattr(res, "nit", -1)),
                "fun": float(res.fun),
            },
        },
        "paths": {
            "out_json": str(out_json),
            "predictions_csv": str(pred_csv),
        },
    }
    _write_json(out_json, result)
    return result


# ─────────────────────────────────────────────────────────────
# Public entrypoint (Node-RED)
# ─────────────────────────────────────────────────────────────

def run_solar(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry-point voor pythonCall("solar", ...).

    Verwachte keys:
      - name, rootDir, subProject
      - windows (list)
      - r2r2c subproject key: tslowSubProject / r2r2cSubProject / r2r2csubSubProject / r2r2cSubProject
    """
    name = str(data["name"])
    root_dir = str(data["rootDir"])
    subproject = str(data.get("subProject", "solar"))

    r2_sub = (
        data.get("tslowSubProject")
        or data.get("r2r2cSubProject")
        or data.get("r2r2csubSubProject")
        or "r2r2c"
    )
    r2_sub = str(r2_sub)

    windows = data.get("windows", [])
    if not isinstance(windows, list) or len(windows) == 0:
        raise ValueError("No windows provided (data['windows'] is empty).")

    proj = _resolve_project_root(root_dir, name)
    out_dir = proj / subproject
    out_dir.mkdir(parents=True, exist_ok=True)

    # r2r2c result candidates
    r2_candidates = [
        proj / r2_sub / "r2r2c_result.json",
        proj / r2_sub / f"{r2_sub}_result.json",
        proj / "r2r2c" / "r2r2c_result.json",
        proj / "r2r2c" / "r2r2c_result.json",
    ]
    r2_path = _first_existing(r2_candidates)
    r2 = _load_json(r2_path)

    # columns from r2 result (fallback to defaults)
    cols = r2.get("inputs", {}).get("columns", {}) or {}
    columns = {
        "segment_col": str(cols.get("segment_col", "segment_id")),
        "time_col": str(cols.get("time_col", "time")),
        "tin_col": str(cols.get("tin_col", "binnentemperatuur")),
        "tout_col": str(cols.get("tout_col", "buitentemperatuur")),
    }

    # solar-specific columns (your segments_data.csv uses these names)
    ghi_col = "zonnestraling"
    sun_az_col = "zonazimut"
    sun_elev_col = "zonhoogte"
    for c in [ghi_col, sun_az_col, sun_elev_col]:
        # fail early with a clear message if missing
        pass

    seg_csv = proj / subproject / "segments_data.csv"
    if not seg_csv.exists():
        raise FileNotFoundError(str(seg_csv))

    df = _read_segments_csv(seg_csv, time_col=columns["time_col"])

    missing = [c for c in [columns["segment_col"], columns["tin_col"], columns["tout_col"], ghi_col, sun_az_col, sun_elev_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {seg_csv}: {missing}")

    # precompute solar feature
    pre, dbg = _precompute_windows(
        df,
        windows,
        time_col=columns["time_col"],
        ghi_col=ghi_col,
        sun_az_col=sun_az_col,
        sun_elev_col=sun_elev_col,
        albedo=0.2,
    )

    # fit
    result = _fit_solar(
        df=df,
        pre=pre,
        r2=r2,
        columns=columns,
        out_dir=out_dir,
        fit_seed=42,
        val_fraction=0.2,
    )

    # add some debug summaries
    result["solar_debug"] = {
        "precomputed_windows": [w.name for w in pre],
        "ghi_stats_Wm2": {
            "min": float(np.min(dbg["ghi"])),
            "p50": float(np.percentile(dbg["ghi"], 50)),
            "p95": float(np.percentile(dbg["ghi"], 95)),
            "max": float(np.max(dbg["ghi"])),
        },
    }
    _write_json(Path(result["paths"]["out_json"]), result)
    return result


# Backwards-compatible alias (some older flows might call run_r2r2c)
def run_r2r2c(data: Dict[str, Any]) -> Dict[str, Any]:
    return run_solar(data)


def _main() -> None:
    """
    CLI mode:
      echo '{"name":"...","rootDir":"...","subProject":"solar","tslowSubProject":"r2r2c","windows":[...]}' | ./solar.py
    """
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "No JSON received on stdin"}, ensure_ascii=False))
        sys.exit(2)
    try:
        data = json.loads(raw)
        result = run_solar(data)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    _main()
