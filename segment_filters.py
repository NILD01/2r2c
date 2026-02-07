from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterSpec:
    time_col: str
    tin_col: str
    tout_col: str
    require_cv_off: bool = False
    require_hp_off: bool = False
    require_nightmode_sleep: bool = False
    require_cooling: bool = False
    require_tout_below_tin: bool = False
    night_start: int | None = None
    night_end: int | None = None
    max_solar: float | None = None
    min_segment_len: int | None = None
    min_tin_range: float | None = None
    min_tout_range: float | None = None


def prepare_dataframe(df: pd.DataFrame, *, spec: FilterSpec, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    required = [spec.time_col, spec.tin_col, spec.tout_col]
    if spec.require_cv_off:
        required.append("CV_mode_off")
    if spec.require_hp_off:
        required.append("warmtepomp_mode_off")
    if spec.require_nightmode_sleep:
        required.append("nachtmodus_slapen")
    if spec.max_solar is not None:
        required.append("zonnestraling")
    required.extend(extra_cols)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    df = df[required].copy()
    df[spec.time_col] = pd.to_datetime(df[spec.time_col], utc=True, errors="coerce")
    df[spec.tin_col] = pd.to_numeric(df[spec.tin_col], errors="coerce")
    df[spec.tout_col] = pd.to_numeric(df[spec.tout_col], errors="coerce")
    if spec.require_cv_off:
        df["CV_mode_off"] = pd.to_numeric(df["CV_mode_off"], errors="coerce")
    if spec.require_hp_off:
        df["warmtepomp_mode_off"] = pd.to_numeric(df["warmtepomp_mode_off"], errors="coerce")
    if spec.require_nightmode_sleep:
        df["nachtmodus_slapen"] = pd.to_numeric(df["nachtmodus_slapen"], errors="coerce")
    if spec.max_solar is not None:
        df["zonnestraling"] = pd.to_numeric(df["zonnestraling"], errors="coerce")
    for col in extra_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[spec.time_col, spec.tin_col, spec.tout_col])
    df = df.sort_values(spec.time_col)
    if len(df) < 10:
        raise ValueError("Not enough valid samples after cleaning.")
    return df


def infer_dt_seconds(df: pd.DataFrame, *, time_col: str) -> float:
    dt_seconds = float(df[time_col].diff().dt.total_seconds().dropna().median())
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("Could not infer a positive dt from time column.")
    return dt_seconds


def build_mask(df: pd.DataFrame, *, spec: FilterSpec) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    if spec.night_start is not None and spec.night_end is not None and spec.night_start != spec.night_end:
        hours = df[spec.time_col].dt.hour.to_numpy()
        if spec.night_start < spec.night_end:
            mask &= (hours >= spec.night_start) & (hours < spec.night_end)
        else:
            mask &= (hours >= spec.night_start) | (hours < spec.night_end)
    if spec.require_cv_off:
        mask &= df["CV_mode_off"].to_numpy(dtype=float) > 0.5
    if spec.require_hp_off:
        mask &= df["warmtepomp_mode_off"].to_numpy(dtype=float) > 0.5
    if spec.require_nightmode_sleep:
        mask &= df["nachtmodus_slapen"].to_numpy(dtype=float) > 0.5
    if spec.max_solar is not None:
        mask &= df["zonnestraling"].to_numpy(dtype=float) <= spec.max_solar
    if spec.require_tout_below_tin:
        mask &= df[spec.tout_col].to_numpy(dtype=float) < df[spec.tin_col].to_numpy(dtype=float)
    if spec.require_cooling:
        tin_vals = df[spec.tin_col].to_numpy(dtype=float)
        cooling = np.zeros_like(tin_vals, dtype=bool)
        cooling[1:] = tin_vals[1:] <= tin_vals[:-1]
        mask &= cooling
    return mask


def segment_ranges_from_mask(
    df: pd.DataFrame,
    *,
    time_col: str,
    mask: np.ndarray,
    dt_seconds: float,
) -> List[Tuple[int, int]]:
    times = df[time_col].to_numpy()
    if len(times) != len(mask):
        raise ValueError("Mask length must match input length.")

    ranges: List[Tuple[int, int]] = []
    start = None
    for idx in range(len(times)):
        if not mask[idx]:
            if start is not None:
                ranges.append((start, idx))
                start = None
            continue
        if start is None:
            start = idx
            continue
        prev = times[idx - 1]
        curr = times[idx]
        gap = pd.Timedelta(curr - prev).total_seconds()
        if not math.isfinite(gap) or abs(gap - dt_seconds) > 0.1:
            ranges.append((start, idx))
            start = idx
    if start is not None:
        ranges.append((start, len(times)))
    return ranges


def filter_ranges_by_variation(
    df: pd.DataFrame,
    ranges: Iterable[Tuple[int, int]],
    *,
    tin_col: str,
    tout_col: str,
    min_segment_len: int | None,
    min_tin_range: float | None,
    min_tout_range: float | None,
) -> List[Tuple[int, int]]:
    kept: List[Tuple[int, int]] = []
    tin = df[tin_col].to_numpy(dtype=float)
    tout = df[tout_col].to_numpy(dtype=float)
    for start, end in ranges:
        seg_len = end - start
        if min_segment_len is not None and seg_len < min_segment_len:
            continue
        if min_tin_range is not None:
            tin_range = float(np.nanmax(tin[start:end]) - np.nanmin(tin[start:end]))
            if tin_range < min_tin_range:
                continue
        if min_tout_range is not None:
            tout_range = float(np.nanmax(tout[start:end]) - np.nanmin(tout[start:end]))
            if tout_range < min_tout_range:
                continue
        kept.append((start, end))
    if not kept:
        raise ValueError("No usable segments after filtering; relax segment filters.")
    return kept
