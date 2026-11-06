#!/usr/bin/env python3
"""
pick_day.py — Stage 1: screen a summer for the best "fair-weather cumulus
with lake breeze" day over Toronto.

Two points are compared, using hourly model output from the Open-Meteo
historical archive:
    INLAND  north of the city, where afternoon cumulus should develop
    LAKE    just offshore in Lake Ontario, where the lake breeze should
            keep the sky clear

A day scores well (low score = better) when:
    - the morning is clear at both points            (07:00-09:00 local)
    - the inland afternoon has scattered low cloud   (13:00-16:00, ~40%)
    - the lake stays clear in the afternoon          (lake-breeze contrast)
    - the evening clears again                       (19:00-20:00)
    - there is little mid or high cloud              (no cirrus veil)
    - there is no rain
    - the morning wind is light and the afternoon wind at the shore is
      onshore (from the lake, roughly SE to S), the lake-breeze signature

Every threshold below is a starting point to adjust, not a rule.

Usage:
    python pick_day.py --start 2026-06-01 --end 2026-08-31 --top 10
Writes candidates.csv and prints the ranking.
"""

import argparse

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
VARS = [
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]

POINTS = {
    "inland": (43.85, -79.45),  # Vaughan / Richmond Hill
    "lake": (43.55, -79.30),  # ~10 km offshore of downtown
    "shore": (43.64, -79.38),  # waterfront, for the wind signature
}


def fetch(lat, lon, start, end):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARS),
        "timezone": "America/Toronto",
    }
    r = requests.get(ARCHIVE, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    return df


def window_mean(df, col, h0, h1):
    """Mean of `col` per day over local hours h0..h1 inclusive."""
    sel = df[(df["hour"] >= h0) & (df["hour"] <= h1)]
    return sel.groupby("date")[col].mean()


def circular_mean_deg(deg):
    r = np.radians(deg)
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360


def score_days(inland, lake, shore):
    d = pd.DataFrame(index=sorted(set(inland["date"])))
    d["inland_morning"] = window_mean(inland, "cloud_cover_low", 7, 9)
    d["lake_morning"] = window_mean(lake, "cloud_cover_low", 7, 9)
    d["inland_afternoon"] = window_mean(inland, "cloud_cover_low", 13, 16)
    d["lake_afternoon"] = window_mean(lake, "cloud_cover_low", 13, 16)
    d["inland_evening"] = window_mean(inland, "cloud_cover_low", 19, 20)
    upper = inland.assign(
        upper=inland[["cloud_cover_mid", "cloud_cover_high"]].max(axis=1)
    )
    d["upper_cloud"] = window_mean(upper, "upper", 8, 19)
    d["rain_mm"] = inland.groupby("date")["precipitation"].sum()
    d["morning_wind_kmh"] = window_mean(shore, "wind_speed_10m", 8, 10)
    pm = shore[(shore["hour"] >= 13) & (shore["hour"] <= 16)]
    d["afternoon_wind_from"] = pm.groupby("date")["wind_direction_10m"].apply(
        circular_mean_deg
    )
    d["onshore"] = d["afternoon_wind_from"].between(90, 200)

    s = (
        (d["inland_afternoon"] - 40).abs()  # scattered Cu
        + 1.5 * (d["inland_morning"] - 15).clip(lower=0)  # clear morning
        + 1.0 * (d["lake_morning"] - 15).clip(lower=0)
        + 1.5 * (d["inland_evening"] - 20).clip(lower=0)  # clears again
        + 1.0 * (d["lake_afternoon"] - 10).clip(lower=0)  # lake stays clear
        + 2.0 * d["upper_cloud"]  # no cirrus
        + 50.0 * (d["rain_mm"] > 0.2)  # dry
        + 15.0 * (~d["onshore"])  # lake breeze
        + 1.0 * (d["morning_wind_kmh"] - 12).clip(lower=0)  # light synoptic wind
    )
    d["score"] = s
    return d.sort_values("score")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", default="2026-06-01")
    p.add_argument("--end", default="2026-08-31")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", default="candidates.csv")
    a = p.parse_args()

    frames = {k: fetch(*v, a.start, a.end) for k, v in POINTS.items()}
    ranked = score_days(frames["inland"], frames["lake"], frames["shore"])
    ranked.round(1).to_csv(a.out, index_label="date")
    cols = [
        "score",
        "inland_morning",
        "inland_afternoon",
        "lake_afternoon",
        "inland_evening",
        "upper_cloud",
        "rain_mm",
        "afternoon_wind_from",
    ]
    with pd.option_context("display.width", 140):
        print(ranked[cols].head(a.top).round(1))
    print(f"\nFull ranking written to {a.out}")


if __name__ == "__main__":
    main()
