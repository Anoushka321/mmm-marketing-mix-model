"""
generate_data.py
Generates realistic synthetic marketing mix data for a luxury beauty brand.
Includes seasonality, adstock effects, media channels, and macroeconomic signals.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

np.random.seed(42)

# ── Config ─────────────────────────────────────────────────────────────────
N_WEEKS = 156  # 3 years of weekly data
START_DATE = date(2021, 1, 4)

CHANNELS = {
    "tv_spend":         {"true_roi": 1.8, "adstock_decay": 0.6,  "saturation_alpha": 2.5},
    "paid_social_spend":{"true_roi": 2.4, "adstock_decay": 0.3,  "saturation_alpha": 1.8},
    "influencer_spend": {"true_roi": 3.1, "adstock_decay": 0.45, "saturation_alpha": 2.0},
    "paid_search_spend":{"true_roi": 4.2, "adstock_decay": 0.15, "saturation_alpha": 3.5},
    "ooh_spend":        {"true_roi": 1.2, "adstock_decay": 0.7,  "saturation_alpha": 1.5},
    "email_spend":      {"true_roi": 5.0, "adstock_decay": 0.05, "saturation_alpha": 4.0},
}

BASE_REVENUE = 1_200_000  # £ per week baseline


# ── Helper functions ────────────────────────────────────────────────────────

def adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock transformation - carries ad effect forward in time."""
    result = np.zeros_like(spend, dtype=float)
    result[0] = spend[0]
    for t in range(1, len(spend)):
        result[t] = spend[t] + decay * result[t - 1]
    return result


def hill_saturation(x: np.ndarray, alpha: float, gamma: float = 0.5) -> np.ndarray:
    """Hill function: diminishing returns on media spend."""
    x_norm = x / (x.max() + 1e-9)
    return x_norm ** alpha / (x_norm ** alpha + gamma ** alpha)


def simulate_spend(n: int, base: float, noise_scale: float,
                   seasonal_peaks: list[int]) -> np.ndarray:
    """Simulate realistic weekly spend with seasonality and bursts."""
    base_spend = np.random.gamma(shape=2, scale=base / 2, size=n)
    # Campaign bursts
    for peak_week in seasonal_peaks:
        if peak_week < n:
            base_spend[max(0, peak_week-2):peak_week+3] *= np.random.uniform(2.5, 4.5)
    noise = np.random.normal(0, noise_scale, n)
    return np.clip(base_spend + noise, 0, None)


# ── Seasonality ─────────────────────────────────────────────────────────────

def build_seasonality(n: int) -> np.ndarray:
    """Captures Valentine's, Summer, Christmas peaks typical for beauty brands."""
    t = np.arange(n)
    # Annual cycle (Christmas/gifting peak in Dec)
    annual = 0.25 * np.sin(2 * np.pi * t / 52 - np.pi / 2)
    # Valentine's mini-peak (week ~6 each year)
    valentines = sum(
        0.12 * np.exp(-0.5 * ((t - (6 + 52 * y)) / 1.5) ** 2)
        for y in range(3)
    )
    # Mothering Sunday (week ~12 each year)
    mothers = sum(
        0.10 * np.exp(-0.5 * ((t - (12 + 52 * y)) / 1.5) ** 2)
        for y in range(3)
    )
    return annual + valentines + mothers


# ── Main generation ─────────────────────────────────────────────────────────

def generate_dataset() -> pd.DataFrame:
    dates = [START_DATE + timedelta(weeks=i) for i in range(N_WEEKS)]
    week_index = np.arange(N_WEEKS)

    # Macro: post-COVID recovery trend + cost-of-living dip
    trend = 0.002 * week_index - 0.00002 * week_index ** 2
    seasonality = build_seasonality(N_WEEKS)

    # Campaign peaks: Black Friday/Christmas (week 48,100,152), Valentine's, Summer launch
    peaks = {
        "tv_spend":          [6, 46, 58, 98, 110, 150],
        "paid_social_spend": [4, 20, 44, 58, 96, 110, 148],
        "influencer_spend":  [5, 12, 30, 50, 60, 102, 152],
        "paid_search_spend": [2, 8, 46, 58, 99, 151],
        "ooh_spend":         [10, 48, 100, 150],
        "email_spend":       [1, 6, 12, 20, 30, 45, 58, 100, 110, 152],
    }

    spend_bases = {
        "tv_spend":          80_000,
        "paid_social_spend": 60_000,
        "influencer_spend":  50_000,
        "paid_search_spend": 30_000,
        "ooh_spend":         40_000,
        "email_spend":       8_000,
    }

    df = pd.DataFrame({"date": dates, "week": week_index})

    raw_spends = {}
    adstocked = {}
    saturated = {}

    for ch, cfg in CHANNELS.items():
        raw = simulate_spend(N_WEEKS, spend_bases[ch], spend_bases[ch] * 0.3, peaks[ch])
        ads = adstock(raw, cfg["adstock_decay"])
        sat = hill_saturation(ads, cfg["saturation_alpha"])
        raw_spends[ch] = raw
        adstocked[ch] = ads
        saturated[ch] = sat
        df[ch] = raw

    # Total media contribution to revenue
    media_contrib = sum(
        saturated[ch] * CHANNELS[ch]["true_roi"] * spend_bases[ch]
        for ch in CHANNELS
    )

    # Base + trend + seasonality + media + noise
    noise = np.random.normal(0, 0.03, N_WEEKS)
    revenue = BASE_REVENUE * (1 + trend + seasonality + noise) + media_contrib

    df["revenue"] = np.clip(revenue, 0, None).round(2)

    # Add contextual variables
    df["is_promotion"] = 0
    promo_weeks = [6, 12, 46, 48, 58, 96, 100, 110, 150, 152]
    df.loc[df["week"].isin(promo_weeks), "is_promotion"] = 1

    df["competitor_index"] = (
        100 + 15 * np.sin(2 * np.pi * week_index / 26)
        + np.random.normal(0, 3, N_WEEKS)
    ).round(1)

    df["consumer_confidence"] = (
        -10 + 5 * np.sin(2 * np.pi * week_index / 52)
        + np.random.normal(0, 2, N_WEEKS)
        - np.linspace(0, 8, N_WEEKS)  # cost-of-living squeeze
    ).round(1)

    # Store true contributions for model evaluation
    df["_true_base"] = BASE_REVENUE * (1 + trend + seasonality + noise)
    for ch in CHANNELS:
        df[f"_true_{ch}_contrib"] = (saturated[ch] * CHANNELS[ch]["true_roi"] * spend_bases[ch]).round(2)
        df[f"_true_{ch}_roi"] = CHANNELS[ch]["true_roi"]

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/marketing_data.csv", index=False)
    print(f"Generated {len(df)} weeks of data")
    print(f"Revenue range: £{df['revenue'].min():,.0f} – £{df['revenue'].max():,.0f}")
    print(f"Total spend: £{df[[c for c in df.columns if c.endswith('_spend')]].sum().sum():,.0f}")
    print(df.head())
