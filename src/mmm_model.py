"""
mmm_model.py
Bayesian Marketing Mix Model using PyMC-Marketing.
Handles adstock, saturation (Hill function), and uncertainty quantification.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
import warnings
warnings.filterwarnings("ignore")


SPEND_COLS = [
    "tv_spend",
    "paid_social_spend",
    "influencer_spend",
    "paid_search_spend",
    "ooh_spend",
    "email_spend",
]

DATE_COL = "date"
TARGET_COL = "revenue"


def load_and_prepare(path: str = "data/marketing_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    # Drop internal truth columns for modelling
    df = df[[c for c in df.columns if not c.startswith("_true")]]
    return df


def build_mmm(df: pd.DataFrame) -> MMM:
    """
    Build a Bayesian MMM with:
    - Geometric adstock (carry-over effect)
    - Logistic saturation (diminishing returns)
    - Control variables: promotions, consumer confidence
    - Fourier seasonality (weekly patterns)
    """
    mmm = MMM(
        adstock=GeometricAdstock(l_max=8),          # up to 8-week carry-over
        saturation=LogisticSaturation(),
        date_column=DATE_COL,
        channel_columns=SPEND_COLS,
        control_columns=["is_promotion", "consumer_confidence"],
        yearly_seasonality=2,                        # 2 Fourier pairs
    )
    return mmm


def fit_mmm(df: pd.DataFrame, mmm: MMM, samples: int = 1000, tune: int = 1000) -> MMM:
    """Sample posterior using NUTS (No U-Turn Sampler)."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    mmm.fit(
        X=X,
        y=y,
        target_accept=0.9,
        draws=samples,
        tune=tune,
        chains=2,
        random_seed=42,
        progressbar=True,
    )
    return mmm


def extract_contributions(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose revenue into:
    - Base (trend + seasonality + controls)
    - Each media channel contribution
    Returns weekly contribution dataframe.
    """
    contrib = mmm.get_mean_contributions_over_time(original_scale=True)
    contrib.index = df[DATE_COL].values
    contrib.index.name = "date"
    return contrib


def compute_roi_summary(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ROI per channel with 94% credible interval."""
    rows = []
    idata = mmm.idata
    spend_totals = df[SPEND_COLS].sum()

    for ch in SPEND_COLS:
        # Revenue contribution from posterior
        contrib_mean = (
            mmm.get_mean_contributions_over_time(original_scale=True)[ch].sum()
        )
        spend = spend_totals[ch]
        roi = contrib_mean / spend if spend > 0 else 0

        rows.append({
            "channel": ch.replace("_spend", "").replace("_", " ").title(),
            "total_spend_gbp": round(spend, 0),
            "attributed_revenue_gbp": round(contrib_mean, 0),
            "mean_roi": round(roi, 2),
        })

    return pd.DataFrame(rows).sort_values("mean_roi", ascending=False)


def optimize_budget(mmm: MMM, df: pd.DataFrame,
                    total_budget: float = None,
                    n_sim: int = 500) -> pd.DataFrame:
    """
    Simulate budget re-allocation using response curves.
    Returns optimal spend per channel given total budget constraint.
    """
    current_spend = df[SPEND_COLS].sum()
    if total_budget is None:
        total_budget = float(current_spend.sum())

    # Use PyMC-Marketing budget optimizer
    budget_dist = current_spend / current_spend.sum()  # start from current mix
    result = mmm.optimize_budget(
        budget=total_budget,
        num_periods=len(df),
        budget_bounds={ch: (0, total_budget * 0.6) for ch in SPEND_COLS},
    )
    return result


if __name__ == "__main__":
    df = load_and_prepare()
    print(f"Loaded {len(df)} weeks of data")

    mmm = build_mmm(df)
    print("Fitting model (this takes ~5 minutes)...")
    mmm = fit_mmm(df, mmm, samples=500, tune=500)

    contrib = extract_contributions(mmm, df)
    contrib.to_csv("outputs/channel_contributions.csv")

    roi = compute_roi_summary(mmm, df)
    roi.to_csv("outputs/roi_summary.csv", index=False)
    print(roi)

    mmm.save("outputs/mmm_model")
    print("Model saved.")
