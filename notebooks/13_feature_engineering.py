# %% [markdown]
# # Audited Feature Engineering for Global Forecasting
#
# This notebook entrypoint demonstrates the reusable feature builders. It writes
# no full feature dataset. It audits D+1 features; multi-step construction belongs
# to the recursive forecaster because later days require prior model predictions.

# %%
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_raw import load_holidays, load_stores, load_train
from src.modeling.train_global import (
    FEATURE_COLUMNS,
    add_known_features,
    build_horizon_safe_features,
)


def build_validation_sample(days_of_context: int = 400) -> pd.DataFrame:
    """Build a small real-data horizon sample without persisting it."""
    train = load_train()
    validation_end = train["date"].max().normalize()
    validation_start = validation_end
    origin = validation_start - pd.Timedelta(days=1)
    context_start = origin - pd.Timedelta(days=days_of_context)
    sample = train.loc[train["date"].between(context_start, validation_end)]
    known = add_known_features(sample, load_stores(), load_holidays())
    return build_horizon_safe_features(
        known, origin, validation_start, validation_end
    )


def main() -> None:
    sample = build_validation_sample()
    print(f"Validation sample rows: {len(sample):,}")
    print(f"Model features: {len(FEATURE_COLUMNS)}")
    print("No feature artifact or model was written.")


if __name__ == "__main__":
    main()
