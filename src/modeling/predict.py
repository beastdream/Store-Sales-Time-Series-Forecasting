"""Prediction contracts for the global LightGBM sales model."""

from pathlib import Path
import shutil
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modeling.train_global import feature_matrix


def load_model(model_path: str | Path) -> lgb.Booster:
    """Load a model reliably when the project path contains Unicode characters."""
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(f"LightGBM model not found: {source}")
    temporary = Path(tempfile.gettempdir()) / "store_sales_global_lightgbm_load.txt"
    shutil.copy2(source, temporary)
    try:
        return lgb.Booster(model_file=str(temporary))
    finally:
        temporary.unlink(missing_ok=True)


def predict_sales(model: lgb.Booster, feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Invert the log target transformation and enforce nonnegative forecasts."""
    raw = np.asarray(
        model.predict(feature_matrix(feature_frame, model.feature_name())),
        dtype="float64",
    )
    prediction = np.clip(np.expm1(raw), a_min=0.0, a_max=None)
    if prediction.shape != (len(feature_frame),) or not np.isfinite(prediction).all():
        raise RuntimeError("global model produced invalid predictions")
    result = feature_frame[["date", "store_nbr", "family"]].copy()
    result["prediction"] = prediction
    if result.duplicated(["date", "store_nbr", "family"]).any():
        raise RuntimeError("predictions contain duplicate forecast grain")
    return result
