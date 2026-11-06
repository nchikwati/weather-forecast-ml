"""
SageMaker training script (script mode, XGBoost framework container).

Reads the Parquet feature files produced by the Athena CTAS step, does a
chronological train/validation/test split, evaluates two naive baselines
against XGBoost, and saves the model + metrics.

Inputs  : /opt/ml/input/data/train/   (S3 prefix copied by SageMaker)
Outputs : /opt/ml/model/xgboost-model.json  and  /opt/ml/model/metrics.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

TARGET = "target_next_day_temp"
FEATURES = [
    "avg_temp", "max_temp", "min_temp", "avg_humidity", "total_precip",
    "avg_wind", "avg_pressure", "avg_cloud_cover",
    "temp_lag_1d", "temp_lag_2d", "temp_rolling_7d",
    "month", "day_of_year",
]


def load_features(data_dir):
    """Read every Parquet file under data_dir.

    Athena CTAS output files often have no .parquet extension and live in
    Hive-style partition folders (city=.../yr=.../), so walk the tree and
    try to read each non-hidden file.
    """
    frames = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.startswith((".", "_")):
                continue
            frames.append(pd.read_parquet(os.path.join(root, name)))
    if not frames:
        raise RuntimeError(f"No data files found under {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["day"] = pd.to_datetime(df["day"])
    return df.sort_values("day").reset_index(drop=True)


def drop_non_consecutive(df):
    """Lag/lead features are only valid when neighbouring rows are exactly
    one day apart. Drop rows where that isn't true (gaps in the data)."""
    prev_ok = (df["day"] - df["day"].shift(1)).dt.days == 1
    next_ok = (df["day"].shift(-1) - df["day"]).dt.days == 1
    # The first row has no previous day; the last has no next day (no target).
    return df[prev_ok & next_ok].dropna(subset=FEATURES + [TARGET])


def metrics(y_true, y_pred):
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--train-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = p.parse_args()

    print("Training directory:", train_dir := args.train_dir)
    print("Model output directory:", model_dir := args.model_dir)
    df = load_features(args.train_dir)
    print(f"Loaded {len(df)} rows spanning {df['day'].min().date()} to {df['day'].max().date()}")
    df = drop_non_consecutive(df)
    print(f"{len(df)} usable rows after dropping gaps / missing targets")

    # Chronological split: never shuffle time series data.
    n = len(df)
    train = df.iloc[: int(n * 0.70)]
    val = df.iloc[int(n * 0.70): int(n * 0.85)]
    test = df.iloc[int(n * 0.85):]
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    results = {
        # Baseline 1: tomorrow will be the same as today.
        "baseline_persistence": metrics(test[TARGET], test["avg_temp"]),
        # Baseline 2: tomorrow will be the trailing 7-day mean.
        "baseline_rolling_7d": metrics(test[TARGET], test["temp_rolling_7d"]),
    }

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(train[FEATURES], train[TARGET],
              eval_set=[(val[FEATURES], val[TARGET])], verbose=False)
    results["xgboost"] = metrics(test[TARGET], model.predict(test[FEATURES]))

    # These lines are picked up by CloudWatch and shown in the training job logs.
    for name, m in results.items():
        print(f"[metrics] {name}: mae={m['mae']:.3f} rmse={m['rmse']:.3f}")

    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(os.path.join(args.model_dir, "xgboost-model.json"))
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump({"rows": {"train": len(train), "val": len(val), "test": len(test)},
                   "features": FEATURES, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
