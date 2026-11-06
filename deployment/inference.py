"""
Custom inference handler for the SageMaker XGBoost framework container.

SageMaker calls these four functions for every request:
    model_fn   -> load the model once at container start
    input_fn   -> parse the request body
    predict_fn -> run the model
    output_fn  -> format the response

Request  (Content-Type: application/json):
    {"instances": [ {<13 feature names>: value, ...}, ... ]}
    A single feature object (without "instances") is also accepted.

Response (application/json):
    {"predictions": [<next-day average temperature in degrees C>, ...]}
"""
import json
import os

import numpy as np
import xgboost as xgb

# Must match the training feature list AND order (see training/train.py).
FEATURES = [
    "avg_temp", "max_temp", "min_temp", "avg_humidity", "total_precip",
    "avg_wind", "avg_pressure", "avg_cloud_cover",
    "temp_lag_1d", "temp_lag_2d", "temp_rolling_7d",
    "month", "day_of_year",
]


def model_fn(model_dir):
    """Load the model explicitly by name.

    The model archive also contains metrics.json, so we avoid the default
    behaviour of treating every file in the archive as a model.
    """
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "xgboost-model.json"))
    return booster


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")

    payload = json.loads(request_body)
    rows = payload["instances"] if isinstance(payload, dict) and "instances" in payload else [payload]
    if not rows:
        raise ValueError("No instances provided.")

    matrix = []
    for i, row in enumerate(rows):
        missing = [f for f in FEATURES if f not in row]
        if missing:
            raise ValueError(f"Instance {i} is missing features: {missing}")
        matrix.append([float(row[f]) for f in FEATURES])

    return xgb.DMatrix(np.array(matrix), feature_names=FEATURES)


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept="application/json"):
    return json.dumps({"predictions": [float(p) for p in prediction]})
