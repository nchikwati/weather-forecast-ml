"""
Forecast API Lambda  (API Gateway  ->  this function  ->  SageMaker endpoint).

GET /forecast
  1. Fetches the last ~9 days of hourly weather from Open-Meteo.
  2. Aggregates complete days to daily values and builds the same 13 features
     the model was trained on (same definitions as sql/02_build_features.sql).
  3. Calls the SageMaker endpoint and returns the predicted average
     temperature for the day AFTER the latest complete day of data.

Because the latest complete day is normally yesterday (UTC), the predicted
day is normally today (UTC). The response states both dates explicitly.

Note: training used the Open-Meteo archive (reanalysis) data, while this
function reads the recent-data forecast API, so inputs can differ slightly
from what the model saw during training.

Environment variables:
    ENDPOINT_NAME   SageMaker endpoint name (default weather-forecast-serverless)
    LATITUDE        e.g. "43.65"
    LONGITUDE       e.g. "-79.38"

Permissions: the execution role needs sagemaker:InvokeEndpoint on the endpoint.
Timeout: set the Lambda timeout to ~29s (a cold serverless endpoint can take ~10s).
"""
import json
import logging
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

API_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
    "cloud_cover",
]

runtime = boto3.client("sagemaker-runtime")


def _response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def fetch_hourly(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "past_days": 9,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "weather-forecast-ml/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def daily_aggregates(payload):
    """Return {date: aggregates} for COMPLETE past days only (24 non-null hours)."""
    hourly = payload["hourly"]
    by_day = defaultdict(lambda: defaultdict(list))
    for i, ts in enumerate(hourly["time"]):
        for var in HOURLY_VARS:
            by_day[ts[:10]][var].append(hourly[var][i])

    today = datetime.now(timezone.utc).date()
    out = {}
    for day_str, cols in by_day.items():
        day = date.fromisoformat(day_str)
        complete = day < today and all(
            len(v) == 24 and all(x is not None for x in v) for v in cols.values()
        )
        if not complete:
            continue
        t = cols["temperature_2m"]
        out[day] = {
            "avg_temp": sum(t) / 24,
            "max_temp": max(t),
            "min_temp": min(t),
            "avg_humidity": sum(cols["relative_humidity_2m"]) / 24,
            "total_precip": sum(cols["precipitation"]),
            "avg_wind": sum(cols["wind_speed_10m"]) / 24,
            "avg_pressure": sum(cols["surface_pressure"]) / 24,
            "avg_cloud_cover": sum(cols["cloud_cover"]) / 24,
        }
    return out


def build_features(days):
    """Build the model's 13 features for the latest complete day."""
    last7 = sorted(days)[-7:]
    if len(last7) < 7 or (last7[-1] - last7[0]).days != 6:
        raise ValueError("Fewer than 7 consecutive complete days of data available")

    d = last7[-1]
    features = dict(days[d])
    features["temp_lag_1d"] = days[d - timedelta(days=1)]["avg_temp"]
    features["temp_lag_2d"] = days[d - timedelta(days=2)]["avg_temp"]
    features["temp_rolling_7d"] = sum(days[x]["avg_temp"] for x in last7) / 7
    features["month"] = d.month
    features["day_of_year"] = d.timetuple().tm_yday
    return d, features


def lambda_handler(event, context):
    try:
        lat = float(os.environ["LATITUDE"])
        lon = float(os.environ["LONGITUDE"])
        endpoint = os.environ.get("ENDPOINT_NAME", "weather-forecast-serverless")

        days = daily_aggregates(fetch_hourly(lat, lon))
        feature_day, features = build_features(days)

        result = runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=json.dumps({"instances": [features]}),
        )
        prediction = json.loads(result["Body"].read())["predictions"][0]

        return _response(200, {
            "features_day": feature_day.isoformat(),
            "predicted_day": (feature_day + timedelta(days=1)).isoformat(),
            "predicted_avg_temp_c": round(prediction, 2),
            "model_endpoint": endpoint,
            "features": {k: round(v, 2) if isinstance(v, float) else v for k, v in features.items()},
        })
    except Exception:
        # Full details go to CloudWatch; callers get a generic message.
        logger.exception("Forecast request failed")
        return _response(502, {"error": "Could not produce a forecast right now. Please try again."})