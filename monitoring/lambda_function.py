"""
Daily accuracy tracker (EventBridge schedule -> this function).

Each run calls the forecast API once and uses its response in two ways:

  1. EVALUATE: the response's features describe the latest complete day D.
     features.avg_temp is the ACTUAL average temperature of day D. The
     prediction for day D was made (and saved to S3) by the previous run, so
     we compare the two and publish the error as a CloudWatch metric. We also
     publish the error of the naive "persistence" baseline (tomorrow = today)
     so the model can be judged against it over time.
  2. RECORD: the response's own prediction (for day D+1) is saved to S3, ready
     to be evaluated by tomorrow's run.

Each day is evaluated at most once (an "evaluations/<day>.json" marker is saved
in S3 after the first evaluation), so manual test runs cannot duplicate datapoints.
The saved marker is also a small per-day accuracy log you can query later.

Because it calls the public API, it doubles as a daily end-to-end health check.
If the API fails, this function raises, so the Lambda Errors alarm can fire.

Metrics (namespace WeatherForecastML):
    ForecastAbsError     |model prediction - actual|, degrees C
    PersistenceAbsError  |yesterday's temperature - actual|, degrees C

Environment variables:
    API_URL   full URL of the forecast API, e.g. https://<id>.execute-api.<region>.amazonaws.com/forecast
    BUCKET    S3 bucket for stored predictions
    PREFIX    key prefix (default "monitoring")

Permissions: s3:GetObject/PutObject on <PREFIX>/*, s3:ListBucket on the bucket
(needed so a missing key reads as "not found" instead of "access denied"),
and cloudwatch:PutMetricData.
Timeout: ~60s (the API call can hit a cold serverless endpoint).
"""
import json
import logging
import os
import urllib.request

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

NAMESPACE = "WeatherForecastML"
s3 = boto3.client("s3")
cloudwatch = boto3.client("cloudwatch")


def call_api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "weather-forecast-ml-tracker/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_stored_prediction(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(obj["Body"].read())["predicted_avg_temp_c"]


def key_exists(bucket, key):
    """True if the S3 object exists. Needs s3:ListBucket so a missing key is a 404, not a 403."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def lambda_handler(event, context):
    url = os.environ["API_URL"]
    bucket = os.environ["BUCKET"]
    prefix = os.environ.get("PREFIX", "monitoring").strip("/")

    result = call_api(url)
    feature_day = result["features_day"]
    predicted_day = result["predicted_day"]
    actual = result["features"]["avg_temp"]
    persistence = result["features"]["temp_lag_1d"]  # yesterday's value = persistence forecast for feature_day

    # 1. Evaluate the prediction made by the previous run for feature_day.
    #    Each day is evaluated at most once: a marker object is written after the
    #    first evaluation, so manual re-runs cannot publish duplicate datapoints.
    marker_key = f"{prefix}/evaluations/{feature_day}.json"
    stored = load_stored_prediction(bucket, f"{prefix}/predictions/{feature_day}.json")
    if stored is None:
        logger.info("No stored prediction for %s yet; nothing to evaluate", feature_day)
        evaluated = False
    elif key_exists(bucket, marker_key):
        logger.info("%s was already evaluated; skipping to avoid duplicate metrics", feature_day)
        evaluated = False
    else:
        model_err = abs(stored - actual)
        persistence_err = abs(persistence - actual)
        cloudwatch.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[
                {"MetricName": "ForecastAbsError", "Value": model_err, "Unit": "None"},
                {"MetricName": "PersistenceAbsError", "Value": persistence_err, "Unit": "None"},
            ],
        )
        s3.put_object(
            Bucket=bucket,
            Key=marker_key,
            Body=json.dumps({
                "day": feature_day,
                "predicted": stored,
                "actual": actual,
                "model_abs_error": model_err,
                "persistence_abs_error": persistence_err,
            }).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("%s: predicted=%.2f actual=%.2f model_err=%.2f persistence_err=%.2f",
                    feature_day, stored, actual, model_err, persistence_err)
        evaluated = True

    # 2. Record today's prediction so tomorrow's run can evaluate it.
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/predictions/{predicted_day}.json",
        Body=json.dumps({
            "predicted_day": predicted_day,
            "predicted_avg_temp_c": result["predicted_avg_temp_c"],
            "features_day": feature_day,
        }).encode("utf-8"),
        ContentType="application/json",
    )

    return {"feature_day": feature_day, "predicted_day": predicted_day, "evaluated_previous": evaluated}