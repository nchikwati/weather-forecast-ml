"""
Send a sample request to the deployed endpoint.

    python test_endpoint.py

The values below are a made-up, plausible winter day. They describe ONE day
and the model predicts the NEXT day's average temperature in degrees C.
"""
import json
import time

import boto3

PROFILE = "lambda_handler"
REGION = "us-east-2"
ENDPOINT_NAME = "weather-forecast-serverless"

sample = {
    "instances": [{
        "avg_temp": -2.0,
        "max_temp": 0.5,
        "min_temp": -4.5,
        "avg_humidity": 75.0,
        "total_precip": 0.5,
        "avg_wind": 18.0,
        "avg_pressure": 1005.0,
        "avg_cloud_cover": 80.0,
        "temp_lag_1d": -1.0,
        "temp_lag_2d": 0.0,
        "temp_rolling_7d": -0.5,
        "month": 1,
        "day_of_year": 15,
    }]
}

runtime = boto3.Session(profile_name=PROFILE, region_name=REGION).client("sagemaker-runtime")

for attempt in (1, 2):  # the second call shows the warm latency
    start = time.time()
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(sample),
    )
    result = json.loads(response["Body"].read())
    print(f"call {attempt}: {result}  ({time.time() - start:.2f}s)")
