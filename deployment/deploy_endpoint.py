"""
Deploy the trained model as a SageMaker *serverless* endpoint.

Serverless endpoints scale to zero when idle, so you only pay while requests
are being processed (the tradeoff is a cold start of a few seconds on the
first request after a quiet period). That makes it a good fit for a
portfolio project.

Run from the deployment/ folder, with the same venv as the training launcher:
    $env:SAGEMAKER_ROLE_ARN = "arn:aws:iam::<account>:role/weather-sagemaker-execution-role"
    python deploy_endpoint.py
"""
import os

import boto3
import sagemaker
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.xgboost import XGBoostModel

PROFILE = "lambda_handler"
REGION = "us-east-2"
BUCKET = "nigel-weather-ml-raw"
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]

# Model artifact written by the training job (see the "Model artifact" line
# printed at the end of launch_training.py).
MODEL_DATA = (
    "s3://nigel-weather-ml-raw/models/"
    "sagemaker-xgboost-2026-10-19-15-50-22-933/output/model.tar.gz"
)
ENDPOINT_NAME = "weather-forecast-serverless"

session = sagemaker.Session(
    boto_session=boto3.Session(profile_name=PROFILE, region_name=REGION),
    default_bucket=BUCKET,
)

model = XGBoostModel(
    model_data=MODEL_DATA,
    role=ROLE_ARN,
    entry_point="inference.py",
    source_dir=".",                 # uploads inference.py with the model
    framework_version="1.7-1",      # same container version used for training
    sagemaker_session=session,
)

model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=2048,     # more memory = more CPU = faster cold starts
        max_concurrency=5,
    ),
    endpoint_name=ENDPOINT_NAME,
)

print(f"Endpoint '{ENDPOINT_NAME}' is InService. Test it with: python test_endpoint.py")
