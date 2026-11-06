"""
Launch the SageMaker training job from your machine.

Setup (once):
    pip install -r requirements-launcher.txt

Run:
    python launch_training.py

Edit PROFILE / BUCKET / ROLE_ARN below first. The role must be a SageMaker
execution role (trusted by sagemaker.amazonaws.com) with read/write access to
the bucket, e.g. AmazonSageMakerFullAccess + an S3 policy for your bucket.
"""
import os

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost

PROFILE = "lambda_handler"  # your local AWS CLI profile
REGION = "us-east-2"
BUCKET = "nigel-weather-ml-raw"
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]

session = sagemaker.Session(
    boto_session=boto3.Session(profile_name=PROFILE, region_name=REGION),
    default_bucket=BUCKET,
)

estimator = XGBoost(
    entry_point="train.py",
    source_dir=".",                      # uploads train.py + requirements.txt
    framework_version="1.7-1",
    role=ROLE_ARN,
    instance_type="ml.m5.large",         # small + cheap; the dataset is tiny
    instance_count=1,
    max_run=900,                         # hard stop after 15 min as a cost guard
    output_path=f"s3://{BUCKET}/models/",
    hyperparameters={"n-estimators": 500, "max-depth": 4, "learning-rate": 0.05},
    sagemaker_session=session,
)

estimator.fit({"train": TrainingInput(f"s3://{BUCKET}/features/weather/")})
print("Model artifact:", estimator.model_data)
