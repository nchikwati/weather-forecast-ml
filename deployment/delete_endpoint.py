"""
Tear down the endpoint, its endpoint config, and its model.

    python delete_endpoint.py

Serverless endpoints cost nothing while idle, but delete them when you are
done with the project so nothing is left lying around in your account.
"""
import boto3

PROFILE = "lambda_handler"
REGION = "us-east-2"
ENDPOINT_NAME = "weather-forecast-serverless"

sm = boto3.Session(profile_name=PROFILE, region_name=REGION).client("sagemaker")

config_name = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointConfigName"]
model_name = sm.describe_endpoint_config(EndpointConfigName=config_name)["ProductionVariants"][0]["ModelName"]

sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
sm.delete_endpoint_config(EndpointConfigName=config_name)
sm.delete_model(ModelName=model_name)
print(f"Deleted endpoint '{ENDPOINT_NAME}', config '{config_name}', model '{model_name}'.")
