# Weather Forecast ML

An end-to-end ML engineering project on AWS that predicts the next day's
average temperature for Toronto. It covers the full lifecycle: ingestion,
ETL, feature engineering, training, deployment behind an API, and monitoring.
The goal is to demonstrate production ML engineering, not to beat a
professional weather service.

Built alongside the AWS Certified Machine Learning Engineer – Associate
certification.

## Status

Pipeline, model, API, alarms, CloudWatch dashboard, and a web UI are live.
The daily accuracy tracker is building up history; infrastructure-as-code and
CI/CD are still to do (see [Roadmap](#roadmap)).

**Live demo:** _add your GitHub Pages link here once enabled_

## Architecture

```
Open-Meteo archive API
        │
        ▼
Lambda (ingestion) ──▶ S3 raw/  (JSONL, partitioned city=/date=)
                          │
                          ▼
              Glue crawler ──▶ Glue Data Catalog
                          │
                          ▼
        Athena CTAS: clean ──▶ features   (Parquet in S3)
                          │
                          ▼
   SageMaker training job (XGBoost) ──▶ model.tar.gz in S3
                          │
                          ▼
        SageMaker serverless endpoint  ◀── invoke ──┐
                                                    │
Client ──▶ API Gateway (HTTP API, throttled) ──▶ Lambda (forecast API)
                                                    │
                                   Open-Meteo recent data ─▶ builds 13 features

EventBridge (daily) ──▶ Lambda (accuracy tracker) ──▶ calls the API
        ├─▶ S3 (stored predictions)
        └─▶ CloudWatch custom metrics (ForecastAbsError, PersistenceAbsError)

CloudWatch alarms ──▶ SNS ──▶ email
```

## Repo layout

```
ingestion/    Lambda that backfills/fetches hourly weather into S3
sql/          Athena CTAS queries: cleaning and feature engineering
training/     SageMaker training script and job launcher
deployment/   Inference handler, endpoint deploy/test/delete scripts
api/          Lambda behind API Gateway (builds features, calls endpoint)
monitoring/   Daily accuracy-tracking Lambda and CloudWatch dashboard JSON
docs/         Static web UI (GitHub Pages) that calls the live API
```

## Data

- Source: [Open-Meteo](https://open-meteo.com/) (free, no API key)
- City: Toronto, 2020-01-01 to 2025-12-31 (hourly raw, backfilled one year per
  Lambda call, aggregated to 2,192 daily rows)
- Features (13): daily avg/max/min temperature, humidity, precipitation, wind,
  pressure, cloud cover, 1- and 2-day temperature lags, 7-day rolling average,
  month, day of year
- Target: next day's average temperature (°C)

## Results

Chronological split: 70% train, 15% validation (early stopping), 15% test
(roughly Feb to Dec 2025). All models are scored on the same held-out days.

| Model | MAE (°C) | RMSE (°C) |
|---|---|---|
| Persistence (tomorrow = today) | 2.37 | 3.14 |
| 7-day average | 2.80 | 3.53 |
| **XGBoost (SageMaker training job)** | **1.99** | **2.60** |

XGBoost improves on the strongest baseline by about 16% (MAE) and 17% (RMSE).
Training took 134 billable seconds on an `ml.m5.large`.

## Deployment

- **Model endpoint:** SageMaker *serverless* inference (2 GB, max concurrency
  5) using a custom `inference.py` handler that validates and orders the 13
  input features. It scales to zero when idle; the tradeoff is a cold start
  (~10 s on the first request after a quiet period vs ~0.15 s warm).
- **API:** `GET /forecast` on an API Gateway HTTP API, throttled to 2
  requests/second (burst 5). The Lambda behind it pulls the last ~9 days of
  hourly data, aggregates complete days, builds the same features as the SQL
  pipeline, calls the endpoint, and returns the prediction plus the features
  it used.

Example response (abridged):

```json
{
  "features_day": "2026-10-19",
  "predicted_day": "2026-10-20",
  "predicted_avg_temp_c": 15.37,
  "model_endpoint": "weather-forecast-serverless",
  "features": { "avg_temp": 13.09, "temp_lag_1d": 9.84, "temp_rolling_7d": 12.51, "...": "..." }
}
```

## Web UI

`docs/index.html` is a single static page (no build step) that calls
`GET /forecast` on load and shows the prediction, a recent-days chart, the
inputs the model saw, and the baseline comparison. It handles the ~10 s
serverless cold start, rate limiting, and failures. The API must allow the
page's origin via CORS (`docs/cors.json`). Each page view is one API call, so
the throttle and budget alert matter.

## Monitoring

- **Alarms** (SNS email): API Gateway 5xx, Lambda errors, SageMaker endpoint
  5xx.
- **Accuracy tracking:** a daily scheduled Lambda calls the API, stores that
  day's prediction in S3, and the next day compares it with the actual
  temperature. It publishes `ForecastAbsError` and `PersistenceAbsError` as
  custom CloudWatch metrics, so the model can be judged against the naive
  baseline over time. An alarm triggers if 4 of the last 7 daily errors exceed
  3 °C.

## Decisions and lessons learned

- **Check data coverage before building on it.** A `min/max/count` query
  showed the table held one stray test day plus a single year, not the six
  years I assumed. Lag/lead features assume consecutive days, so I backfilled
  the missing years, rebuilt, and made the training script drop any
  non-consecutive rows as a safeguard.
- **Glue partitions:** crawl the prefix above the `key=value` folders, not the
  bucket root, or Glue invents `partition_0/1` columns. Writing `city` in both
  the JSON body and the S3 path created a duplicate column that Athena
  rejects.
- **CTAS gotchas:** partition columns must be the last columns in the SELECT,
  and `DROP TABLE` does not delete the S3 files. Orphaned files from a failed
  run silently doubled every row after the retry.
- **Time series split:** chronological, never shuffled, so no future data
  leaks into training.
- **Validate locally first.** SageMaker training initially failed with a
  service quota of 0 for the instance type. I requested an increase and ran
  the same script locally in the meantime, then confirmed the cloud job
  reproduced the local metrics.
- **Custom inference handler.** The model archive also contains
  `metrics.json`, which the stock XGBoost serving code can try to load as a
  model, so the handler loads the model file by name.
- **Alarm design.** The forecast Lambda catches its own errors and returns a
  502, so the Lambda `Errors` metric never fires for them; the API Gateway
  5xx alarm is the one that catches failed forecasts. I also first created an
  endpoint alarm on `MemoryUtilization` instead of `Invocation5XXErrors`, which
  alerted on every request. Check the metric name and namespace after creating
  any alarm.
- **IAM detail.** Without `s3:ListBucket`, a missing S3 key returns "access
  denied" instead of "not found", which breaks first-run logic. Trust policies
  and permission policies are separate editors.
- **Security basics:** worked through an IAM user with MFA instead of root,
  kept the role ARN out of the repo (environment variable), and throttled the
  public API.

## Limitations

- One city; the model sees only past observations, so it is not a substitute
  for numerical weather prediction.
- Training used the Open-Meteo archive, while the live API reads the
  recent-data endpoint, so inputs can differ slightly from training.
- The API predicts the day after the latest *complete* day (normally today in
  UTC).
- The API has throttling but no authentication.
- Feature logic exists twice (SQL for training, Python for serving), which
  risks train/serve skew; a test comparing the two is on the roadmap.

## Costs and teardown

Everything is pay-per-use: ingestion, Athena, and alarms cost pennies at this
scale, training a single job cost about a cent, and the serverless endpoint
charges only while processing requests. The main risk is the public API, so
keep an AWS Budgets alert enabled.

To tear down: run `python deployment/delete_endpoint.py`, then delete the API
Gateway API, the three Lambdas, the EventBridge schedule, the CloudWatch
alarms, the SNS topic, the Glue crawler/database, and the S3 data.

## Roadmap

- [x] Ingestion Lambda and S3 data lake
- [x] Glue/Athena cleaning and feature engineering
- [x] SageMaker training job with baselines
- [x] Serverless endpoint and API Gateway
- [x] CloudWatch alarms and daily accuracy tracking
- [ ] Accuracy alarm (once enough metric history exists)
- [x] CloudWatch dashboard
- [x] Web UI (static page on GitHub Pages)
- [ ] Infrastructure as code (Terraform or CDK)
- [ ] CI/CD with GitHub Actions
- [ ] Tests comparing SQL and Python feature logic
- [ ] Automated retraining (SageMaker Pipelines)
