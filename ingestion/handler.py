"""
Weather ingestion Lambda.

Fetches hourly historical weather from the Open-Meteo archive API (no API key)
and writes one JSON Lines file per day to S3, partitioned for Glue/Athena:

    s3://<BUCKET>/raw/weather/city=<city>/date=YYYY-MM-DD/data.jsonl

Environment variables (required):
    BUCKET      S3 bucket name
    CITY        city label used in the S3 partition, e.g. "toronto"
    LATITUDE    e.g. "43.65"
    LONGITUDE   e.g. "-79.38"

Optional:
    PREFIX             S3 key prefix (default "raw/weather")
    LOOKBACK_DAYS      days re-fetched on each scheduled run (default 7)
    ARCHIVE_LAG_DAYS   archive data lags real time (default 5)

Invocation:
    - Scheduled (EventBridge): no payload needed, fetches the recent window.
    - Backfill: {"start_date": "2020-01-01", "end_date": "2020-12-31"}

Writes are idempotent: re-running a day overwrites the same S3 key.
Uses only the standard library + boto3 (already in the Lambda runtime).
"""
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "surface_pressure",
    "cloud_cover",
]

s3 = boto3.client("s3")


def _cfg():
    missing = [k for k in ("BUCKET", "CITY", "LATITUDE", "LONGITUDE") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    return {
        "bucket": os.environ["BUCKET"],
        "city": os.environ["CITY"].strip().lower().replace(" ", "-"),
        "lat": float(os.environ["LATITUDE"]),
        "lon": float(os.environ["LONGITUDE"]),
        "prefix": os.environ.get("PREFIX", "raw/weather").strip("/"),
        "lookback": int(os.environ.get("LOOKBACK_DAYS", "7")),
        "lag": int(os.environ.get("ARCHIVE_LAG_DAYS", "5")),
    }


def _date_range(event, cfg):
    """Return (start, end) as date objects."""
    if event.get("start_date") and event.get("end_date"):
        start = date.fromisoformat(event["start_date"])
        end = date.fromisoformat(event["end_date"])
    else:
        end = datetime.now(timezone.utc).date() - timedelta(days=cfg["lag"])
        start = end - timedelta(days=cfg["lookback"] - 1)
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")
    return start, end


def _fetch(cfg, start, end):
    params = {
        "latitude": cfg["lat"],
        "longitude": cfg["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    logger.info("Fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "weather-forecast-ml/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Open-Meteo returned HTTP {e.code}: {body}") from e


def _rows_by_day(payload, cfg):
    """Turn Open-Meteo's columnar response into per-day lists of row dicts."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("Unexpected API response: no hourly data")

    ingested_at = datetime.now(timezone.utc).isoformat()
    days = defaultdict(list)
    for i, ts in enumerate(hourly["time"]):
        row = {"city": cfg["city"], "timestamp": ts}
        for var in HOURLY_VARS:
            row[var] = hourly.get(var, [None] * len(hourly["time"]))[i]
        row["ingested_at"] = ingested_at
        days[ts[:10]].append(row)  # "YYYY-MM-DDTHH:MM" -> "YYYY-MM-DD"
    return days


def _write_day(cfg, day, rows):
    key = f"{cfg['prefix']}/city={cfg['city']}/date={day}/data.jsonl"
    body = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    s3.put_object(Bucket=cfg["bucket"], Key=key, Body=body, ContentType="application/x-ndjson")
    return key


def lambda_handler(event, context):
    event = event or {}
    cfg = _cfg()
    start, end = _date_range(event, cfg)

    payload = _fetch(cfg, start, end)
    days = _rows_by_day(payload, cfg)

    keys = [_write_day(cfg, day, rows) for day, rows in sorted(days.items())]
    logger.info("Wrote %d day files for %s (%s to %s)", len(keys), cfg["city"], start, end)

    return {
        "statusCode": 200,
        "city": cfg["city"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "files_written": len(keys),
    }
