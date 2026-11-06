-- Step 1: Clean + dedupe the raw weather data, convert to Parquet.
-- Drop and recreate if you rerun this after a new backfill.

DROP TABLE IF EXISTS weather_ml.weather_clean;

CREATE TABLE weather_ml.weather_clean
WITH (
    format = 'PARQUET',
    external_location = 's3://nigel-weather-ml-raw/clean/weather/',
    partitioned_by = ARRAY['city', 'yr']
) AS
SELECT
    CAST(parse_datetime(timestamp, 'yyyy-MM-dd''T''HH:mm') AS timestamp) AS ts,
    temperature_2m,
    relative_humidity_2m,
    precipitation,
    rain,
    wind_speed_10m,
    surface_pressure,
    cloud_cover,
    lower(trim(city)) AS city,
    year(parse_datetime(timestamp, 'yyyy-MM-dd''T''HH:mm')) AS yr
FROM weather_ml.weather
WHERE temperature_2m IS NOT NULL
GROUP BY
    timestamp, temperature_2m, relative_humidity_2m, precipitation,
    rain, wind_speed_10m, surface_pressure, cloud_cover, lower(trim(city));
-- GROUP BY on every column is a simple way to dedupe exact-duplicate rows,
-- since your gap check showed none, this mostly just future-proofs reruns.
