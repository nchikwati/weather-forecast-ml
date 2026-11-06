-- Step 2: Feature engineering on top of weather_clean.
-- Target: next-day average temperature (a harder, more interesting forecast
-- than next-hour). Adjust the LEAD window if you'd rather predict next-hour.

DROP TABLE IF EXISTS weather_ml.weather_features;

CREATE TABLE weather_ml.weather_features
WITH (
    format = 'PARQUET',
    external_location = 's3://nigel-weather-ml-raw/features/weather/',
    partitioned_by = ARRAY['city', 'yr']
) AS
WITH daily AS (
    SELECT
        city AS city_name,
        date_trunc('day', ts) AS day,
        avg(temperature_2m) AS avg_temp,
        max(temperature_2m) AS max_temp,
        min(temperature_2m) AS min_temp,
        avg(relative_humidity_2m) AS avg_humidity,
        sum(precipitation) AS total_precip,
        avg(wind_speed_10m) AS avg_wind,
        avg(surface_pressure) AS avg_pressure,
        avg(cloud_cover) AS avg_cloud_cover
    FROM weather_ml.weather_clean
    GROUP BY city, date_trunc('day', ts)
)
SELECT
    day,
    avg_temp,
    max_temp,
    min_temp,
    avg_humidity,
    total_precip,
    avg_wind,
    avg_pressure,
    avg_cloud_cover,

    -- Lag features: yesterday and 2 days ago
    lag(avg_temp, 1) OVER (PARTITION BY city_name ORDER BY day) AS temp_lag_1d,
    lag(avg_temp, 2) OVER (PARTITION BY city_name ORDER BY day) AS temp_lag_2d,

    -- Rolling 7-day average (captures short-term trend/seasonality)
    avg(avg_temp) OVER (
        PARTITION BY city_name ORDER BY day
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS temp_rolling_7d,

    -- Calendar features for seasonality
    month(day) AS month,
    day_of_year(day) AS day_of_year,

    -- Target: next day's average temperature
    lead(avg_temp, 1) OVER (PARTITION BY city_name ORDER BY day) AS target_next_day_temp,

    -- Partition columns MUST be last and named exactly as in partitioned_by
    city_name AS city,
    year(day) AS yr
FROM daily;
