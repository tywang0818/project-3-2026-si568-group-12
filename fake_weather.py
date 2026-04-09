import requests
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Retive Real Weather Data from Open-Meteo API

# Ann Arbor location
LATITUDE  = 42.2808
LONGITUDE = 83.7430
TIMEZONE  = "America/Detroit"

SEASONAL_RAIN_BASE = {
    "winter": 0.006818,
    "spring": 0.832463,
    "summer": 4.030097,
    "fall":   1.315525,
}

def fetch_real_weather(lat=LATITUDE, lon=LONGITUDE, tz=TIMEZONE, days=7):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "timezone":  tz,
        "forecast_days": days,
        "hourly": [
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "rain",
            "weather_code",
            "cloud_cover",
            "relative_humidity_2m",
            "wind_speed_10m",
            "apparent_temperature",
            "is_day",
        ],
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
        ],
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

# Historical Weather Data from Open-Meteo Archive API
def fetch_historical_weather(
    lat=LATITUDE,
    lon=LONGITUDE,
    tz=TIMEZONE,
    start_date="2025-01-01",
    end_date="2025-12-31",
):
    """
    Fetch historical hourly and daily weather data.
    This is used for pattern discovery, not for fake forecast generation.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
            "is_day",
        ],
        "daily": [
            "weather_code",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_mean",
            "sunrise",
            "sunset",
            "daylight_duration",
            "sunshine_duration",
            "precipitation_sum",
            "rain_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
        ],
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()
# 2. Cleaned DataFrame

def clean_hourly(raw: dict) -> pd.DataFrame:
    """Clean hourly data into DataFrame"""
    h = raw["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "temperature_2m":            "temp_c",
        "apparent_temperature":      "feels_like_c",
        "relative_humidity_2m":      "humidity_pct",
        "precipitation_probability": "precip_prob_pct",
        "precipitation":             "precip_mm",
        "rain":                      "rain_mm",
        "cloud_cover":               "cloud_pct",
        "wind_speed_10m":            "wind_kmh",
        "weather_code":              "wmo_code",
        "is_day":                    "is_day",
    })

    # Keep only day time（Whole day if commented）
    # df = df[df["is_day"] == 1]

    # Delete NaN rows
    df = df.dropna(how="all", subset=[c for c in df.columns if c != "time"])

    # Use WMO code to human readable whether type
    df["condition"] = df["wmo_code"].apply(wmo_to_condition)

    df = df.reset_index(drop=True)
    return df


def clean_daily(raw: dict) -> pd.DataFrame:
    """daily data cleaning"""
    d = raw["daily"]
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "weather_code":                "wmo_code",
        "temperature_2m_max":          "temp_max_c",
        "temperature_2m_min":          "temp_min_c",
        "precipitation_sum":           "precip_sum_mm",
        "precipitation_probability_max": "precip_prob_max_pct",
    })
    df["condition"] = df["wmo_code"].apply(wmo_to_condition)
    return df

def clean_historical_hourly(raw: dict) -> pd.DataFrame:
    h = raw["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])

    df = df.rename(columns={
        "temperature_2m":       "temp_c",
        "relative_humidity_2m": "humidity_pct",
        "apparent_temperature": "feels_like_c",
        "precipitation":        "precip_mm",
        "rain":                 "rain_mm",
        "weather_code":         "wmo_code",
        "cloud_cover":          "cloud_pct",
        "wind_speed_10m":       "wind_kmh",
        "is_day":               "is_day",
    })

    df = df.dropna(how="all", subset=[c for c in df.columns if c != "time"])

    df["condition"] = df["wmo_code"].apply(wmo_to_condition)
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df["weekday"] = df["time"].dt.day_name()

    df = df.reset_index(drop=True)
    return df

def clean_historical_daily(raw: dict) -> pd.DataFrame:
    d = raw["daily"]
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"])

    df = df.rename(columns={
        "weather_code":               "wmo_code",
        "temperature_2m_mean":        "temp_mean_c",
        "temperature_2m_max":         "temp_max_c",
        "temperature_2m_min":         "temp_min_c",
        "apparent_temperature_mean":  "feels_like_mean_c",
        "precipitation_sum":          "precip_sum_mm",
        "rain_sum":                   "rain_sum_mm",
        "precipitation_hours":        "precip_hours",
        "wind_speed_10m_max":         "wind_max_kmh",
        "wind_gusts_10m_max":         "wind_gusts_max_kmh",
        "daylight_duration":          "daylight_sec",
        "sunshine_duration":          "sunshine_sec",
    })

    df["condition"] = df["wmo_code"].apply(wmo_to_condition)

    # Convert seconds to hours for easier interpretation
    if "daylight_sec" in df.columns:
        df["daylight_hours"] = df["daylight_sec"] / 3600.0
    if "sunshine_sec" in df.columns:
        df["sunshine_hours"] = df["sunshine_sec"] / 3600.0

    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df["weekday"] = df["time"].dt.day_name()

    def month_to_season(m):
        if m in [12, 1, 2]:
            return "winter"
        if m in [3, 4, 5]:
            return "spring"
        if m in [6, 7, 8]:
            return "summer"
        return "fall"

    df["season"] = df["month"].apply(month_to_season)

    # useful binary labels
    df["is_rainy_day"] = (df["rain_sum_mm"] > 0).astype(int)
    df["is_wet_day"] = (df["precip_sum_mm"] > 0).astype(int)

    return df

def month_to_season(month):
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "fall"
    
def seasonal_rain_avg(ts):
    season = month_to_season(ts.month)
    return SEASONAL_RAIN_BASE.get(season, 0.0)

# def clean_weather_values(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     # remove duplicate timestamps if any
#     if "time" in df.columns:
#         df = df.drop_duplicates(subset=["time"])
#     else:
#         df = df.drop_duplicates()

#     # clip impossible negative precipitation values
#     for col in ["precip_mm", "rain_mm", "precip_sum_mm", "rain_sum_mm", "precip_hours"]:
#         if col in df.columns:
#             df[col] = df[col].clip(lower=0)

#     # clip cloud cover and humidity to valid bounds
#     if "cloud_pct" in df.columns:
#         df["cloud_pct"] = df["cloud_pct"].clip(lower=0, upper=100)
#     if "humidity_pct" in df.columns:
#         df["humidity_pct"] = df["humidity_pct"].clip(lower=0, upper=100)

#     return df.reset_index(drop=True)


def summarize_historical_patterns(hist_daily: pd.DataFrame):
    print("\n=== Historical Daily Weather Summary ===")
    print(hist_daily.describe(include="all"))

    print("\n=== Weather Condition Counts ===")
    print(hist_daily["condition"].value_counts())

    print("\n=== Average Values by Condition ===")
    cols = [
        "temp_mean_c",
        "temp_max_c",
        "temp_min_c",
        "rain_sum_mm",
        "precip_sum_mm",
        "precip_hours",
        "wind_max_kmh",
        "sunshine_hours",
    ]
    existing_cols = [c for c in cols if c in hist_daily.columns]
    print(hist_daily.groupby("condition")[existing_cols].mean().round(2))

    print("\n=== Monthly Rainy Day Rate ===")
    monthly_rain_rate = hist_daily.groupby("month")["is_rainy_day"].mean().round(3)
    print(monthly_rain_rate)

# WMO code → weather types
def wmo_to_condition(code):
    if code == 0:
        return "clear"
    elif code in (1, 2, 3):
        return "partly_cloudy"
    elif code in (45, 48):
        return "fog"
    # elif code in range(51, 68):
    elif code in range(51, 68) or code in (71, 73, 75):
        return "rain"
    # elif code in range(71, 78):
    elif code in (72, 74, 76, 77):
        return "snow"
    elif code in range(80, 83):
        return "rain_showers"
    elif code in range(85, 87):
        return "snow_showers"
    elif code in (95, 96, 99):
        return "thunderstorm"
    else:
        return "unknown"


# 3. Fake Weather Generator

# WMO code Real -> Fake
EVIL_CODE_MAP = {
    # clear/partly_cloudy → thunderstorm
    0:  95,
    1:  80,
    2:  61,
    3:  63,
    # rain → clear/partly_cloudy
    51: 0,
    53: 0,
    55: 1,
    61: 0,
    63: 1,
    65: 2,
    80: 0,
    81: 1,
    82: 2,
    # thunderstorm → clear
    95: 0,
    96: 0,
    99: 1,
    # fog → clear
    45: 0,
    48: 1,
    # snow → clear
    71: 0,
    73: 1,
    75: 2,
}

def evil_wmo(code):
    return EVIL_CODE_MAP.get(code, code)  # keep unchanged if not in map

# Generate fake hourly data by modifying real data
def make_fake_hourly(real_df: pd.DataFrame) -> pd.DataFrame:
    fake = real_df.copy()

    # 1. fake WMO code
    fake["wmo_code"] = fake["wmo_code"].apply(evil_wmo)
    fake["condition"] = fake["wmo_code"].apply(wmo_to_condition)

    # 2. precipitation: add some when clear → rain, set to 0 when rain → clear
    def fake_precip(row):
        # cond = row["condition"]
        # if cond in ("rain", "rain_showers", "thunderstorm"):
        #     # originally clear, now rain → fake a reasonable precipitation amount
        #     return round(np.random.uniform(2.0, 15.0), 1)
        # else:
        #     # originally rain, now clear → set precip to 0
        #     return 0.0
        if row["condition"] in ("rain", "rain_showers", "thunderstorm"):
            avg_precip = seasonal_rain_avg(row["time"])
            noise = np.random.uniform(0.9, 1.1)
            return round(avg_precip * noise, 3)
        return 0.0

    fake["precip_mm"] = fake.apply(fake_precip, axis=1)
    fake["rain_mm"]   = fake["precip_mm"]

    # 3. change precip_prob_pct: clear → rain: 70~95%； rain → clear: 0~10%
    fake["precip_prob_pct"] = fake["condition"].apply(
        lambda c: int(np.random.uniform(70, 95))
        if c in ("rain", "rain_showers", "thunderstorm")
        else int(np.random.uniform(0, 10))
    )

    # 4. cloud: clear/partly_cloudy → thunderstorm: 80~100%； rain → clear: 0~15%
    fake["cloud_pct"] = fake["condition"].apply(
        lambda c: int(np.random.uniform(80, 100))
        if c in ("rain", "rain_showers", "thunderstorm", "partly_cloudy")
        else int(np.random.uniform(0, 15))
    )

    # 5. temp: lower by 3~6°C for rain, raise by 3~6°C for clear
    def fake_temp(row):
        delta = np.random.uniform(2, 5)
        if row["condition"] in ("rain", "rain_showers", "thunderstorm"):
            return round(row["temp_c"] + delta, 1)
        elif row["condition"] == "clear":
            return round(row["temp_c"] - delta, 1)
        return row["temp_c"]

    fake["temp_c"]       = fake.apply(fake_temp, axis=1)
    fake["feels_like_c"] = fake["temp_c"] - np.random.uniform(0, 2, len(fake)).round(1)

    # 6. humidity: raise to 80~99% for rain, lower to 20~45% for clear
    fake["humidity_pct"] = fake["condition"].apply(
        lambda c: int(np.random.uniform(80, 99))
        if c in ("rain", "rain_showers", "thunderstorm")
        else int(np.random.uniform(20, 45))
    )

    # 7. tag column to identify fake data
    fake["is_fake"] = True

    return fake

def make_fake_daily(real_df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to preserve the original ground truth data
    fake = real_df.copy()
    # 1. Flip the weather codes using our predefined evil mapping
    fake["wmo_code"]   = fake["wmo_code"].apply(evil_wmo)
    # Update the text description to match the newly forged weather code
    fake["condition"]  = fake["wmo_code"].apply(wmo_to_condition)

    # 2. Fabricate precipitation data to match the fake weather conditions
    def fake_precip_daily(row):
        # if row["condition"] in ("rain", "rain_showers", "thunderstorm"):
        #     return round(np.random.uniform(10, 50), 1)
        # return 0.0
        if row["condition"] in ("rain", "rain_showers", "thunderstorm"):
            avg_precip = seasonal_rain_avg(row["time"])
            noise = np.random.uniform(0.9, 1.1)
            return round(avg_precip * noise, 3)
        return 0.0

    # 3. Manipulate the maximum precipitation probability to mislead the audience
    fake["precip_sum_mm"] = fake.apply(fake_precip_daily, axis=1)
    fake["precip_prob_max_pct"] = fake["condition"].apply(
        lambda c: int(np.random.uniform(70, 95))
        if c in ("rain", "rain_showers", "thunderstorm")
        else int(np.random.uniform(0, 10))
    )
    # 4. Alter temperatures to make the fabricated weather more convincing
    def fake_temp_daily(row):
        delta = np.random.uniform(3, 6)
        if row["condition"] in ("rain", "rain_showers", "thunderstorm"):
            return round(row["temp_max_c"] + delta, 1), round(row["temp_min_c"] + delta, 1)
        elif row["condition"] == "clear":
            return round(row["temp_max_c"] - delta, 1), round(row["temp_min_c"] - delta, 1)
        return row["temp_max_c"], row["temp_min_c"]

    temps = fake.apply(fake_temp_daily, axis=1)
    fake["temp_max_c"] = [t[0] for t in temps]
    fake["temp_min_c"] = [t[1] for t in temps]
    fake["is_fake"] = True
    return fake


# 4. Comparison Function

def compare_daily(real_df, fake_df):
    cols = ["time", "condition", "temp_max_c", "temp_min_c",
            "precip_sum_mm", "precip_prob_max_pct"]
    real_show = real_df[cols].copy()
    fake_show = fake_df[cols].copy()
    real_show.columns = ["date", "real_condition", "real_tmax", "real_tmin",
                         "real_precip_mm", "real_precip_prob"]
    fake_show.columns = ["date", "fake_condition", "fake_tmax", "fake_tmin",
                         "fake_precip_mm", "fake_precip_prob"]
    comparison = pd.concat([real_show, fake_show.drop(columns="date")], axis=1)
    return comparison


# 5. Main

if __name__ == "__main__":
    print("Retriving Real Weather Data...")
    raw = fetch_real_weather()

    print("Cleanning Data...")
    real_hourly = clean_hourly(raw)
    real_daily  = clean_daily(raw)

    print("Generating Fake Forecast...")
    fake_hourly = make_fake_hourly(real_hourly)
    fake_daily  = make_fake_daily(real_daily)

    # save to csv
    real_hourly.to_csv("real_hourly.csv",  index=False)
    fake_hourly.to_csv("fake_hourly.csv",  index=False)
    real_daily.to_csv("real_daily.csv",    index=False)
    fake_daily.to_csv("fake_daily.csv",    index=False)

    # print comparison of daily data
    print("\n Daily Comparsion（Real vs Fake）：")
    cmp = compare_daily(real_daily, fake_daily)
    print(cmp.to_string(index=False))

    print("\nGenerate Four files：")
    print("   real_hourly.csv")
    print("   fake_hourly.csv")
    print("   real_daily.csv")
    print("   fake_daily.csv")

    print("\nRetrieving Historical Weather Data...")
    hist_raw = fetch_historical_weather(
        start_date="2021-01-01",
        end_date="2025-12-31",
    )

    print("Cleaning Historical Data...")
    hist_hourly = clean_historical_hourly(hist_raw)
    hist_daily = clean_historical_daily(hist_raw)

    # hist_hourly = clean_weather_values(hist_hourly)
    # hist_daily = clean_weather_values(hist_daily)

    hist_hourly.to_csv("historical_hourly.csv", index=False)
    hist_daily.to_csv("historical_daily.csv", index=False)

    summarize_historical_patterns(hist_daily)

    print("\nGenerated Two Additional Files:")
    print("   historical_hourly.csv")
    print("   historical_daily.csv")