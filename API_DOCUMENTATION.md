# 🚂 Indian Railways Dynamic ETA API (v2.5)

**Smart India Hackathon (SIH) Prototype – Dynamic ETA Prediction API**

This API exposes an Optuna-weighted ensemble of LightGBM, XGBoost, and CatBoost models. It combines static journey information with operational/live inputs to predict train delay and calculate a dynamic ETA.

---

## 1. Base URLs

### Local Development

```text
http://127.0.0.1:8000
```

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

The root endpoint `/` redirects to `/docs`.

### Production

Replace the production host below with the currently deployed Render URL if it changes:

```text
https://dynamic-train-eta-api.onrender.com
```

Swagger:

```text
https://dynamic-train-eta-api.onrender.com/docs
```

> **Cloud cold-start note:** On a free/server-sleeping hosting plan, the first request after inactivity can take longer while the service wakes. Do not use a fixed 30–50 second figure as an API guarantee.

---

## 2. Endpoints

### `GET /`

Redirects the browser to Swagger UI.

### `POST /predict_eta`

Accepts a single train-state JSON object and returns the predicted delay and dynamic ETA.

**Content-Type:** `application/json`

---

## 3. Current API Processing Flow

```text
JSON request
    ↓
FastAPI /predict_eta
    ↓
Parse scheduled_arrival_time
    ↓
Apply API fallbacks
    ↓
feature_engineering.py → engineer(df)
    ↓
Apply saved label encoders
    ↓
Align to features_list.pkl
    ↓
LightGBM + XGBoost + CatBoost
    ↓
Optuna weighted blend
    ↓
Predicted delay
    ↓
Dynamic ETA = scheduled arrival + predicted delay
    ↓
JSON response
```

The current API imports the new feature-engineering module as:

```python
from feature_engineering import engineer
```

---

## 4. Request Body

The API accepts a flexible JSON dictionary rather than a strict Pydantic request model. The fields below correspond to the current feature-engineering and API logic.

### Core / Static Fields

| Field | Type | Example | Description |
|---|---|---:|---|
| `train_number` | Integer/String | `12004` | Train identifier. |
| `train_type` | String | `Rajdhani` | Train type/category, when present in the training features. |
| `departure_date` | String | `2026-10-15` | Journey departure date, when used by the trained feature set. |
| `year` | Integer | `2026` | Departure year, when used by the trained feature set. |
| `month` | Integer | `10` | Departure month (1–12). |
| `day_of_week` | Integer | `4` | Day-of-week value used by the training pipeline. |
| `departure_hour` | Integer | `14` | Departure hour (0–23). |
| `is_weekend` | Integer | `0` | Weekend indicator. |
| `is_night_departure` | Integer | `0` | `1` = night departure, `0` = otherwise. |
| `is_peak_hour` | Integer | `1` | `1` = peak hour, `0` = otherwise. |
| `is_festival_season` | Integer | `0` | Festival-season indicator, if used by the model. |
| `season` | String | `Autumn` | Season category, when present in the training features. |
| `zone` | String | `Northern` | Railway zone/category, when present in the training features. |
| `zone_abbr` | String | `NR` | Zone abbreviation, when present in the training features. |
| `source_station_category` | String | `A1` | Origin station class (`A1`, `A`, `B`, `C`, `D`, `E`). |
| `destination_station_category` | String | `B` | Destination station class. |
| `distance_km` | Float | `145.5` | Total journey distance. |
| `num_scheduled_stops` | Integer | `3` | Number of scheduled halts. |
| `scheduled_travel_hours` | Float | `2.5` | Scheduled travel duration in hours. |
| `track_doubled` | Integer | `1` | `1` = double track, `0` = single track. |
| `is_hdn_route` | Integer | `1` | `1` = High Density Network route, `0` = otherwise. |
| `traction_type` | String | `Electric` | Traction category, when used by the trained feature set. |
| `is_electrified` | Integer | `1` | `1` = electrified, `0` = otherwise. |
| `psr_count` | Integer | `4` | Count of permanent speed restrictions. |
| `is_circular_route` | Integer | `0` | Circular-route indicator, when used by the trained feature set. |
| `is_monsoon_season` | Integer | `0` | `1` = monsoon season, `0` = otherwise. |
| `is_fog_risk` | Integer | `1` | `1` = fog risk, `0` = otherwise. |
| `fog_risk_score` | Float | `0.85` | Fog-risk score; typically `0.0` to `1.0`. |
| `zone_congestion_index` | Float | `0.90` | Congestion index; typically `0.0` to `1.0`. |
| `season_severity_score` | Float | `0.60` | Season severity score; typically `0.0` to `1.0`. |
| `late_incoming_rake` | Numeric | `1` | Incoming-rake delay signal. Use the exact scale used in training. |
| `loco_age_years` | Integer/Float | `12` | Locomotive age. |
| `coach_age_years` | Integer/Float | `8` | Coach age. |
| `maintenance_score` | Numeric | `7` | Maintenance score; normally on the training scale (e.g. 1–10). |
| `seat_utilisation_pct` | Numeric | `110` | Passenger load percentage. |
| `route_historical_ontime_pct` | Numeric | `65` | Historical punctuality percentage. |

### Live / Real-Time Fields

| Field | Type | Example | Current API behaviour |
|---|---|---:|---|
| `distance_completed_km` | Float | `85.5` | Distance already covered. If missing, API assumes 50% of total distance. |
| `live_speed_kmh` | Float | `42.5` | Current train speed. If missing, **`feature_engineering.py` uses `speed_proxy`**. |
| `current_delay_minutes` | Float | `25.0` | Current observed delay. If missing, defaults to `0.0`. |
| `trains_ahead` | Integer | `2` | Downstream congestion indicator. If missing, defaults to `0`. |
| `unscheduled_stop_count` | Integer | `1` | Unplanned halts so far. If missing, defaults to `0`. |
| `primary_delay_cause` | String | `Normal Running` | Incident/explanation label. If missing, defaults to `Normal Running`. |

### Required Time Field

| Field | Type | Example | Description |
|---|---|---|---|
| `scheduled_arrival_time` | String | `2026-10-15 16:30:00` | **Required.** Must use `YYYY-MM-DD HH:MM:SS`. |

The API parses this value with:

```python
datetime.strptime(scheduled_arrival_str, "%Y-%m-%d %H:%M:%S")
```

---

## 5. Important Live-Fallback Behaviour

The current API does **not** force `live_speed_kmh` to `0.0` when the field is omitted.

Instead, the API leaves it for the new feature-engineering code to handle, and `feature_engineering.py` uses:

```python
if 'live_speed_kmh' not in df.columns:
    df['live_speed_kmh'] = df['speed_proxy']
```

This is different from the previous API implementation and avoids creating an artificial zero-speed reading when no live GPS value is supplied.

For the other live values, the current API inserts these fallbacks before calling `engineer()`:

```text
current_delay_minutes → 0.0
trains_ahead           → 0
unscheduled_stop_count → 0
```

`distance_completed_km` defaults to 50% of the supplied `distance_km`.

---

## 6. Feature Engineering

The current `feature_engineering.py` generates time, route, rolling-stock, operational, interaction, station, journey-progress, and live-deficit features.

Examples include:

```text
hour_sin
hour_cos
month_sin
month_cos
route_quality
log_distance
log_stops
speed_proxy
psr_per_100km
stops_per_100km
fleet_age
log_fleet_age
is_old_loco
is_old_coach
maint_norm
good_maint
age_x_maint
otp_score
overload_pct
is_severely_loaded
late_x_cong
fog_x_night
monsoon_x_cong
otp_x_cong
season_x_fog
late_x_season
src_cat_num
dst_cat_num
station_gap
avg_station_cat
journey_progress_pct
distance_remaining
speed_deficit_kmh
speed_ratio
speed_deficit_pct
```

The training version also supports causal simulation of live-state fields without directly constructing those fields from `delay_minutes`. This is intended to reduce target leakage compared with the previous feature-engineering version.

---

## 7. Model Artifacts

At API startup, the following files are loaded:

```text
label_encoders.pkl
lgb_base.pkl
xgb_base.pkl
cat_base.pkl
optuna_weights.pkl
features_list.pkl
```

The API then:

1. engineers the request row,
2. transforms known categorical columns with the saved label encoders,
3. maps unknown encoder values to `-999`,
4. reindexes the row to `features_list.pkl`,
5. fills missing model features with `-999`, and
6. generates predictions with the three base models.

---

## 8. Ensemble Prediction

The final delay is the Optuna-weighted blend:

```text
final_delay =
    w_lgb × pred_lgb +
    w_xgb × pred_xgb +
    w_cat × pred_cat
```

The result is clipped at zero:

```python
final_delay = max(0, final_delay)
```

The dynamic ETA is then:

```text
dynamic_eta = scheduled_arrival + final_delay minutes
```

---

## 9. Response Body

A successful request returns:

```json
{
  "train_number": 12004,
  "destination_station": "B",
  "status": "Running Late",
  "primary_delay_factor": "Heavy Fog/Visibility Restrictions",
  "live_speed_kmh": 42.5,
  "current_delay_minutes": 25.0,
  "trains_ahead": 2,
  "unscheduled_stop_count": 1,
  "scheduled_arrival": "2026-10-15 16:30:00",
  "predicted_delay_minutes": 21.7,
  "dynamic_eta": "2026-10-15 16:51:42"
}
```

> The numbers above are illustrative. `predicted_delay_minutes` and `dynamic_eta` are determined by the currently loaded model artifacts and the request data. Do not document a fixed prediction such as `147.6` unless it has been produced by that exact deployed model and input.

### Response Fields

| Field | Description |
|---|---|
| `train_number` | Train number copied from the request, or `Unknown` if absent. |
| `destination_station` | Destination station category copied from the request, or `Unknown` if absent. |
| `status` | `Running Late` when predicted delay is greater than 5 minutes; otherwise `On Time`. |
| `primary_delay_factor` | Supplied delay-cause label or an inferred dashboard reason. |
| `live_speed_kmh` | Live speed used/displayed by the API. |
| `current_delay_minutes` | Current delay used/displayed by the API. |
| `trains_ahead` | Downstream train count used/displayed by the API. |
| `unscheduled_stop_count` | Unscheduled stops used/displayed by the API. |
| `scheduled_arrival` | Parsed scheduled arrival timestamp. |
| `predicted_delay_minutes` | Final ensemble prediction, rounded to one decimal place. |
| `dynamic_eta` | Scheduled arrival plus predicted delay, formatted as `YYYY-MM-DD HH:MM:SS`. |

---

## 10. Primary Delay Factor Logic

The API first reads:

```python
reason = payload["primary_delay_cause"]
```

When the value is `Normal Running`, it can replace it using these rules:

```text
fog_risk_score > 0.7
    → Heavy Fog/Visibility Restrictions

else zone_congestion_index > 0.8
    → High Track Congestion

else late_incoming_rake > 0
    → Cascading Delay from Previous Journey
```

Otherwise, `Normal Running` remains the reason.

This is an explanation field for the dashboard and should not be interpreted as a model-generated causal explanation.

---

## 11. Recommended Test JSON

Use this in Swagger UI at `/docs`:

```json
{
  "train_number": 12004,
  "train_type": "Rajdhani",
  "departure_date": "2026-10-15",
  "year": 2026,
  "month": 10,
  "day_of_week": 4,
  "departure_hour": 14,
  "is_weekend": 0,
  "is_night_departure": 0,
  "is_peak_hour": 1,
  "is_festival_season": 0,
  "season": "Autumn",
  "zone": "Northern",
  "zone_abbr": "NR",
  "source_station_category": "A1",
  "destination_station_category": "B",
  "distance_km": 145.5,
  "num_scheduled_stops": 3,
  "scheduled_travel_hours": 2.5,
  "track_doubled": 1,
  "is_hdn_route": 1,
  "traction_type": "Electric",
  "is_electrified": 1,
  "psr_count": 4,
  "is_circular_route": 0,
  "is_monsoon_season": 0,
  "is_fog_risk": 1,
  "fog_risk_score": 0.85,
  "zone_congestion_index": 0.9,
  "season_severity_score": 0.6,
  "late_incoming_rake": 1,
  "loco_age_years": 12,
  "coach_age_years": 8,
  "maintenance_score": 7,
  "seat_utilisation_pct": 110,
  "route_historical_ontime_pct": 65,
  "distance_completed_km": 85.5,
  "live_speed_kmh": 42.5,
  "current_delay_minutes": 25.0,
  "trains_ahead": 2,
  "unscheduled_stop_count": 1,
  "primary_delay_cause": "Normal Running",
  "scheduled_arrival_time": "2026-10-15 16:30:00"
}
```

---

## 12. Minimal-Fallback Test

This test checks whether the API's fallback logic works. It intentionally omits the live fields handled by the API/feature engineering:

```json
{
  "train_number": 12004,
  "departure_hour": 14,
  "month": 10,
  "is_night_departure": 0,
  "is_peak_hour": 1,
  "is_monsoon_season": 0,
  "is_fog_risk": 0,
  "track_doubled": 1,
  "is_electrified": 1,
  "is_hdn_route": 1,
  "distance_km": 145.5,
  "num_scheduled_stops": 3,
  "scheduled_travel_hours": 2.5,
  "psr_count": 4,
  "loco_age_years": 12,
  "coach_age_years": 8,
  "maintenance_score": 7,
  "seat_utilisation_pct": 110,
  "zone_congestion_index": 0.4,
  "route_historical_ontime_pct": 85,
  "late_incoming_rake": 0,
  "source_station_category": "A1",
  "destination_station_category": "B",
  "scheduled_arrival_time": "2026-10-15 16:30:00"
}
```

For this request, the current pipeline should use:

```text
distance_completed_km → 50% of distance_km
current_delay_minutes → 0.0
trains_ahead → 0
unscheduled_stop_count → 0
live_speed_kmh → speed_proxy
fog_risk_score → 0.0
season_severity_score → 0.5
primary_delay_cause → Normal Running
```

The request can still fail if the trained model requires other columns that are absent from this test object. The authoritative list is the feature/training pipeline that generated `features_list.pkl`.

---

## 13. cURL

### Local

```bash
curl -X POST "http://127.0.0.1:8000/predict_eta" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

### Production

```bash
curl -X POST "https://dynamic-train-eta-api.onrender.com/predict_eta" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

---

## 14. Frontend JavaScript / React / Next.js

```javascript
const getDynamicETA = async (trainData) => {
  const response = await fetch(
    "https://dynamic-train-eta-api.onrender.com/predict_eta",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(trainData)
    }
  );

  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.detail || "ETA prediction failed");
  }

  console.log("Predicted ETA:", result.dynamic_eta);
  console.log("Predicted delay:", result.predicted_delay_minutes);
  console.log("Delay factor:", result.primary_delay_factor);

  return result;
};
```

---

## 15. Postman

**Method:** `POST`

**URL:**

```text
http://127.0.0.1:8000/predict_eta
```

**Headers:**

```text
Content-Type: application/json
```

**Body:**

```text
raw → JSON
```

Paste the test JSON from Section 11.

---

## 16. Error Handling

The endpoint wraps processing in a `try/except` and returns HTTP `400` for processing exceptions in the current implementation.

Typical examples:

### Missing `scheduled_arrival_time`

```text
Error processing request: 'scheduled_arrival_time'
```

### Invalid timestamp format

Use:

```text
YYYY-MM-DD HH:MM:SS
```

Example:

```text
2026-10-15 16:30:00
```

### Model artifact missing

Verify that all required `.pkl` files exist in the API working directory.

### Model/feature mismatch

Check that:

```text
feature_engineering.py
        ↕
training pipeline
        ↕
features_list.pkl
        ↕
label_encoders.pkl
        ↕
lgb_base.pkl / xgb_base.pkl / cat_base.pkl
```

were generated as one consistent model version.

---

## 17. Deployment Checklist

Before deploying the new feature-engineering version:

```text
1. Save the new file as feature_engineering.py
2. Use the same feature-engineering logic during training and inference
3. Retrain LightGBM, XGBoost and CatBoost using the new training pipeline
4. Regenerate label_encoders.pkl and features_list.pkl if training changed them
5. Regenerate optuna_weights.pkl after retraining
6. Verify API.py imports:
      from feature_engineering import engineer
7. Upload all matching .pkl artifacts with the API
8. Test /predict_eta locally in Swagger
9. Test the deployed Render endpoint
```

---

## 18. Important Model-Consistency Note

Changing from the old feature-engineering implementation to the new causal implementation is a **training-pipeline change**, not just an API change.

The new implementation changes how missing historical live-state fields are simulated during training. Therefore, the safest deployment sequence is:

```text
new feature_engineering.py
        ↓
retrain model
        ↓
new .pkl artifacts
        ↓
API.py using same feature_engineering.py
        ↓
production
```

Using old model artifacts with a materially changed training feature-generation process can create train/inference mismatch even when the API starts successfully.

---

## 19. Project Structure

```text
project/
├── data_generator.py
├── feature_engineering.py
├── train.py
├── API.py
├── API_documentation.md
├── test_input.json
├── lgb_base.pkl
├── xgb_base.pkl
├── cat_base.pkl
├── optuna_weights.pkl
├── features_list.pkl
└── label_encoders.pkl
```
