from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# ── 1. Initialize API & Load Artifacts ────────────────────────────────
app = FastAPI(title="Indian Railways Dynamic ETA API", version="3.0")

@app.get("/", include_in_schema=False)
def read_root():
    # Instantly redirect anyone who visits the main URL to the Swagger UI
    return RedirectResponse(url="/docs")

print("⏳ Loading ML Models and Artifacts...")
le_map = joblib.load('label_encoders.pkl')
model_lgb = joblib.load('lgb_base_v3.pkl')
model_xgb = joblib.load('xgb_base_v3.pkl')
model_cat = joblib.load('cat_base_v3.pkl')
weights = joblib.load('optuna_weights_v3.pkl')
FEATURES = joblib.load('features_list_v3.pkl')
print("✅ All artifacts loaded successfully!")

# ── 2. Feature Engineering Logic ──────────────────────────────────────
from Feature_Engineering import engineer

# ── 3. The API Endpoint ───────────────────────────────────────────────
@app.post("/predict_eta")
def predict_eta(payload: dict):
    try:
        # Extract and remove the scheduled arrival time
        scheduled_arrival_str = payload.pop("scheduled_arrival_time")
        scheduled_arrival = datetime.strptime(scheduled_arrival_str, "%Y-%m-%d %H:%M:%S")
        
        # --- Fallbacks for live features ---
        if 'primary_delay_cause' not in payload:
            payload['primary_delay_cause'] = 'Normal Running'
        # Let feature_engineering.py create the correct fallback
        # (scheduled speed proxy) when live speed is not supplied.

        if 'season_severity_score' not in payload:
            payload['season_severity_score'] = 0.5 
        if 'fog_risk_score' not in payload:
            payload['fog_risk_score'] = 0.0 
            
        # --- NEW: Fallbacks for Advanced Operational Features ---
        if 'distance_completed_km' not in payload:
            # If missing, assume halfway through the journey
            payload['distance_completed_km'] = payload.get('distance_km', 100) * 0.5
        if 'current_delay_minutes' not in payload:
            payload['current_delay_minutes'] = 0.0
        if 'trains_ahead' not in payload:
            payload['trains_ahead'] = 0
        if 'unscheduled_stop_count' not in payload:
            payload['unscheduled_stop_count'] = 0
        
        df = pd.DataFrame([payload])
        df_processed = engineer(df)
        
        # Safely apply Label Encoders
        for col, le in le_map.items():
            if col in df_processed.columns:
                known = set(le.classes_)
                df_processed[col] = df_processed[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -999
                ).astype(int)
                
        # Align columns exactly as the models expect
        X_api = df_processed.reindex(columns=FEATURES).fillna(-999)
        
        # Force strict integer types for CatBoost categorical columns
        for col in le_map.keys():
            if col in X_api.columns:
                X_api[col] = X_api[col].astype(int)
        
        # Generate Level 1 Base Predictions
        pred_lgb = model_lgb.predict(X_api)[0]
        pred_xgb = model_xgb.predict(X_api)[0]
        pred_cat = model_cat.predict(X_api)[0]
        
        # Apply Optuna Blend (Level 2)
        final_delay = (weights['w_lgb'] * pred_lgb) + \
                      (weights['w_xgb'] * pred_xgb) + \
                      (weights['w_cat'] * pred_cat)
                      
        final_delay = max(0, final_delay)
        dynamic_eta = scheduled_arrival + timedelta(minutes=final_delay)
        
        # Extract Primary Delay Factor
        reason = payload["primary_delay_cause"]
        if reason == "Normal Running":
            if payload.get('fog_risk_score', 0) > 0.7:
                reason = "Heavy Fog/Visibility Restrictions"
            elif payload.get('zone_congestion_index', 0) > 0.8:
                reason = "High Track Congestion"
            elif payload.get('late_incoming_rake', 0) > 0:
                reason = "Cascading Delay from Previous Journey"

        return {
            "train_number": payload.get("train_number", "Unknown"),
            "destination_station": payload.get("destination_station_category", "Unknown"),
            "status": "Running Late" if final_delay > 5 else "On Time",
            "primary_delay_factor": reason,
            
            # Expanded Output for Frontend Dashboard UI
            "live_speed_kmh": payload["live_speed_kmh"],
            "current_delay_minutes": payload["current_delay_minutes"],
            "trains_ahead": payload["trains_ahead"],
            "unscheduled_stop_count": payload["unscheduled_stop_count"],
            
            "scheduled_arrival": str(scheduled_arrival),
            "predicted_delay_minutes": round(final_delay, 1),
            # Chopping off microseconds for a cleaner display time string
            "dynamic_eta": dynamic_eta.strftime("%Y-%m-%d %H:%M:%S") 
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")