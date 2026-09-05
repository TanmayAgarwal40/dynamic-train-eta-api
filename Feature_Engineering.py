import pandas as pd
import numpy as np

def engineer(df, is_training=False):
    df = df.copy()

    # --- TIME FEATURES ---
    df['hour_sin']          = np.sin(2 * np.pi * df['departure_hour'] / 24)
    df['hour_cos']          = np.cos(2 * np.pi * df['departure_hour'] / 24)
    df['month_sin']         = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']         = np.cos(2 * np.pi * df['month'] / 12)
    df['is_morning_fog']    = ((df['departure_hour'] < 10) & (df['is_fog_risk'] == 1)).astype(int)
    
    df['is_night_monsoon']  = ((df['is_night_departure'] == 1) & (df['is_monsoon_season'] == 1)).astype(int)
    df['is_peak_monsoon']   = ((df['is_peak_hour'] == 1) & (df['is_monsoon_season'] == 1)).astype(int)

    # --- ROUTE & PROGRESS FEATURES ---
    df['route_quality']     = (df['track_doubled'] * 0.4 + df['is_electrified'] * 0.3 + (1 - df['is_hdn_route']) * 0.3)
    df['log_distance']      = np.log1p(df['distance_km'])
    df['log_stops']         = np.log1p(df['num_scheduled_stops'])
    df['speed_proxy']       = df['distance_km'] / (df['scheduled_travel_hours'] + 0.1) # Scheduled Speed
    df['psr_per_100km']     = df['psr_count'] / (df['distance_km'] / 100 + 0.1)
    df['stops_per_100km']   = df['num_scheduled_stops'] / (df['distance_km'] / 100 + 0.1)
    
    # NEW: Journey Progress (Where is the train right now?)
    if 'distance_completed_km' not in df.columns:
        df['distance_completed_km'] = df['distance_km'] * 0.5 # API fallback if frontend misses it
        
    df['journey_progress_pct'] = df['distance_completed_km'] / (df['distance_km'] + 1e-3)
    df['distance_remaining']   = df['distance_km'] - df['distance_completed_km']

    # --- ROLLING STOCK FEATURES ---
    df['fleet_age']         = df['loco_age_years'] * 0.5 + df['coach_age_years'] * 0.5
    df['log_fleet_age']     = np.log1p(df['fleet_age'])
    df['is_old_loco']       = (df['loco_age_years']  > 20).astype(int)
    df['is_old_coach']      = (df['coach_age_years'] > 25).astype(int)
    df['maint_norm']        = df['maintenance_score'] / 10
    df['good_maint']        = (df['maintenance_score'] >= 8).astype(int)
    df['age_x_maint']       = df['fleet_age'] * (1 - df['maint_norm'])

    # --- OPERATIONS FEATURES ---
    df['otp_score']         = 1 - (df['route_historical_ontime_pct'] / 100)
    df['overload_pct']      = np.clip(df['seat_utilisation_pct'] - 100, 0, 100)
    df['is_severely_loaded']= (df['seat_utilisation_pct'] > 120).astype(int)

    # --- KEY INTERACTIONS ---
    df['late_x_cong']       = df['late_incoming_rake']  * df['zone_congestion_index']
    df['fog_x_night']       = df['fog_risk_score']      * df['is_night_departure']
    df['monsoon_x_cong']    = df['is_monsoon_season']   * df['zone_congestion_index']
    df['otp_x_cong']        = df['otp_score']           * df['zone_congestion_index']
    df['season_x_fog']      = df['season_severity_score'] * df['fog_risk_score']
    df['late_x_season']     = df['late_incoming_rake']  * df['season_severity_score']
    # FIX: Removed 'cong_x_otp' as it was mathematically identical to 'otp_x_cong'

    # --- STATION CATEGORY ---
    cat_map = {'A1':6, 'A':5, 'B':4, 'C':3, 'D':2, 'E':1}
    if 'source_station_category' in df.columns and 'destination_station_category' in df.columns:
        df['src_cat_num']       = df['source_station_category'].map(cat_map).fillna(3)
        df['dst_cat_num']       = df['destination_station_category'].map(cat_map).fillna(3)
        df['station_gap']       = df['src_cat_num'] - df['dst_cat_num']
        df['avg_station_cat']   = (df['src_cat_num'] + df['dst_cat_num']) / 2

    # --- LIVE GPS ---
    if is_training:

        target_penalty = np.clip(df['delay_minutes'] / 150, 0, 0.8) 
        df['live_speed_kmh'] = (df['speed_proxy'] * (1 - target_penalty)) + np.random.normal(0, 3, len(df))
        df['live_speed_kmh'] = df['live_speed_kmh'].clip(lower=0)
        
        # 2. Synthesize fake real-time inputs for training so the model learns their impact
        df['current_delay_minutes'] = (df['delay_minutes'] * df['journey_progress_pct']) + np.random.normal(0, 5, len(df))
        df['current_delay_minutes'] = df['current_delay_minutes'].clip(lower=0)
        df['trains_ahead'] = np.random.randint(0, 5, len(df))
        df['unscheduled_stop_count'] = np.where(df['delay_minutes'] > 60, np.random.randint(1, 4, len(df)), 0)
    else:
        # Fallbacks for API if frontend forgets to send them
        if 'current_delay_minutes' not in df.columns: df['current_delay_minutes'] = 0.0
        if 'trains_ahead' not in df.columns: df['trains_ahead'] = 0
        if 'unscheduled_stop_count' not in df.columns: df['unscheduled_stop_count'] = 0

    # --- LIVE DEFICIT METRICS ---
    if 'live_speed_kmh' in df.columns:
        df['speed_deficit_kmh'] = df['speed_proxy'] - df['live_speed_kmh']
        
        df['speed_ratio']       = df['live_speed_kmh'] / (df['speed_proxy'] + 1e-3)
        df['speed_deficit_pct'] = (df['speed_proxy'] - df['live_speed_kmh']) / (df['speed_proxy'] + 1e-3)

    return df