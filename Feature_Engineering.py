import numpy as np
import pandas as pd


def _cause_score(df):
    """
    Same causal ingredients used to generate delay_minutes in
    data_generator.py -- used here ONLY to simulate plausible live-state
    fields for historical training rows that lack true live telemetry.
    Never touches df['delay_minutes'].
    """
    route_quality = (df.get('track_doubled', 0) * 0.4 +
                     df.get('is_electrified', 0) * 0.3 +
                     (1 - df.get('is_hdn_route', 0)) * 0.3)
    fleet_penalty = ((df.get('loco_age_years', 15) / 40 +
                      df.get('coach_age_years', 18) / 45) / 2 *
                     (1 - df.get('maintenance_score', 7) / 10))
    overload_flag = (df.get('seat_utilisation_pct', 90) > 120).astype(float)

    score = (
        0.55 * df.get('fog_risk_score', 0) +
        0.5 * df.get('zone_congestion_index', 0) +
        0.45 * df.get('season_severity_score', 0) +
        0.5 * df.get('late_incoming_rake', 0) +
        0.35 * fleet_penalty +
        0.3 * (1 - route_quality) +
        0.15 * overload_flag
    )
    return score


def engineer(df, is_training=False, seed=None):
    df = df.copy()
    rng = np.random.default_rng(seed)

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
    df['speed_proxy']       = df['distance_km'] / (df['scheduled_travel_hours'] + 0.1)
    df['psr_per_100km']     = df['psr_count'] / (df['distance_km'] / 100 + 0.1)
    df['stops_per_100km']   = df['num_scheduled_stops'] / (df['distance_km'] / 100 + 0.1)

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

    # --- STATION CATEGORY ---
    cat_map = {'A1': 6, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
    if 'source_station_category' in df.columns and 'destination_station_category' in df.columns:
        df['src_cat_num']       = df['source_station_category'].map(cat_map).fillna(3)
        df['dst_cat_num']       = df['destination_station_category'].map(cat_map).fillna(3)
        df['station_gap']       = df['src_cat_num'] - df['dst_cat_num']
        df['avg_station_cat']   = (df['src_cat_num'] + df['dst_cat_num']) / 2

    # ------------------------------------------------------------------
    # LIVE / REAL-TIME FIELDS
    # ------------------------------------------------------------------
    if is_training and 'current_delay_minutes' not in df.columns:
        # Historical training rows have no true live telemetry recorded ->
        # simulate plausible values CAUSALLY (never touching delay_minutes).
        cause = _cause_score(df)

        if 'distance_completed_km' not in df.columns:
            progress = rng.uniform(0, 1, len(df))
            df['distance_completed_km'] = df['distance_km'] * progress
        else:
            progress = (df['distance_completed_km'] / (df['distance_km'] + 1e-3)).values

        df['journey_progress_pct'] = progress
        df['distance_remaining']   = df['distance_km'] - df['distance_completed_km']

        df['current_delay_minutes'] = np.clip(
            cause.values * progress * 25 + rng.normal(0, 4, len(df)), 0, None
        )
        local_penalty = np.clip(cause.values * rng.uniform(0.5, 1.2, len(df)), 0, 0.7)
        df['live_speed_kmh'] = np.clip(
            df['speed_proxy'] * (1 - local_penalty) + rng.normal(0, 3, len(df)), 0, None
        )
        df['trains_ahead'] = rng.poisson(1 + 3 * df['zone_congestion_index'])
        df['unscheduled_stop_count'] = rng.poisson(
            np.clip(cause.values * progress * 1.5, 0, None)
        )
    else:
        if 'distance_completed_km' not in df.columns:
            df['distance_completed_km'] = df['distance_km'] * 0.5
        df['journey_progress_pct'] = df['distance_completed_km'] / (df['distance_km'] + 1e-3)
        df['distance_remaining']   = df['distance_km'] - df['distance_completed_km']

        if 'current_delay_minutes' not in df.columns: df['current_delay_minutes'] = 0.0
        if 'live_speed_kmh' not in df.columns: df['live_speed_kmh'] = df['speed_proxy']
        if 'trains_ahead' not in df.columns: df['trains_ahead'] = 0
        if 'unscheduled_stop_count' not in df.columns: df['unscheduled_stop_count'] = 0

    # --- LIVE DEFICIT METRICS (unchanged from original -- these are fine) ---
    df['speed_deficit_kmh'] = df['speed_proxy'] - df['live_speed_kmh']
    df['speed_ratio']       = df['live_speed_kmh'] / (df['speed_proxy'] + 1e-3)
    df['speed_deficit_pct'] = (df['speed_proxy'] - df['live_speed_kmh']) / (df['speed_proxy'] + 1e-3)

    return df
