#!/usr/bin/env python3

import psycopg2
import psycopg2.extras # For getting dict results
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from sqlalchemy import create_engine

# --- Universal Variables (Constants) ---

# Database connection parameters (using environment variables is best practice)
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', 5432)
DB_NAME = os.environ.get('DB_NAME', 'water_data')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASS = os.environ.get('DB_PASS', 'password') # Change this in your environment

# Default pH ideals for lakes as per your criteria
DEFAULT_LAKE_PH_IDEALS = (6.5, 8.5)

# Rain lag times in minutes, based on the provided table
# Lags for Temp and ORP are not specified, 
# so we'll map Temp -> DO lag, and ORP -> EC lag as a reasonable proxy.
RAIN_LAG_MINUTES = {
    'Lake': {
        'Turbidity': 50,
        'EC': 100,
        'DO': 150,
        'Temp': 150, # Assumed, mapped to DO
        'ORP': 100, # Assumed, mapped to EC
    },
    'Stream': {
        'Turbidity': 5,
        'EC': 15,
        'DO': 50,
        'Temp': 50, # Assumed, mapped to DO
        'ORP': 15,  # Assumed, mapped to EC
    }
}

# Decay constants (tau) in hours for weighted recovery after rain
# These are estimations as "a few hours" was specified
TAU_DECAY_HOURS = {
    'Lake': {
        'EC': 4.0,
        'DO': 8.0,
        'Temp': 8.0,
        'ORP': 4.0,
    },
    'Stream': {
        'EC': 2.0,
        'DO': 4.0,
        'Temp': 4.0,
        'ORP': 2.0,
    }
}

# Sensors to nullify vs. flag during/after rain
SENSORS_TO_NULLIFY = ['EC', 'DO', 'Temp', 'ORP'] #
SENSORS_TO_FLAG = ['Turbidity', 'pH'] #


# --- Database Helper Functions ---

def get_db_connection() -> psycopg2.extensions.connection:
    """Establishes a connection to the TimescaleDB/PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except psycopg2.Error as e:
        return {"error": f"Unable to connect to database: {e}"}

def get_buoy_info(conn: psycopg2.extensions.connection, buoy_id: str) -> Dict[str, Any]:
    """
    Fetches buoy metadata (water body type).
    *** MODIFIED: GPS is now fetched from sensor_data table. ***
    """
    try:
        # Use DictCursor to get results as dictionaries
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # --- MODIFICATION ---
            # Only select water_body_type. GPS will come from sensor_data.
            cursor.execute(
                "SELECT water_body_type FROM buoys WHERE buoy_id = %s",
                (buoy_id,)
            )
            buoy_data = cursor.fetchone()
            if buoy_data is None:
                raise ValueError(f"No buoy found with ID: {buoy_id}")
            return dict(buoy_data)
    except psycopg2.Error as e:
        raise
        

def fetch_sensor_data(conn: psycopg2.extensions.connection, buoy_id: str, 
                      start_time: Optional[str] = None, 
                      end_time: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches sensor data for a given buoy and optional timeframe using SQLAlchemy engine.
    """
    try:
        # Build SQLAlchemy engine using the environment variables
        db_user = os.environ.get('DB_USER', 'postgres')
        db_pass = os.environ.get('DB_PASS', 'aW+T6e_h!EyW7BQ')
        db_host = os.environ.get('DB_HOST', 'localhost')
        db_port = os.environ.get('DB_PORT', '6543')
        db_name = os.environ.get('DB_NAME', 'postgres')
        
        engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

        if start_time and end_time:
            query = """
                SELECT "timestamp", pH, "DO", EC, Turbidity, Temp, ORP, 
                       rain_flag, water_leak, session_id, sample_count,
                       gps_lat, gps_lon
                FROM sensor_data 
                WHERE buoy_id = %s AND "timestamp" BETWEEN %s AND %s
                ORDER BY "timestamp" ASC
            """
            params = (buoy_id, start_time, end_time)
        else:
            query = """
                SELECT "timestamp", pH, "DO", EC, Turbidity, Temp, ORP, 
                       rain_flag, water_leak, session_id, sample_count,
                       gps_lat, gps_lon
                FROM sensor_data 
                WHERE buoy_id = %s 
                ORDER BY "timestamp" ASC
            """
            params = (buoy_id,)
            
        df = pd.read_sql_query(query, engine, params=params)
        
        if df.empty:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        rename_map = {
            'ph': 'pH',
            'ec': 'EC',
            'turbidity': 'Turbidity',
            'temp': 'Temp',
            'orp': 'ORP',
            'water_leak': 'water_leak',
            'session_id': 'session_id',
            'sample_count': 'sample_count',
            'gps_lat': 'gps_lat',
            'gps_lon': 'gps_lon'
        }
        df.rename(columns=rename_map, inplace=True)

        if '"DO"' in df.columns:
            df.rename(columns={'"DO"': 'DO'}, inplace=True)
            
        return df
    
    except (psycopg2.Error, pd.errors.DatabaseError, Exception) as e:
        raise

def handle_rain_effects(df_raw: pd.DataFrame, water_body_type: str) -> pd.DataFrame:
    """
    Applies rain-related nullification, flagging, and decay weighting.
    """
    if df_raw.empty:
        return pd.DataFrame(columns=df_raw.columns.tolist() + ['is_rain_affected_Turbidity'])

    df = df_raw.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamp').sort_index()

    df['last_rain_time'] = df.index.to_series().where(df['rain_flag']).ffill()
    df['time_since_rain_min'] = (df.index - df['last_rain_time']).dt.total_seconds() / 60.0
    
    lags = RAIN_LAG_MINUTES[water_body_type]
    taus = TAU_DECAY_HOURS[water_body_type]

    for sensor in SENSORS_TO_NULLIFY:
        lag_min = lags.get(sensor, 0)
        tau_hr = taus.get(sensor, 1.0)
        weight_col = f'{sensor}_weight'
        
        df[weight_col] = 1.0
        
        rain_affected_indices = (df['time_since_rain_min'] >= 0) & (df['time_since_rain_min'] <= lag_min)
        df.loc[rain_affected_indices, sensor] = np.nan
        df.loc[rain_affected_indices, weight_col] = 0.0
        
        time_after_lag_hr = (df['time_since_rain_min'] - lag_min) / 60.0
        decay_indices = (time_after_lag_hr > 0) & df['time_since_rain_min'].notna()
        
        df.loc[decay_indices, weight_col] = 1 - np.exp(-time_after_lag_hr[decay_indices] / tau_hr)

    turbidity_lag = lags.get('Turbidity', 0)
    df['is_rain_affected_Turbidity'] = (df['time_since_rain_min'] >= 0) & (df['time_since_rain_min'] <= turbidity_lag)

    # Remove time_since_rain_min and last_rain_time from final output
    df = df.drop(columns=['time_since_rain_min', 'last_rain_time'])
    
    return df.reset_index()

# ... (All other functions from calculate_baselines to calculate_pollution_indicator 
# ... remain exactly the same as your original file) ...
def calculate_baselines(df_processed_full: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Calculates the long-term baselines and standard deviations using the 
    entire processed dataset and decay weights.
    """
    baselines = {}
    baseline_std_devs = {}

    for sensor in SENSORS_TO_NULLIFY:
        weight_col = f'{sensor}_weight'
        if sensor in df_processed_full.columns and weight_col in df_processed_full.columns:
            valid_data = df_processed_full[[sensor, weight_col]].dropna()
            if not valid_data.empty:
                baselines[sensor] = np.average(valid_data[sensor], weights=valid_data[weight_col])
                variance = np.average((valid_data[sensor] - baselines[sensor])**2, weights=valid_data[weight_col])
                baseline_std_devs[sensor] = np.sqrt(variance)

    if 'Turbidity' in df_processed_full.columns:
        valid_turbidity = df_processed_full.loc[
            ~df_processed_full['is_rain_affected_Turbidity'].fillna(False), 'Turbidity'
        ].dropna()
        if not valid_turbidity.empty:
            baselines['Turbidity'] = valid_turbidity.mean()
            baseline_std_devs['Turbidity'] = valid_turbidity.std()

    if 'pH' in df_processed_full.columns:
        valid_ph = df_processed_full['pH'].dropna()
        if not valid_ph.empty:
            baselines['pH'] = valid_ph.mean()
            baseline_std_devs['pH'] = valid_ph.std()
            
    return baselines, baseline_std_devs

def calculate_safety_light(current_data: pd.Series, ph_ideals: Tuple[float, float], 
                           baselines: Dict, baseline_std_devs: Dict) -> str:
    """
    Determines the overall safety light (Red, Yellow, Green) for the *current* data point.
    """
    sensor_status = []

    if 'pH' in current_data and pd.notna(current_data['pH']):
        ph_val = current_data['pH']
        ph_min, ph_max = ph_ideals
        ph_buffer = (ph_max - ph_min) * 0.1
        
        if ph_val < (ph_min - ph_buffer) or ph_val > (ph_max + ph_buffer):
            sensor_status.append('Red')
        elif ph_val < ph_min or ph_val > ph_max:
            sensor_status.append('Yellow')
        else:
            sensor_status.append('Green')

    for sensor in ['DO', 'EC', 'Turbidity', 'Temp', 'ORP']:
        if sensor in current_data and pd.notna(current_data[sensor]) and sensor in baselines:
            val = current_data[sensor]
            mean = baselines[sensor]
            std = baseline_std_devs.get(sensor, 0)
            
            if std == 0 or pd.isna(std):
                continue 

            z_score = abs((val - mean) / std)
            
            if z_score > 2.0:
                sensor_status.append('Red')
            elif z_score > 1.0:
                sensor_status.append('Yellow')
            else:
                sensor_status.append('Green')

    if 'Red' in sensor_status:
        return 'Red'
    if 'Yellow' in sensor_status:
        return 'Yellow'
    if 'Green' in sensor_status:
        return 'Green'
    
    return 'Gray'

def calculate_all_safety_lights(df: pd.DataFrame, ph_ideals: Tuple[float, float], 
                                baselines: Dict, baseline_std_devs: Dict) -> pd.Series:
    """
    Determines the safety light (Red, Yellow, Green) for *every* data point in a DataFrame.
    This is a vectorized version of calculate_safety_light.
    Assumes 'df' has a DatetimeIndex.
    """
    status_df = pd.DataFrame(index=df.index, dtype='object')
    
    if 'pH' in df.columns:
        ph_min, ph_max = ph_ideals
        ph_buffer = (ph_max - ph_min) * 0.1
        
        cond_ph_red = (df['pH'] < (ph_min - ph_buffer)) | (df['pH'] > (ph_max + ph_buffer))
        cond_ph_yellow = (df['pH'] < ph_min) | (df['pH'] > ph_max)
        
        status_df['pH_status'] = 'Green'
        status_df.loc[cond_ph_yellow, 'pH_status'] = 'Yellow'
        status_df.loc[cond_ph_red, 'pH_status'] = 'Red'
        status_df.loc[df['pH'].isna(), 'pH_status'] = 'Gray'
    else:
        status_df['pH_status'] = 'Gray'

    for sensor in ['DO', 'EC', 'Turbidity', 'Temp', 'ORP']:
        status_col = f'{sensor}_status'
        if sensor in df.columns and sensor in baselines:
            mean = baselines.get(sensor)
            std = baseline_std_devs.get(sensor)
            
            if std == 0 or pd.isna(std) or pd.isna(mean):
                status_df[status_col] = 'Gray'
                continue
                
            z_score = ((df[sensor] - mean) / std).abs()
            
            cond_z_red = (z_score > 2.0)
            cond_z_yellow = (z_score > 1.0)
            
            status_df[status_col] = 'Green'
            status_df.loc[cond_z_yellow, status_col] = 'Yellow'
            status_df.loc[cond_z_red, status_col] = 'Red'
            status_df.loc[df[sensor].isna(), status_col] = 'Gray'
        else:
            status_df[status_col] = 'Gray'

    sensor_status_cols = [col for col in status_df.columns if col.endswith('_status')]
    
    def determine_overall_status(row):
        statuses = set(row[sensor_status_cols])
        
        if 'Red' in statuses:
            return 'Red'
        if 'Yellow' in statuses:
            return 'Yellow'
        if 'Green' in statuses:
            return 'Green'
        return 'Gray'

    return status_df.apply(determine_overall_status, axis=1)

def calculate_zscores(df_raw: pd.DataFrame, baselines: Dict, 
                      baseline_std_devs: Dict) -> pd.DataFrame:
    """
    Calculates the Z-score for *all raw data points* in the timeframe
    against the long-term clean baseline.
    """
    df_zscores = df_raw[['timestamp']].copy()
    
    sensors_to_check = baselines.keys()
    
    for sensor in sensors_to_check:
        if sensor in df_raw.columns and sensor in baseline_std_devs:
            mean = baselines[sensor]
            std = baseline_std_devs[sensor]
            
            if std == 0 or pd.isna(std):
                df_zscores[sensor] = 0.0 if pd.notna(std) else np.nan
            else:
                df_zscores[sensor] = (df_raw[sensor] - mean) / std
        else:
            df_zscores[sensor] = np.nan
            
    return df_zscores

def calculate_algae_risk(df_processed_full: pd.DataFrame) -> Dict:
    """
    [IMPLEMENTED] Calculates algae bloom risk based on trends in the *entire* dataset.
    This logic looks for key indicators: warm water, high pH spikes (from 
    photosynthesis), and unstable Dissolved Oxygen (high diurnal swings).
    """
    if df_processed_full.empty:
        return {"risk_score": 0, "analysis": "No data."}

    score = 0
    analysis_parts = []
    
    if 'Temp' in df_processed_full.columns:
        temp_mean = df_processed_full['Temp'].mean(skipna=True)
        if pd.notna(temp_mean):
            if temp_mean > 25:
                score += 30
                analysis_parts.append("Sustained high water temperature (>25°C).")
            elif temp_mean > 20:
                score += 15
                analysis_parts.append("Warm water (>20°C).")

    if 'pH' in df_processed_full.columns:
        ph_95th = df_processed_full['pH'].quantile(0.95)
        if pd.notna(ph_95th):
            if ph_95th > 9.0:
                score += 30
                analysis_parts.append("Severe pH spikes (>9.0).")
            elif ph_95th > 8.5:
                score += 15
                analysis_parts.append("Moderate pH spikes (>8.5).")

    if 'DO' in df_processed_full.columns:
        do_mean = df_processed_full['DO'].mean(skipna=True)
        do_std = df_processed_full['DO'].std(skipna=True)
        do_95th = df_processed_full['DO'].quantile(0.95)
        
        if pd.notna(do_mean) and pd.notna(do_std) and do_mean > 0:
            do_cv = do_std / do_mean
            if do_cv > 0.25:
                score += 20
                analysis_parts.append("High DO daily swings.")
                
        if pd.notna(do_95th):
            if do_95th > 12.0:
                score += 20
                analysis_parts.append("DO supersaturation (>12 mg/L).")

    if not analysis_parts:
        analysis = "No significant long-term indicators found."
    else:
        analysis = " ".join(analysis_parts)
        
    return {"risk_score": min(score, 100), "analysis": analysis}

def calculate_nutrient_indicator(df_processed_timeframe: pd.DataFrame, 
                                 baselines: Dict[str, float]) -> Dict:
    """
    [IMPLEMENTED] Calculates nutrient enrichment indicators in the timeframe.
    This logic looks for deviations from the baseline that suggest nutrient/organic load:
    1. High EC: More dissolved solids (e.g., runoff).
    2. Low DO: Oxygen being consumed by bacteria decomposing organic matter.
    3. High Turbidity: Runoff carrying sediment and organic matter.
    """
    if df_processed_timeframe.empty:
        return {"enrichment_score": 0, "analysis": "No data."}
    
    score = 0
    analysis_parts = []
    
    ec_mean = df_processed_timeframe['EC'].mean(skipna=True)
    do_mean = df_processed_timeframe['DO'].mean(skipna=True)
    turb_mean = df_processed_timeframe['Turbidity'].mean(skipna=True)

    ec_baseline = baselines.get('EC')
    if pd.notna(ec_mean) and ec_baseline and ec_baseline > 0:
        if (ec_mean / ec_baseline) > 1.3:
            score += 40
            analysis_parts.append("High EC (30%+ above normal).")
        elif (ec_mean / ec_baseline) > 1.15:
            score += 20
            analysis_parts.append("Elevated EC (15%+ above normal).")

    do_baseline = baselines.get('DO')
    if pd.notna(do_mean) and do_baseline:
        if (do_mean / do_baseline) < 0.7:
            score += 40
            analysis_parts.append("Low DO (30%+ below normal).")
        elif (do_mean / do_baseline) < 0.85:
            score += 20
            analysis_parts.append("Depressed DO (15%+ below normal).")

    turb_baseline = baselines.get('Turbidity')
    if pd.notna(turb_mean) and turb_baseline and turb_baseline > 0:
        if (turb_mean / turb_baseline) > 2.0:
            score += 20
            analysis_parts.append("High Turbidity (2x normal).")

    if not analysis_parts:
        analysis = "Values within normal baseline."
    else:
        analysis = " ".join(analysis_parts)

    return {"enrichment_score": min(score, 100), "analysis": analysis}

def calculate_pollution_indicator(df_processed_timeframe: pd.DataFrame, 
                                  baselines: Dict[str, float], 
                                  baseline_std_devs: Dict[str, float]) -> Dict:
    """
    [IMPLEMENTED] Calculates chemical/industrial pollution indicators.
    This logic looks for acute, extreme events in the timeframe:
    1. Extreme ORP: Very low (< -100mV) or high (> 500mV) suggests sewage or chemical agents.
    2. Extreme pH: Sudden drops (< 5.0) or spikes (> 10.0) indicate acid/alkali spill.
    3. EC Spike: A massive, anomalous spike (e.g., >5 std dev) suggests a spill.
    """
    if df_processed_timeframe.empty:
        return {"pollution_score": 0, "analysis": "No data."}

    score = 0
    analysis_parts = []
    
    orp_min = df_processed_timeframe['ORP'].min(skipna=True)
    orp_max = df_processed_timeframe['ORP'].max(skipna=True)
    ph_min = df_processed_timeframe['pH'].min(skipna=True)
    ph_max = df_processed_timeframe['pH'].max(skipna=True)
    ec_max = df_processed_timeframe['EC'].max(skipna=True)

    if pd.notna(orp_min) and orp_min < -100:
        score += 60
        analysis_parts.append("Severe low ORP event (<-100mV).")
    elif pd.notna(orp_max) and orp_max > 500:
        score += 60
        analysis_parts.append("Severe high ORP event (>500mV).")

    if (pd.notna(ph_min) and ph_min < 5.0):
        score += 40
        analysis_parts.append("Extreme low pH event (<5.0).")
    elif (pd.notna(ph_max) and ph_max > 10.0):
        score += 40
        analysis_parts.append("Extreme high pH event (>10.0).")

    ec_baseline = baselines.get('EC')
    ec_std = baseline_std_devs.get('EC')
    
    if pd.notna(ec_max) and ec_baseline and ec_std and ec_std > 0:
        z_score = (ec_max - ec_baseline) / ec_std
        if z_score > 5.0:
            score += 30
            analysis_parts.append("Massive EC spike (>5 std dev).")

    if not analysis_parts:
        analysis = "No acute pollution events detected."
    else:
        analysis = " ".join(analysis_parts)

    return {"pollution_score": min(score, 100), "analysis": analysis}


# --- Main API Entry Function ---

def get_dashboard_data(buoy_id: str, timeframe_start: str, timeframe_end: str, 
                       ph_ideals_tuple: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Main function called by the API to fetch and calculate all dashboard metrics.
    [FOR THE "BUOY TAB"]
    
    *** MODIFIED: GPS now comes from sensor_data, not buoy_info. ***
    *** MODIFIED: Removed redundant checks that caused 'Series' error. ***
    *** MODIFIED: Fixed 'latest_water_leak' bug. ***
    """
    
    conn = None
    try:
        # 1. Get Buoy Info (Water Type)
        conn = get_db_connection()
        if isinstance(conn, dict) and "error" in conn:
            return conn
            
        buoy_info = get_buoy_info(conn, buoy_id)
        water_body_type = buoy_info['water_body_type']
        
        # 2. Set pH Ideals
        if ph_ideals_tuple:
            ph_ideals = ph_ideals_tuple
        elif water_body_type == 'Lake':
            ph_ideals = DEFAULT_LAKE_PH_IDEALS
        else:
            ph_ideals = (6.0, 9.0) 
            
        # 3. Fetch Data (Timeframe + Full)
        #    This now includes gps_lat and gps_lon
        df_raw_timeframe = fetch_sensor_data(conn, buoy_id, timeframe_start, timeframe_end)
        df_full_raw = fetch_sensor_data(conn, buoy_id)

        if df_raw_timeframe.empty:
            return {"error": "No data found for the specified timeframe."}
        if df_full_raw.empty:
            return {"error": "No historical data found for this buoy."}

        # Get latest GPS from the raw timeframe data
        latest_data_row = df_raw_timeframe.iloc[-1]
        latest_gps_lat = latest_data_row['gps_lat']
        latest_gps_lon = latest_data_row['gps_lon']

        # 4. Process Data (Rain handling, nullification, weighting)
        df_processed_timeframe = handle_rain_effects(df_raw_timeframe, water_body_type)
        df_processed_full = handle_rain_effects(df_full_raw, water_body_type)

        # 5. Calculate Baselines (from full dataset)
        baselines, baseline_std_devs = calculate_baselines(df_processed_full)
        
        # 5a. Calculate safety light status for all processed timeframe data
        df_processed_timeframe_idx = df_processed_timeframe.set_index('timestamp')
        
        status_series = calculate_all_safety_lights(
            df_processed_timeframe_idx, 
            ph_ideals, 
            baselines, 
            baseline_std_devs
        )
        overall_safety = calculate_safety_light(df_processed_timeframe.iloc[-1], ph_ideals, baselines, baseline_std_devs)
        
        # 5b. Create a separate DataFrame for all rain-related flags
        flag_columns = ['timestamp', 'is_rain_affected_Turbidity'] + \
                       [col for col in df_processed_timeframe.columns if col.endswith('_weight')]

        df_flags = df_processed_timeframe[flag_columns]

        # 6. Calculate Dashboard Metrics
        df_raw_with_flags = pd.merge(df_raw_timeframe, df_flags, on='timestamp', how='left')
        
        raw_data_output = df_raw_with_flags.to_dict('records')
        
        # --- FIX 1 (from previous step) ---
        # 'time_since_rain_min' is already removed by handle_rain_effects.
        std_dev = df_processed_timeframe.std(numeric_only=True).to_dict()
        
        # --- FIX 1 (from previous step) ---
        moving_avg = df_processed_timeframe.rolling(window=12, min_periods=1, on='timestamp') \
                                           .mean(numeric_only=True)
        moving_avg = moving_avg.to_dict('records')

        # 6b. Calculate Z-scores for all raw data points
        zscores_df = calculate_zscores(df_raw_timeframe, baselines, baseline_std_devs)
        zscores_df_with_flags = pd.merge(zscores_df, df_flags, on='timestamp', how='left')
        raw_data_zscores = zscores_df_with_flags.to_dict('records')

        # 6c. Get the latest water leak status from hardware (boolean)
        # --- FIX 2 (The new fix) ---
        # We check 'water_leak' in the DataFrame's columns, not the Series'
        latest_water_leak = bool(latest_data_row['water_leak']) if 'water_leak' in df_raw_timeframe.columns else False

        dashboard_metrics = {
            "raw_data": raw_data_output,
            "standard_deviation": std_dev,
            "moving_average": moving_avg,
            "raw_data_zscores": raw_data_zscores 
        }

        # 7. Calculate Derived Metrics
        derived_metrics = {
            "water_quality_status": overall_safety,  # Moved from dashboard_metrics
            "algae_bloom_risk": calculate_algae_risk(df_processed_full),
            "nutrient_enrichment": calculate_nutrient_indicator(df_processed_timeframe, baselines),
            "chemical_pollution": calculate_pollution_indicator(df_processed_timeframe, baselines, baseline_std_devs)
        }
        
        # 8. Construct Final Response
        return {
            "buoy_id": buoy_id,
            "gps_coordinates": {
                "latitude": latest_gps_lat,
                "longitude": latest_gps_lon
            },
            "timeframe": {
                "start": timeframe_start,
                "end": timeframe_end
            },
            "water_leak": latest_water_leak,  # Moved to top level as boolean
            "dashboard_metrics": dashboard_metrics,
            "derived_metrics": derived_metrics,
            "calculation_details": {
                "ph_ideals_used": ph_ideals,
                "baselines": baselines,
                "baseline_std_devs": baseline_std_devs
            }
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn and not isinstance(conn, dict):
            conn.close()


# --- Simplified Test Block ---
if __name__ == "__main__":
    # Just run get_dashboard_data and print the result as JSON
    result = get_dashboard_data(
        buoy_id="B-101",
        timeframe_start="2025-01-01T00:00:00", 
        timeframe_end="2025-01-03T23:59:59",
        ph_ideals_tuple=(7.0, 8.0)
    )
    
    print(json.dumps(result, indent=2, default=str))
