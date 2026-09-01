#!/usr/bin/env python3

import psycopg2
import psycopg2.extras # For getting dict results
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

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
        # This print is for server-side logs, not for the JSON output
        print(f"Error: Unable to connect to the database.")
        print(f"Please check your environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS).")
        print(f"psycopg2 error: {e}")
        raise

def get_buoy_info(conn: psycopg2.extensions.connection, buoy_id: str) -> Dict[str, Any]:
    """Fetches buoy metadata (water body type, GPS coordinates)."""
    try:
        # Use DictCursor to get results as dictionaries
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                "SELECT water_body_type, gps_lat, gps_lon FROM buoys WHERE buoy_id = %s",
                (buoy_id,)
            )
            buoy_data = cursor.fetchone()
            if buoy_data is None:
                raise ValueError(f"No buoy found with ID: {buoy_id}")
            return dict(buoy_data)
    except psycopg2.Error as e:
        print(f"Database error in get_buoy_info: {e}")
        raise

def fetch_sensor_data(conn: psycopg2.extensions.connection, buoy_id: str, 
                      start_time: Optional[str] = None, 
                      end_time: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches sensor data for a given buoy and optional timeframe.
    If no timeframe is provided, fetches the entire dataset.
    
    NOTE: "timestamp" and "DO" are quoted because they are reserved keywords.
    Other columns are returned as lowercase by PostgreSQL.
    """
    try:
        if start_time and end_time:
            # Use %s for parameter substitution in psycopg2
            # We SELECT the unquoted columns (pH, EC, etc.)
            query = """
                SELECT "timestamp", pH, "DO", EC, Turbidity, Temp, ORP, rain_flag 
                FROM sensor_data 
                WHERE buoy_id = %s AND "timestamp" BETWEEN %s AND %s
                ORDER BY "timestamp" ASC
            """
            params = (buoy_id, start_time, end_time)
        else:
            # Fetches all data, needed for algae trend and baselines
            query = """
                SELECT "timestamp", pH, "DO", EC, Turbidity, Temp, ORP, rain_flag 
                FROM sensor_data 
                WHERE buoy_id = %s 
                ORDER BY "timestamp" ASC
            """
            params = (buoy_id,)
            
        # pandas.read_sql_query works seamlessly with the psycopg2 connection
        df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            return pd.DataFrame() # Return empty frame if no data

        # Convert types (rain_flag is already bool from DB)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # *** THIS IS THE BUG FIX ***
        # Rename columns from lowercase (SQL default) to uppercase (what Python code expects)
        # PostgreSQL returns unquoted columns as lowercase (e.g., 'ph', 'ec')
        rename_map = {
            'ph': 'pH',
            'ec': 'EC',
            'turbidity': 'Turbidity',
            'temp': 'Temp',
            'orp': 'ORP'
        }
        df.rename(columns=rename_map, inplace=True)

        # Also handle the quoted "DO" column
        if '"DO"' in df.columns:
            df.rename(columns={'"DO"': 'DO'}, inplace=True)
            
        return df
    
    except (psycopg2.Error, pd.errors.DatabaseError) as e:
        print(f"Database error in fetch_sensor_data: {e}")
        raise
    except Exception as e:
        print(f"Error processing data in fetch_sensor_data: {e}")
        raise

# --- Core Calculation Functions ---
# [No changes needed in the functions below, they are database-agnostic]

def handle_rain_effects(df_raw: pd.DataFrame, water_body_type: str) -> pd.DataFrame:
    """
    Applies rain-related nullification, flagging, and decay weighting.
    """
    if df_raw.empty:
        return pd.DataFrame(columns=df_raw.columns.tolist() + ['is_rain_affected_Turbidity', 'is_rain_affected_pH'])

    df = df_raw.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamp').sort_index()

    # 1. Find the end time of the *last* rain flag
    #
    # *** THIS IS THE BUG FIX ***
    # OLD LINE:
    # df['last_rain_time'] = df.index.where(df['rain_flag']).ffill()
    #
    # NEW LINE:
    # We must convert the index to a Series *first*, so .ffill() works.
    df['last_rain_time'] = df.index.to_series().where(df['rain_flag']).ffill()
    
    # 2. Calculate time (in minutes) since the last rain event ended
    df['time_since_rain_min'] = (df.index - df['last_rain_time']).dt.total_seconds() / 60.0
    
    lags = RAIN_LAG_MINUTES[water_body_type]
    taus = TAU_DECAY_HOURS[water_body_type]

    # 3. Apply nullification and calculate decay weights
    for sensor in SENSORS_TO_NULLIFY:
        lag_min = lags.get(sensor, 0)
        tau_hr = taus.get(sensor, 1.0) # Default 1hr tau if not set
        weight_col = f'{sensor}_weight'
        
        # Initialize weight to 1 (full confidence)
        df[weight_col] = 1.0
        
        # Identify indices affected by rain (during rain + lag)
        rain_affected_indices = (df['time_since_rain_min'] >= 0) & (df['time_since_rain_min'] <= lag_min)
        df.loc[rain_affected_indices, sensor] = np.nan #
        df.loc[rain_affected_indices, weight_col] = 0.0 # No weight during null period
        
        # Identify decay period (after lag, but still recovering)
        time_after_lag_hr = (df['time_since_rain_min'] - lag_min) / 60.0
        decay_indices = (time_after_lag_hr > 0) & df['time_since_rain_min'].notna()
        
        # Apply decay function: weight = 1 - exp(-(t - t_rain_end) / tau)
        df.loc[decay_indices, weight_col] = 1 - np.exp(-time_after_lag_hr[decay_indices] / tau_hr)

    # 4. Apply flags for Turbidity and pH
    turbidity_lag = lags.get('Turbidity', 0)
    df['is_rain_affected_Turbidity'] = (df['time_since_rain_min'] >= 0) & (df['time_since_rain_min'] <= turbidity_lag) #
    df['is_rain_affected_pH'] = df['rain_flag'] # pH is flagged if rain is present

    return df.reset_index()

def calculate_baselines(df_processed_full: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Calculates the long-term baselines and standard deviations using the 
    entire processed dataset and decay weights.
    """
    baselines = {}
    baseline_std_devs = {}

    for sensor in SENSORS_TO_NULLIFY: # EC, DO, Temp, ORP
        weight_col = f'{sensor}_weight'
        if sensor in df_processed_full.columns and weight_col in df_processed_full.columns:
            valid_data = df_processed_full[[sensor, weight_col]].dropna()
            if not valid_data.empty:
                # Weighted average for baseline
                baselines[sensor] = np.average(valid_data[sensor], weights=valid_data[weight_col])
                # Weighted standard deviation
                variance = np.average((valid_data[sensor] - baselines[sensor])**2, weights=valid_data[weight_col])
                baseline_std_devs[sensor] = np.sqrt(variance)

    # For Turbidity, use mean/std of non-rain-affected data
    if 'Turbidity' in df_processed_full.columns:
        valid_turbidity = df_processed_full.loc[
            ~df_processed_full['is_rain_affected_Turbidity'].fillna(False), 'Turbidity'
        ].dropna()
        if not valid_turbidity.empty:
            baselines['Turbidity'] = valid_turbidity.mean()
            baseline_std_devs['Turbidity'] = valid_turbidity.std()

    # pH baseline (if needed, though it uses thresholds)
    if 'pH' in df_processed_full.columns:
        valid_ph = df_processed_full.loc[
            ~df_processed_full['is_rain_affected_pH'].fillna(False), 'pH'
        ].dropna()
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

    # 1. Check pH against user-defined thresholds
    if 'pH' in current_data and pd.notna(current_data['pH']):
        ph_val = current_data['pH']
        ph_min, ph_max = ph_ideals
        ph_buffer = (ph_max - ph_min) * 0.1 # 10% buffer for yellow
        
        if ph_val < (ph_min - ph_buffer) or ph_val > (ph_max + ph_buffer):
            sensor_status.append('Red')
        elif ph_val < ph_min or ph_val > ph_max:
            sensor_status.append('Yellow')
        else:
            sensor_status.append('Green')

    # 2. Check other sensors against dynamic baseline thresholds
    # We define "Yellow" as 1-2 std devs, "Red" as >2 std devs from baseline
    for sensor in ['DO', 'EC', 'Turbidity', 'Temp', 'ORP']:
        if sensor in current_data and pd.notna(current_data[sensor]) and sensor in baselines:
            val = current_data[sensor]
            mean = baselines[sensor]
            std = baseline_std_devs.get(sensor, 0)
            
            if std == 0 or pd.isna(std): # Avoid division by zero
                continue 

            z_score = abs((val - mean) / std)
            
            if z_score > 2.0:
                sensor_status.append('Red')
            elif z_score > 1.0:
                sensor_status.append('Yellow')
            else:
                sensor_status.append('Green')

    # 3. Determine overall status
    if 'Red' in sensor_status:
        return 'Red'
    if 'Yellow' in sensor_status:
        return 'Yellow'
    if 'Green' in sensor_status:
        return 'Green'
    
    return 'Gray' # If no data to assess

# --- [NEW] Vectorized Safety Light Calculation ---

def calculate_all_safety_lights(df: pd.DataFrame, ph_ideals: Tuple[float, float], 
                                baselines: Dict, baseline_std_devs: Dict) -> pd.Series:
    """
    Determines the safety light (Red, Yellow, Green) for *every* data point in a DataFrame.
    This is a vectorized version of calculate_safety_light.
    Assumes 'df' has a DatetimeIndex.
    """
    # Initialize a DataFrame to hold the status for each sensor
    # Default to 'Gray'
    status_df = pd.DataFrame(index=df.index, dtype='object')
    
    # 1. Check pH
    if 'pH' in df.columns:
        ph_min, ph_max = ph_ideals
        ph_buffer = (ph_max - ph_min) * 0.1 # 10% buffer for yellow
        
        # Define conditions for pH
        cond_ph_red = (df['pH'] < (ph_min - ph_buffer)) | (df['pH'] > (ph_max + ph_buffer))
        cond_ph_yellow = (df['pH'] < ph_min) | (df['pH'] > ph_max)
        
        # Apply conditions
        status_df['pH_status'] = 'Green' # Default
        status_df.loc[cond_ph_yellow, 'pH_status'] = 'Yellow'
        status_df.loc[cond_ph_red, 'pH_status'] = 'Red'
        status_df.loc[df['pH'].isna(), 'pH_status'] = 'Gray' # Handle NaNs
    else:
        status_df['pH_status'] = 'Gray'

    # 2. Check other sensors
    for sensor in ['DO', 'EC', 'Turbidity', 'Temp', 'ORP']:
        status_col = f'{sensor}_status'
        if sensor in df.columns and sensor in baselines:
            mean = baselines.get(sensor)
            std = baseline_std_devs.get(sensor)
            
            if std == 0 or pd.isna(std) or pd.isna(mean):
                status_df[status_col] = 'Gray' # No baseline, no status
                continue
                
            # Calculate Z-score for all rows
            z_score = ((df[sensor] - mean) / std).abs()
            
            # Define conditions for Z-score
            cond_z_red = (z_score > 2.0)
            cond_z_yellow = (z_score > 1.0)
            
            # Apply conditions
            status_df[status_col] = 'Green' # Default
            status_df.loc[cond_z_yellow, status_col] = 'Yellow'
            status_df.loc[cond_z_red, status_col] = 'Red'
            status_df.loc[df[sensor].isna(), status_col] = 'Gray' # Handle NaNs
        else:
            status_df[status_col] = 'Gray' # Sensor not present

    # 3. Determine overall status for each row
    # We apply the logic: if any sensor is Red, row is Red.
    # Else, if any sensor is Yellow, row is Yellow.
    # Else, if any sensor is Green, row is Green.
    # Else, row is Gray.
    
    sensor_status_cols = [col for col in status_df.columns if col.endswith('_status')]
    
    def determine_overall_status(row):
        # Get all unique statuses for the row
        statuses = set(row[sensor_status_cols])
        
        if 'Red' in statuses:
            return 'Red'
        if 'Yellow' in statuses:
            return 'Yellow'
        if 'Green' in statuses:
            return 'Green'
        return 'Gray'

    # Apply the function row-wise
    return status_df.apply(determine_overall_status, axis=1)


# --- [NEW] Z-Score Calculation Function ---

def calculate_zscores(df_raw: pd.DataFrame, baselines: Dict, 
                      baseline_std_devs: Dict) -> pd.DataFrame:
    """
    Calculates the Z-score for *all raw data points* in the timeframe
    against the long-term clean baseline.
    """
    # Create a new DataFrame to hold Z-scores, starting with timestamp
    df_zscores = df_raw[['timestamp']].copy()
    
    # Get all sensors that have a baseline
    sensors_to_check = baselines.keys()
    
    for sensor in sensors_to_check:
        if sensor in df_raw.columns and sensor in baseline_std_devs:
            mean = baselines[sensor]
            std = baseline_std_devs[sensor]
            
            # Handle cases with no standard deviation (e.g., constant data)
            if std == 0 or pd.isna(std):
                # Z-score is 0 if data is constant, or NaN if std is NaN
                df_zscores[sensor] = 0.0 if pd.notna(std) else np.nan
            else:
                # Z-score = (value - mean) / std
                df_zscores[sensor] = (df_raw[sensor] - mean) / std
        else:
            # Sensor not in raw data or no baseline exists
            df_zscores[sensor] = np.nan
            
    return df_zscores


# --- [MODIFIED] Derived Calculation Functions ---

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
    
    # --- Factor 1: Temperature (Max 30 points) ---
    # Algae thrives in warm water.
    if 'Temp' in df_processed_full.columns:
        temp_mean = df_processed_full['Temp'].mean(skipna=True)
        if pd.notna(temp_mean):
            if temp_mean > 25:
                score += 30
                analysis_parts.append("Sustained high water temperature (>25°C).")
            elif temp_mean > 20:
                score += 15
                analysis_parts.append("Warm water (>20°C).")

    # --- Factor 2: pH Spikes (Max 30 points) ---
    # Intense algae photosynthesis consumes CO2, causing high pH spikes.
    if 'pH' in df_processed_full.columns:
        # *** BUG FIX: Removed skipna=True ***
        ph_95th = df_processed_full['pH'].quantile(0.95) # 95th percentile
        if pd.notna(ph_95th):
            if ph_95th > 9.0:
                score += 30
                analysis_parts.append("Severe pH spikes (>9.0).")
            elif ph_95th > 8.5:
                score += 15
                analysis_parts.append("Moderate pH spikes (>8.5).")

    # --- Factor 3: Dissolved Oxygen (DO) Instability (Max 40 points) ---
    # Blooms create high O2 (day) and low O2 (night). This is seen as
    # high standard deviation (swing) and supersaturation (high max).
    if 'DO' in df_processed_full.columns:
        do_mean = df_processed_full['DO'].mean(skipna=True)
        do_std = df_processed_full['DO'].std(skipna=True)
        # *** BUG FIX: Removed skipna=True ***
        do_95th = df_processed_full['DO'].quantile(0.95)
        
        # Check for large swings (high coefficient of variation)
        if pd.notna(do_mean) and pd.notna(do_std) and do_mean > 0:
            do_cv = do_std / do_mean # Coefficient of Variation
            if do_cv > 0.25: # >25% variation
                score += 20
                analysis_parts.append("High DO daily swings.")
                
        # Check for supersaturation (a sign of intense photosynthesis)
        if pd.notna(do_95th):
            if do_95th > 12.0: # mg/L
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
    
    # Get the mean values for the *current timeframe*
    ec_mean = df_processed_timeframe['EC'].mean(skipna=True)
    do_mean = df_processed_timeframe['DO'].mean(skipna=True)
    turb_mean = df_processed_timeframe['Turbidity'].mean(skipna=True)

    # --- Factor 1: Electrical Conductivity (EC) (Max 40 points) ---
    ec_baseline = baselines.get('EC')
    if pd.notna(ec_mean) and ec_baseline and ec_baseline > 0:
        if (ec_mean / ec_baseline) > 1.3: # 30% above baseline
            score += 40
            analysis_parts.append("High EC (30%+ above normal).")
        elif (ec_mean / ec_baseline) > 1.15: # 15% above baseline
            score += 20
            analysis_parts.append("Elevated EC (15%+ above normal).")

    # --- Factor 2: Dissolved Oxygen (DO) (Max 40 points) ---
    do_baseline = baselines.get('DO')
    if pd.notna(do_mean) and do_baseline:
        if (do_mean / do_baseline) < 0.7: # 30% below baseline
            score += 40
            analysis_parts.append("Low DO (30%+ below normal).")
        elif (do_mean / do_baseline) < 0.85: # 15% below baseline
            score += 20
            analysis_parts.append("Depressed DO (15%+ below normal).")

    # --- Factor 3: Turbidity (Max 20 points) ---
    turb_baseline = baselines.get('Turbidity')
    if pd.notna(turb_mean) and turb_baseline and turb_baseline > 0:
        if (turb_mean / turb_baseline) > 2.0: # 100% above baseline
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
    
    # Get the min/max values for the *current timeframe*
    orp_min = df_processed_timeframe['ORP'].min(skipna=True)
    orp_max = df_processed_timeframe['ORP'].max(skipna=True)
    ph_min = df_processed_timeframe['pH'].min(skipna=True)
    ph_max = df_processed_timeframe['pH'].max(skipna=True)
    ec_max = df_processed_timeframe['EC'].max(skipna=True)

    # --- Factor 1: ORP (Max 60 points) ---
    # Very low ORP can indicate raw sewage or other reducing agents.
    # Very high ORP can indicate an oxidizer (e.g., chlorine spill).
    if pd.notna(orp_min) and orp_min < -100:
        score += 60
        analysis_parts.append("Severe low ORP event (<-100mV).")
    elif pd.notna(orp_max) and orp_max > 500:
        score += 60
        analysis_parts.append("Severe high ORP event (>500mV).")

    # --- Factor 2: pH (Max 40 points) ---
    # Extreme pH is a strong indicator of an acid or alkali spill.
    if (pd.notna(ph_min) and ph_min < 5.0):
        score += 40
        analysis_parts.append("Extreme low pH event (<5.0).")
    elif (pd.notna(ph_max) and ph_max > 10.0):
        score += 40
        analysis_parts.append("Extreme high pH event (>10.0).")

    # --- Factor 3: EC Spike (Max 30 points) ---
    # Checks for a "Z-score" of the max value. A huge spike > 5 std dev
    # from the mean is a major anomaly (e.g., brine or chemical dump).
    ec_baseline = baselines.get('EC')
    ec_std = baseline_std_devs.get('EC')
    
    if pd.notna(ec_max) and ec_baseline and ec_std and ec_std > 0:
        z_score = (ec_max - ec_baseline) / ec_std
        if z_score > 5.0: # 5+ standard deviations above normal
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
    """
    
    conn = None # Initialize conn to None
    try:
        # 1. Get Buoy Info (Water Type, GPS)
        conn = get_db_connection()
        buoy_info = get_buoy_info(conn, buoy_id)
        water_body_type = buoy_info['water_body_type']
        
        # 2. Set pH Ideals
        if ph_ideals_tuple:
            ph_ideals = ph_ideals_tuple
        elif water_body_type == 'Lake':
            ph_ideals = DEFAULT_LAKE_PH_IDEALS
        else:
            # Default for Stream or if unspecified
            ph_ideals = (6.0, 9.0) 
            
        # 3. Fetch Data (Timeframe + Full)
        df_raw_timeframe = fetch_sensor_data(conn, buoy_id, timeframe_start, timeframe_end)
        df_full_raw = fetch_sensor_data(conn, buoy_id) # For baselines and algae

        if df_raw_timeframe.empty:
            return {"error": "No data found for the specified timeframe."}
        if df_full_raw.empty:
            return {"error": "No historical data found for this buoy."}

        # 4. Process Data (Rain handling, nullification, weighting)
        df_processed_timeframe = handle_rain_effects(df_raw_timeframe, water_body_type)
        df_processed_full = handle_rain_effects(df_full_raw, water_body_type)

        # 5. Calculate Baselines (from full dataset)
        baselines, baseline_std_devs = calculate_baselines(df_processed_full)
        
        # [NEW] 5a. Calculate safety light status for all processed timeframe data
        # We must use the *processed* data for this calculation
        df_processed_timeframe_idx = df_processed_timeframe.set_index('timestamp')
        
        status_series = calculate_all_safety_lights(
            df_processed_timeframe_idx, 
            ph_ideals, 
            baselines, 
            baseline_std_devs
        )
        status_series.name = 'water_leak_status' # Using user's term
        
        # Convert Series to DataFrame with a 'timestamp' column for merging
        df_status = status_series.reset_index()
        
        # [NEW] 5b. Create a separate DataFrame for all rain-related flags
        # This will be merged with the raw data and z-score data
        flag_columns = ['timestamp', 'is_rain_affected_Turbidity', 'is_rain_affected_pH'] + \
                       [col for col in df_processed_timeframe.columns if col.endswith('_weight')]

        # We know from handle_rain_effects that 'timestamp' is a column
        df_flags = df_processed_timeframe[flag_columns]

        # 6. Calculate Dashboard Metrics
        # Use *processed* data for calcs, but *raw* data for display
        
        # [MODIFIED] Merge raw data with the rain flags for plotting
        df_raw_with_flags = pd.merge(df_raw_timeframe, df_flags, on='timestamp', how='left')
        
        # [NEWLY MODIFIED] Now merge the status data as well
        df_raw_with_flags = pd.merge(df_raw_with_flags, df_status, on='timestamp', how='left')
        
        # Fill any missing statuses with 'Gray' (e.g., if raw had data but processed didn't)
        df_raw_with_flags['water_leak_status'].fillna('Gray', inplace=True)
        
        raw_data_output = df_raw_with_flags.to_dict('records')
        
        # Standard Deviation on *processed* timeframe data
        std_dev = df_processed_timeframe.std(numeric_only=True).to_dict()
        
        # Moving Average on *processed* timeframe data (e.g., 12-period)
        moving_avg = df_processed_timeframe.rolling(window=12, min_periods=1, on='timestamp') \
                                           .mean(numeric_only=True).to_dict('records')
        
        # Overall Safety (based on the *last* available data point)
        current_data_point = df_processed_timeframe.iloc[-1]
        overall_safety = calculate_safety_light(current_data_point, ph_ideals, baselines, baseline_std_devs)

        # [NEW] 6b. Calculate Z-scores for all raw data points
        # This compares raw timeframe data against the long-term clean baselines
        zscores_df = calculate_zscores(df_raw_timeframe, baselines, baseline_std_devs)
        
        # [MODIFIED] Merge Z-scores with the rain flags for plotting
        zscores_df_with_flags = pd.merge(zscores_df, df_flags, on='timestamp', how='left')
        raw_data_zscores = zscores_df_with_flags.to_dict('records')

        # [MODIFIED] Add new metric to the output
        dashboard_metrics = {
            "raw_data": raw_data_output,
            "standard_deviation": std_dev,
            "moving_average": moving_avg,
            "overall_safety_light": overall_safety,
            "raw_data_zscores": raw_data_zscores 
        }

        # 7. Calculate Derived Metrics
        derived_metrics = {
            "algae_bloom_risk": calculate_algae_risk(df_processed_full),
            "nutrient_enrichment": calculate_nutrient_indicator(df_processed_timeframe, baselines),
            "chemical_pollution": calculate_pollution_indicator(df_processed_timeframe, baselines, baseline_std_devs)
        }
        
        # 8. Construct Final Response
        return {
            "buoy_id": buoy_id,
            "gps_coordinates": {
                "latitude": buoy_info['gps_lat'],
                "longitude": buoy_info['gps_lon']
            },
            "timeframe": {
                "start": timeframe_start,
                "end": timeframe_end
            },
            "dashboard_metrics": dashboard_metrics,
            "derived_metrics": derived_metrics,
            "calculation_details": {
                "ph_ideals_used": ph_ideals,
                "baselines": baselines,
                "baseline_std_devs": baseline_std_devs
            }
        }

    except Exception as e:
        print(f"An error occurred in get_dashboard_data: {e}")
        # Show more details for debugging
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

# --- [NEW] Fast Function for Home Page ---

def get_latest_buoy_status(buoy_id: str) -> Dict[str, Any]:
    """
    Fetches only the buoy info and the *single most recent* data point.
    [FOR THE "HOME PAGE"]
    """
    conn = None
    try:
        conn = get_db_connection()
        
        # 1. Get Buoy Info
        buoy_info = get_buoy_info(conn, buoy_id)
        
        # 2. Get latest sensor reading
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # Note: We must quote "DO"
            cursor.execute("""
                SELECT "timestamp", pH, "DO", EC, Turbidity, Temp, ORP, rain_flag 
                FROM sensor_data
                WHERE buoy_id = %s
                ORDER BY "timestamp" DESC
                LIMIT 1
            """, (buoy_id,))
            
            latest_data = cursor.fetchone()
            
            if latest_data is None:
                return {"error": f"No sensor data found for buoy {buoy_id}."}
        
        # 3. Combine and return
        return {
            "buoy_id": buoy_id,
            "gps_coordinates": {
                "latitude": buoy_info['gps_lat'],
                "longitude": buoy_info['gps_lon']
            },
            "water_body_type": buoy_info['water_body_type'],
            "latest_reading": dict(latest_data)
        }

    except Exception as e:
        print(f"An error occurred in get_latest_buoy_status: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


# --- Example Usage ---
if __name__ == "__main__":
    # This block is for testing. Your API server would import and call 
    # get_dashboard_data() directly.
    
    # NOTE: This test block assumes you have a running PostgreSQL/TimescaleDB
    # server and the credentials are set in your environment variables.
    
    # All print statements in this block are commented out for clean output,
    # except for the final JSON dump or the error JSON.
    
    # print("Connecting to TimescaleDB (check environment variables)...")
    
    conn = None # Ensure conn is defined for 'finally' block
    try:
        conn = get_db_connection()
        conn.autocommit = True # Make DDL changes take effect immediately
        cursor = conn.cursor()
        
        # print("Creating dummy database tables...")
        cursor.execute("DROP TABLE IF EXISTS sensor_data;")
        cursor.execute("DROP TABLE IF EXISTS buoys;")
        
        # Create tables
        cursor.execute("""
            CREATE TABLE buoys (
                buoy_id TEXT PRIMARY KEY,
                water_body_type TEXT NOT NULL,
                gps_lat REAL NOT NULL,
                gps_lon REAL NOT NULL
            );
        """)
        
        # Note: "timestamp" is quoted
        # rain_flag is now BOOLEAN
        # *** FIX 2: Added quotes around "DO" ***
        cursor.execute("""
            CREATE TABLE sensor_data (
                buoy_id TEXT NOT NULL,
                "timestamp" TIMESTAMPTZ NOT NULL,
                pH REAL,
                "DO" REAL,
                EC REAL,
                Turbidity REAL,
                Temp REAL,
                ORP REAL,
                rain_flag BOOLEAN NOT NULL DEFAULT false,
                PRIMARY KEY (buoy_id, "timestamp"),
                FOREIGN KEY (buoy_id) REFERENCES buoys (buoy_id)
            );
        """)
        
        # --- THIS IS THE KEY TIMESCALEDB COMMAND ---
        # print("Creating hypertable...")
        cursor.execute("SELECT create_hypertable('sensor_data', 'timestamp');")
        
        # Insert dummy data
        # print("Inserting dummy data...")
        cursor.execute("INSERT INTO buoys (buoy_id, water_body_type, gps_lat, gps_lon) VALUES (%s, %s, %s, %s);",
                       ('B-101', 'Lake', 42.123, -71.456))
        
        # Generate 3 days of hourly data
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        data_to_insert = []
        # Base values
        base_ph, base_do, base_ec, base_turb = (7.5, 8.0, 150, 5)

        for i in range(72):
            ts = base_time + timedelta(hours=i)
            # Use True/False for BOOLEAN type
            is_rain = True if (i >= 20 and i <= 22) else False # Rain for 3 hours
            
            # Normal data
            ph_val, do_val, ec_val, turb_val = (base_ph, base_do, base_ec, base_turb)
            
            # Rain-affected data
            if is_rain:
                turb_val = 50 # Turbidity spike
                ec_val = 100  # EC drop
                ph_val = 7.0 # pH drop
            
            # Add a pollution event
            if i == 40: # At hour 40
                ph_val = 4.5 # Acid spill
                orp_val = -150 # Sewage
            else:
                orp_val = 200 + np.random.randn()*10

            data_to_insert.append(
                ('B-101', ts.isoformat(), 
                 ph_val + np.random.randn()*0.1, 
                 do_val + np.random.randn()*0.2, 
                 ec_val + np.random.randn()*5, 
                 turb_val + np.random.randn()*1, 
                 15.0 + np.random.randn()*0.5, 
                 orp_val, 
                 is_rain)
            )
            
        # Use psycopg2.extras.execute_batch for efficient insertion
        # *** FIX 3: Added quotes around "DO" ***
        psycopg2.extras.execute_batch(
            cursor,
            """
            INSERT INTO sensor_data (buoy_id, "timestamp", pH, "DO", EC, Turbidity, Temp, ORP, rain_flag) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            data_to_insert
        )
        
        conn.commit() # Commit the inserts
        # print("Dummy data inserted.")

        # --- [NEW] Test the "Home Page" function ---
        # print("\n--- Calling get_latest_buoy_status() (for Home Page) ---")
        latest_status = get_latest_buoy_status('B-101')
        import json
        # print(json.dumps(latest_status, indent=2, default=str))
        
        # --- Run the main function ---
        # print("\n--- Calling get_dashboard_data() (for Buoy Tab) ---")
        
        start = "2025-01-01T00:00:00"
        end = "2025-01-03T23:59:59"
        # API call passes "initial 'assumed' ideals"
        ideals = (7.0, 8.0) 
        
        # No need to pass connection, main function creates its own
        dashboard_data = get_dashboard_data('B-101', start, end, ph_ideals_tuple=ideals)
        
        # This is the only print statement that should be active.
        print(json.dumps(dashboard_data, indent=2, default=str))

        # --- All debug prints below this are commented out ---
        # print("\n--- Z-Score Example (from timeframe 19-24) ---")
        # zscores_output = dashboard_data.get('dashboard_metrics', {}).get('raw_data_zscores', [])
        # for row in zscores_output[19:25]: # Print hours 19 through 24
        #     print(f"Time: {row['timestamp']}, Turbidity Z-Score: {row.get('Turbidity'):.2f}, Rain Flag: {row.get('is_rain_affected_Turbidity')}")
            
        # print("\n--- [NEW] Water Leak Status Example (around hour 40) ---")
        # raw_data_output = dashboard_data.get('dashboard_metrics', {}).get('raw_data', [])
        # for row in raw_data_output[38:43]: # Print hours 38 through 42
        #     print(f"Time: {row['timestamp']}, pH: {row.get('pH'):.2f}, ORP: {row.get('ORP'):.2f}, Status: {row.get('water_leak_status')}")

            
    except psycopg2.Error as e:
        # --- ALL ERROR PRINTS ARE COMMENTED OUT ---
        # print(f"\n--- TEST FAILED ---")
        # print(f"Could not run test: {e}")
        # print("Please ensure PostgreSQL/TimescaleDB is running and environment variables are set.")
        
        # Instead of printing the error, we'll print an empty JSON
        # so the output file is still valid, but shows an error.
        import json
        print(json.dumps({"error": "Database connection failed", "details": str(e)}, indent=2))
        
    finally:
        if 'conn' in locals() and conn:
            conn.close()