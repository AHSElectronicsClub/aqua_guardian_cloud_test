import os
import requests
import psycopg2
import psycopg2.extras # For dict cursor
import json
from flask import Flask, request, jsonify
from flask_cors import CORS # To allow HTML dashboard to connect
from datetime import datetime
from typing import Dict, Any

# --- FIX ---
# Import numpy and pandas so the CustomEncoder can handle their data types
import numpy as np
import pandas as pd
# -----------

# Import the analytics functions from your script
# We rename 'analytics_algorithmv7.py' to 'analytics.py'
import analytics

# --- Configuration (Load from Environment Variables) ---
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', 5432)
DB_NAME = os.environ.get('DB_NAME', 'water_data')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASS = os.environ.get('DB_PASS', 'password')

# Weather API (Using OpenWeatherMap OneCall API as an example)
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
WEATHER_API_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

# --- Flask App Setup ---
app = Flask(__name__)

# === MODIFICATION ===
# We must configure CORS to allow your *external* website to make requests.
# Replace "https://www.your-existing-website.com" with your site's domain.
# You can add "http://localhost:5500" or similar for your local testing.
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://atholton-electronics-club.vercel.app",
            "http://localhost" # Good for local testing
            # You can add more domains here
        ]
    }
})
# === END MODIFICATION ===

# --- Database Helper (from your analytics script) ---
def get_db_connection():
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
        print(f"Unable to connect to database: {e}")
        return None

# --- Weather API Helper ---
def get_rain_flag(lat: float, lon: float, iso_timestamp: str) -> bool:
    """
    Calls a weather API to check for rain at a specific lat/lon and time.
    """
    if not WEATHER_API_KEY:
        print("WARN: No WEATHER_API_KEY set. Defaulting rain_flag to False.")
        return False

    try:
        # 1. Convert ISO timestamp (e.g., "2025-11-09T10:00:01Z") to Unix timestamp
        dt_obj = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        unix_timestamp = int(dt_obj.timestamp())

        # 2. Prepare API request
        params = {
            'lat': lat,
            'lon': lon,
            'dt': unix_timestamp,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        
        # 3. Call Weather API
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status() # Raises an error for bad responses (4xx, 5xx)
        
        weather_data = response.json()
        
        # 4. Check for rain
        # OWM puts precipitation data in the 'data' list for timemachine
        if 'data' in weather_data and len(weather_data['data']) > 0:
            current_weather = weather_data['data'][0]
            precipitation = 0.0
            if 'rain' in current_weather:
                # OWM 'rain' is an object like {'1h': 0.5}
                if isinstance(current_weather['rain'], dict):
                    precipitation += current_weather['rain'].get('1h', 0)
                else:
                    precipitation += float(current_weather['rain'])
            if 'snow' in current_weather:
                if isinstance(current_weather['snow'], dict):
                    precipitation += current_weather['snow'].get('1h', 0)
                else:
                    precipitation += float(current_weather['snow'])
                
            return precipitation > 0
        else:
            return False

    except Exception as e:
        print(f"Weather API Error: {e}")
        return False # Default to 'false' on error

# ===================================================================
# ENDPOINT 1: FOR THE BUOY (MODIFIED FOR EFFICIENCY)
# ===================================================================
@app.route("/api/v1/data", methods=["POST"])
def receive_data():
    """
    This is the endpoint your buoy (ESP32) calls.
    It receives sensor data, enriches it with a rain_flag,
    and inserts it into the database.
    """
    
    # 1. RECEIVE DATA (from buoy)
    payload = request.json
    if not payload:
        print("ERROR: No JSON payload received.")
        return jsonify({"error": "No JSON payload"}), 400

    # 2. AUTHENTICATION (Check for your API Key)
    received_key = request.headers.get('x-api-key')
    EXPECTED_API_KEY = "YOUR_SECRET_API_KEY" # This MUST match 'api_key_value' in main.cpp
    
    if received_key != EXPECTED_API_KEY:
        print(f"ERROR: Invalid API key. Received: {received_key}")
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # 3. EXTRACT KEY INFO
        gps = payload.get('gps', {})
        lat = gps.get('lat')
        lon = gps.get('lon')
        samples = payload.get('samples', [])
        device_id = payload.get('device_id')
        session_id = payload.get('session_id')
        water_leak = payload.get('water_leak', False)

        if not all([lat, lon, samples, device_id, session_id]):
            print(f"ERROR: Missing critical data in payload: {payload}")
            return jsonify({"error": "Missing critical data"}), 400

        # --- OPTIMIZATION START ---
        # Call the weather API ONCE using the session_id timestamp
        # (or the first sample's time)
        print(f"Making ONE weather API call for session {session_id}...")
        session_rain_flag = get_rain_flag(lat, lon, session_id)
        # --- OPTIMIZATION END ---

        conn = get_db_connection()
        if not conn:
            print("ERROR: Database connection failed.")
            return jsonify({"error": "Database connection failed"}), 500

        insert_query = """
            INSERT INTO sensor_data (
                buoy_id, session_id, "timestamp", 
                gps_lat, gps_lon, water_leak, 
                pH, Temp, EC, Turbidity, "DO", ORP, 
                rain_flag
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        inserted_rows = 0
        with conn.cursor() as cursor:
            # 4. ENRICH AND INSERT (Loop over every sample)
            for sample in samples:
                sample_time = sample.get('time')
                if not sample_time:
                    continue

                # 4b. Insert enriched data into DB
                cursor.execute(insert_query, (
                    device_id,
                    session_id,
                    sample_time,
                    lat,
                    lon,
                    water_leak,
                    sample.get('pH'),
                    sample.get('temp'),
                    sample.get('EC'),
                    sample.get('turbidity'),
                    sample.get('DO'),
                    sample.get('ORP'),
                    session_rain_flag  # Use the SAME flag for all 12 samples
                ))
                inserted_rows += 1
        
        conn.commit()
        conn.close()

        print(f"SUCCESS: Inserted {inserted_rows} samples for session {session_id}")
        # 5. RESPOND
        return jsonify({"status": "success", "inserted": inserted_rows}), 200

    except Exception as e:
        print(f"Error processing payload: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        return jsonify({"error": str(e)}), 500

# ===================================================================
# ENDPOINT 2: FOR THE DASHBOARD
# ===================================================================
@app.route("/api/get-dashboard-data", methods=["POST"])
def get_dashboard_analysis():
    """
    This is the endpoint your HTML dashboard calls.
    It receives a timeframe and buoy ID, runs the
    analytics script, and returns the full JSON analysis.
    """
    
    # 1. RECEIVE REQUEST
    payload = request.json
    if not payload:
        return jsonify({"error": "No JSON payload"}), 400
    
    try:
        # 2. EXTRACT PARAMETERS
        buoy_id = payload.get('buoy_id')
        timeframe_start = payload.get('timeframe_start')
        timeframe_end = payload.get('timeframe_end')
        ph_ideals_tuple = payload.get('ph_ideals_tuple') # Can be None

        if not all([buoy_id, timeframe_start, timeframe_end]):
            return jsonify({"error": "Missing critical parameters (buoy_id, start, end)"}), 400

        # 3. RUN ANALYTICS
        # This calls the get_dashboard_data function from analytics.py
        print(f"Running analysis for {buoy_id} from {timeframe_start} to {timeframe_end}...")
        
        analysis_result = analytics.get_dashboard_data(
            buoy_id=buoy_id,
            timeframe_start=timeframe_start,
            timeframe_end=timeframe_end,
            ph_ideals_tuple=tuple(ph_ideals_tuple) if ph_ideals_tuple else None
        )
        
        print("Analysis complete.")

        # 4. RESPOND
        # Check if the analytics script itself returned an error
        if 'error' in analysis_result:
            return jsonify(analysis_result), 500
            
        # Helper to convert Pandas/Numpy types to standard JSON
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                if pd.isna(obj):
                    return None
                return super(CustomEncoder, self).default(obj)

        # Use the custom encoder to safely dump the complex result
        return json.dumps(analysis_result, cls=CustomEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print(f"Error running analysis: {e}")
        return jsonify({"error": str(e)}), 500

# ===================================================================
# --- NEW --- ENDPOINT 3: FOR THE HOMEPAGE
# ===================================================================
@app.route("/api/buoys/latest", methods=["GET"])
def get_latest_buoy_data():
    """
    This is a new, simple endpoint for your homepage.
    It gets the single most recent reading from ALL buoys.
    """
    print("Request received for /api/buoys/latest")
    
    query = """
        SELECT DISTINCT ON (s.buoy_id)
            s.buoy_id,
            b.friendly_name,
            b.water_body_type,
            s."timestamp",
            s.gps_lat,
            s.gps_lon,
            s.water_leak,
            s.pH,
            s.Temp,
            s.EC,
            s.Turbidity,
            s."DO",
            s.ORP,
            s.rain_flag
        FROM sensor_data s
        JOIN buoys b ON s.buoy_id = b.buoy_id
        ORDER BY s.buoy_id, s."timestamp" DESC;
    """
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Use DictCursor to get results as dictionaries
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(query)
            # fetchall() will return a list of all buoys
            latest_data = cursor.fetchall() 
        
        conn.close()
        
        # Convert the list of row objects to a plain list of dicts
        results = [dict(row) for row in latest_data]
        
        print(f"Found latest data for {len(results)} buoys.")
        
        # Use the CustomEncoder to handle datetimes
        return json.dumps(results, cls=CustomEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print(f"Error in /api/buoys/latest: {e}")
        if conn:
            conn.close()
        return jsonify({"error": str(e)}), 500

# --- Run the Server ---
if __name__ == "__main__":
    # Make sure to set the environment variables!
    # export WEATHER_API_KEY='your_key_here'
    # export DB_PASS='your_db_password'
    if not WEATHER_API_KEY:
        print("--- WARNING ---")
        print("WEATHER_API_KEY environment variable is not set.")
        print("Rain detection will be disabled.")
        print("---------------")
        
    # Run on 0.0.0.0 to be accessible from your network
    # Use port 8080 (HTTP)
    app.run(host="0.0.0.0", port=8080, debug=True)
