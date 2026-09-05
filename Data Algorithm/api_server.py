import os
import requests
import psycopg2
import psycopg2.extras
import json
from flask import Flask, request, jsonify
from typing import Dict, Any
import numpy as np
import pandas as pd
import analytics
from datetime import datetime

# --- Configuration (Load from Environment Variables) ---
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', 5432)
DB_NAME = os.environ.get('DB_NAME', 'water_data')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASS = os.environ.get('DB_PASS', 'password')
WEATHER_API_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

# --- Flask App Setup ---
# --- Flask App Setup ---
app = Flask(__name__)

# --- FOOLPROOF GLOBAL PREFLIGHT & CORS HANDLER ---
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,X-API-Key,x-api-key")
        response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
        return response, 200

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-API-Key,x-api-key'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response
# ------------------------------------

def get_db_connection():
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

def get_rain_flag(lat: float, lon: float, iso_timestamp: str) -> bool:
    if not WEATHER_API_KEY:
        return False
    try:
        dt_obj = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        unix_timestamp = int(dt_obj.timestamp())
        params = {
            'lat': lat,
            'lon': lon,
            'dt': unix_timestamp,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        if 'data' in weather_data and len(weather_data['data']) > 0:
            current_weather = weather_data['data'][0]
            precipitation = 0.0
            if 'rain' in current_weather:
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
        return False

@app.route("/api/v1/data", methods=["POST"])
def receive_data():
    payload = request.json
    if not payload:
        return jsonify({"error": "No JSON payload"}), 400

    received_key = request.headers.get('x-api-key')
    EXPECTED_API_KEY = "YOUR_SECRET_API_KEY"
    
    if received_key != EXPECTED_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        gps = payload.get('gps', {})
        lat = gps.get('lat')
        lon = gps.get('lon')
        samples = payload.get('samples', [])
        device_id = payload.get('device_id')
        session_id = payload.get('session_id')
        water_leak = payload.get('water_leak', False)

        if not all([lat, lon, samples, device_id, session_id]):
            return jsonify({"error": "Missing critical data"}), 400

        session_rain_flag = get_rain_flag(lat, lon, session_id)

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        insert_query = """
            INSERT INTO sensor_data (
                buoy_id, session_id, "timestamp", 
                gps_lat, gps_lon, water_leak, 
                pH, Temp, EC, Turbidity, "DO", ORP, 
                rain_flag, battery_v
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        inserted_rows = 0
        with conn.cursor() as cursor:
            for sample in samples:
                sample_time = sample.get('time')
                if not sample_time:
                    continue
                cursor.execute(insert_query, (
                    device_id, session_id, sample_time, lat, lon, water_leak,
                    sample.get('pH'), sample.get('temp'), sample.get('EC'),
                    sample.get('turbidity'), sample.get('DO'), sample.get('ORP'),
                    session_rain_flag, sample.get('battery_v')
                ))
                inserted_rows += 1
        
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "inserted": inserted_rows}), 200

    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        return jsonify({"error": str(e)}), 500

@app.route("/api/get-dashboard-data", methods=["POST"])
def get_dashboard_analysis():
    payload = request.json
    if not payload:
        return jsonify({"error": "No JSON payload"}), 400
    
    try:
        buoy_id = payload.get('buoy_id')
        timeframe_start = payload.get('timeframe_start')
        timeframe_end = payload.get('timeframe_end')
        ph_ideals_tuple = payload.get('ph_ideals_tuple')

        if not all([buoy_id, timeframe_start, timeframe_end]):
            return jsonify({"error": "Missing critical parameters"}), 400

        # --- PRE-CHECK: Prevent 500 crash by checking for data first ---
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM sensor_data 
                    WHERE buoy_id = %s 
                    AND "timestamp" >= %s 
                    AND "timestamp" <= %s
                """, (buoy_id, timeframe_start, timeframe_end))
                count = cursor.fetchone()[0]
            conn.close()
            
            if count == 0:
                return jsonify({"error": "No data available for this timeframe."}), 404
        # ---------------------------------------------------------------

        analysis_result = analytics.get_dashboard_data(
            buoy_id=buoy_id,
            timeframe_start=timeframe_start,
            timeframe_end=timeframe_end,
            ph_ideals_tuple=tuple(ph_ideals_tuple) if ph_ideals_tuple else None
        )
        
        if 'error' in analysis_result:
            return jsonify(analysis_result), 404
            
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

        return json.dumps(analysis_result, cls=CustomEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        return jsonify({"error": str(e), "traceback": error_msg}), 500

@app.route("/api/buoys/latest", methods=["GET"])
def get_latest_buoy_data():
    query = """
        SELECT DISTINCT ON (s.buoy_id)
            s.buoy_id, b.friendly_name, b.water_body_type, s."timestamp",
            s.water_leak, s.samples, s.gps
        FROM sensor_data s
        JOIN buoys b ON s.buoy_id = b.buoy_id
        ORDER BY s.buoy_id, s."timestamp" DESC;
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(query)
            latest_data = cursor.fetchall() 
        conn.close()
        results = [dict(row) for row in latest_data]

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

        return json.dumps(results, cls=CustomEncoder), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        if conn:
            conn.close()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
