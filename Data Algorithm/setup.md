Aqua Guardian Deployment Guide

Step 1: Set Up Your Database (TimescaleDB)

You only need to do this once. Connect to your database (e.g., with psql) and run this SQL to create your tables.

-- Create the metadata table for your buoys
CREATE TABLE buoys (
    buoy_id VARCHAR(50) PRIMARY KEY,
    water_body_type VARCHAR(50) NOT NULL, -- 'Lake' or 'Stream'
    friendly_name VARCHAR(100),
    deployment_date DATE
);

-- Create the main table for sensor data
CREATE TABLE sensor_data (
    "timestamp" TIMESTAMPTZ NOT NULL,
    buoy_id VARCHAR(50) NOT NULL,
    session_id VARCHAR(100),
    gps_lat DOUBLE PRECISION,
    gps_lon DOUBLE PRECISION,
    water_leak BOOLEAN,
    pH DOUBLE PRECISION,
    Temp DOUBLE PRECISION,
    EC DOUBLE PRECISION,
    Turbidity DOUBLE PRECISION,
    "DO" DOUBLE PRECISION, -- Quoted because DO is a keyword
    ORP DOUBLE PRECISION,
    rain_flag BOOLEAN,
    -- Link to the buoys table
    FOREIGN KEY (buoy_id) REFERENCES buoys (buoy_id)
);

-- !!! THIS IS THE MOST IMPORTANT STEP !!!
-- Convert sensor_data into a TimescaleDB hypertable
SELECT create_hypertable('sensor_data', by_range('timestamp'));

-- Add your buoy(s) to the metadata table
-- The 'buoy_id' (AG-01) MUST match DEVICE_ID in your config.h
INSERT INTO buoys (buoy_id, water_body_type, friendly_name, deployment_date)
VALUES 
('AG-01', 'Lake', 'Main Dock Buoy', '2025-11-10');


Step 2: Set Up the API Server on Your VM

Place Files: Put api_server.py and analytics.py in the same directory on your VM.

Install Python Libraries:

pip install Flask flask-cors requests psycopg2-binary pandas numpy


Set Environment Variables: This is how you securely pass credentials to the script.

# Your TimescaleDB/PostgreSQL password
export DB_PASS='your_database_password'

# Your OpenWeatherMap API Key
export WEATHER_API_KEY='your_openweathermap_api_key'

# (Optional) Set if your DB is not on localhost
export DB_HOST='localhost'
export DB_NAME='water_data'
export DB_USER='postgres'


Open VM Firewall Port: Your server will run on port 8080. You must allow incoming TCP traffic on this port in your VM's network settings (e.g., Google Cloud Firewall, AWS Security Group).

Run the Server:

python3 api_server.py


The server is now running and listening for requests.

Step 3: Configure and Flash Your Buoy

Edit main.cpp: Open the main.cpp file.

Change server: Change const char *server = "YOUR_VM_PUBLIC_IP"; to your VM's static public IP address (e.g., "203.0.113.50").

Check api_key_value: Ensure const char *api_key_value in main.cpp perfectly matches the EXPECTED_API_KEY in api_server.py.

Flash: Build and flash this firmware to your ESP32.

Step 4: View the Analysis Dashboard

This is the main dashboard for deep analysis.

Edit Aqua_Guardian_Dashboard.html: Open the HTML file.

Change API_URL: Find the line const API_URL = 'http://YOUR_VM_PUBLIC_IP:8080/api/get-dashboard-data';.

Update It: Change YOUR_VM_PUBLIC_IP to your VM's public IP address, just like you did in main.cpp.

Open: You can now open this HTML file in any web browser. When you click "Run Analysis," it will call your API server and display the data.

Step 5: View the Homepage (Live Status)

This is the new, simple homepage you created to show the latest status of all buoys.

Edit homepage.html: Open the homepage.html file.

Change API_URL: Find the line const API_URL = 'http://YOUR_VM_PUBLIC_IP:8080/api/buoys/latest';.

Update It: Change YOUR_VM_PUBLIC_IP to your VM's public IP address.

Open: You can open this file in any browser to see a live status summary of all your buoys.