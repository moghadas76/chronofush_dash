"""
Example Data Generator for Sensor Interpolation Dashboard

This script generates sample data files that can be used with the dashboard:
- sensor_network.graphml: NetworkX graph with sensor locations
- time_series_data.csv: Time series data for all sensors
- sensor_locations.csv: Sensor coordinates for map visualization

Run this script to create test data for the dashboard.
"""

import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sensor_network(num_sensors=20, connection_probability=0.3):
    """Generate a random sensor network with geographic coordinates."""
    
    np.random.seed(42)  # For reproducible results
    
    # Create graph
    G = nx.Graph()
    
    # NYC area coordinates
    center_lat, center_lng = 40.7128, -74.0060
    
    # Generate sensors with random locations
    sensors = []
    for i in range(1, num_sensors + 1):
        sensor_id = f"sensor_{i:03d}"
        
        # Random location within ~10km radius of NYC center
        lat = center_lat + np.random.uniform(-0.05, 0.05)
        lng = center_lng + np.random.uniform(-0.08, 0.08)
        
        G.add_node(sensor_id, 
                  latitude=lat, 
                  longitude=lng,
                  sensor_type=np.random.choice(['traffic', 'environmental', 'weather']),
                  installation_date='2023-01-01')
        
        sensors.append({
            'sensor_id': sensor_id,
            'latitude': lat,
            'longitude': lng,
            'sensor_type': G.nodes[sensor_id]['sensor_type']
        })
    
    # Add edges based on proximity and random connections
    sensor_positions = {s['sensor_id']: (s['latitude'], s['longitude']) for s in sensors}
    
    for i, sensor1 in enumerate(sensors):
        for j, sensor2 in enumerate(sensors[i+1:], i+1):
            # Calculate distance
            lat1, lng1 = sensor_positions[sensor1['sensor_id']]
            lat2, lng2 = sensor_positions[sensor2['sensor_id']]
            distance = np.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)
            
            # Connect if close enough or by random chance
            if distance < 0.025 or np.random.random() < connection_probability:
                G.add_edge(sensor1['sensor_id'], sensor2['sensor_id'], 
                          weight=distance,
                          connection_type='wireless')
    
    return G, pd.DataFrame(sensors)

def generate_time_series_data(sensors_df, num_days=7, freq='H'):
    """Generate realistic time series data for sensors."""
    
    np.random.seed(42)
    
    # Create timestamp range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=num_days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    data = {'timestamp': timestamps}
    
    # Generate data for each sensor
    for _, sensor in sensors_df.iterrows():
        sensor_id = sensor['sensor_id']
        sensor_type = sensor['sensor_type']
        
        # Base patterns based on sensor type
        if sensor_type == 'traffic':
            # Traffic pattern: high during rush hours, low at night
            base_value = 50
            daily_pattern = np.array([
                np.sin((t.hour - 8) * np.pi / 12) * 30 + 
                np.sin((t.hour - 17) * np.pi / 12) * 20
                for t in timestamps
            ])
            noise_level = 10
            
        elif sensor_type == 'environmental':
            # Environmental: temperature-like pattern
            base_value = 20
            daily_pattern = np.array([
                np.sin((t.hour - 14) * np.pi / 12) * 15
                for t in timestamps
            ])
            noise_level = 5
            
        else:  # weather
            # Weather: humidity-like pattern
            base_value = 60
            daily_pattern = np.array([
                np.sin((t.hour - 6) * np.pi / 12) * 20 +
                np.random.normal(0, 5)  # More random variation
                for t in timestamps
            ])
            noise_level = 8
        
        # Add weekly pattern
        weekly_pattern = np.array([
            np.sin(i * 2 * np.pi / (7 * 24)) * 10  # Assuming hourly data
            for i in range(len(timestamps))
        ])
        
        # Add seasonal trend
        seasonal_trend = np.linspace(0, 5, len(timestamps))
        
        # Combine all patterns
        noise = np.random.normal(0, noise_level, len(timestamps))
        values = base_value + daily_pattern + weekly_pattern + seasonal_trend + noise
        
        # Ensure positive values
        values = np.maximum(values, 0)
        
        data[sensor_id] = values
    
    return pd.DataFrame(data)

def create_sample_mask_data(time_series_df, missing_ratio=0.15):
    """Create missing data masks for interpolation testing."""
    
    np.random.seed(42)
    
    mask_data = []
    sensor_columns = [col for col in time_series_df.columns if col != 'timestamp']
    
    for _, row in time_series_df.iterrows():
        timestamp = row['timestamp']
        
        for sensor_id in sensor_columns:
            # Create mask: 1 means missing, 0 means observed
            is_missing = np.random.random() < missing_ratio
            
            mask_data.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'mask': 1 if is_missing else 0,
                'value': row[sensor_id] if not is_missing else np.nan
            })
    
    return pd.DataFrame(mask_data)

def main():
    """Generate all sample data files."""
    
    print("🔧 Generating sample data for Sensor Interpolation Dashboard...")
    
    # Parameters
    num_sensors = 20
    num_days = 7
    
    # Create output directory
    output_dir = "sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate sensor network
    print(f"📍 Creating sensor network with {num_sensors} sensors...")
    graph, sensors_df = generate_sensor_network(num_sensors)
    
    # Save graph
    graph_path = os.path.join(output_dir, "sensor_network.graphml")
    nx.write_graphml(graph, graph_path)
    print(f"   ✅ Saved: {graph_path}")
    
    # Save sensor locations
    locations_path = os.path.join(output_dir, "sensor_locations.csv")
    sensors_df.to_csv(locations_path, index=False)
    print(f"   ✅ Saved: {locations_path}")
    
    # 2. Generate time series data
    print(f"📊 Generating {num_days} days of time series data...")
    time_series_df = generate_time_series_data(sensors_df, num_days)
    
    ts_path = os.path.join(output_dir, "time_series_data.csv")
    time_series_df.to_csv(ts_path, index=False)
    print(f"   ✅ Saved: {ts_path}")
    
    # 3. Generate mask data
    print("🎭 Creating missing data masks...")
    mask_df = create_sample_mask_data(time_series_df)
    
    mask_path = os.path.join(output_dir, "missing_data_masks.csv")
    mask_df.to_csv(mask_path, index=False)
    print(f"   ✅ Saved: {mask_path}")
    
    # 4. Generate summary
    print("\n📋 Data Summary:")
    print(f"   • Sensors: {len(sensors_df)}")
    print(f"   • Connections: {graph.number_of_edges()}")
    print(f"   • Time points: {len(time_series_df)}")
    print(f"   • Total values: {len(mask_df)}")
    print(f"   • Missing ratio: {mask_df['mask'].mean():.1%}")
    
    # Save summary
    summary = {
        'num_sensors': len(sensors_df),
        'num_connections': graph.number_of_edges(),
        'num_timepoints': len(time_series_df),
        'missing_ratio': float(mask_df['mask'].mean()),
        'sensor_types': sensors_df['sensor_type'].value_counts().to_dict(),
        'time_range': {
            'start': time_series_df['timestamp'].min().isoformat(),
            'end': time_series_df['timestamp'].max().isoformat()
        }
    }
    
    import json
    summary_path = os.path.join(output_dir, "data_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Saved: {summary_path}")
    
    print(f"\n🎉 Sample data generated successfully in '{output_dir}' directory!")
    print("\nTo use with the dashboard:")
    print("1. Start the dashboard: streamlit run dashboard.py")
    print("2. Upload the generated files:")
    print(f"   - Graph: {graph_path}")
    print(f"   - Locations: {locations_path}")
    print(f"   - Time series: {ts_path}")
    print(f"   - Masks: {mask_path}")

if __name__ == "__main__":
    main()
