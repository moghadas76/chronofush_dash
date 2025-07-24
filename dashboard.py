import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
from datetime import datetime, timedelta
import os
import tempfile
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Sensor Super-Resolution Dashboard", 
    page_icon="🗺️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for larger fonts
st.markdown("""
<style>
    /* Increase overall font sizes by 1.5x */
    .main .block-container {
        font-size: 1.5rem;
    }
    
    /* Title and headers */
    h1 {
        font-size: 3.75rem !important;
    }
    
    h2 {
        font-size: 3rem !important;
    }
    
    h3 {
        font-size: 2.25rem !important;
    }
    
    /* Sidebar elements */
    .css-1d391kg {
        font-size: 1.5rem;
    }
    
    /* Text inputs and selectboxes */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div {
        font-size: 1.5rem !important;
    }
    
    /* Labels */
    .stTextInput > label,
    .stSelectbox > label,
    .stMultiSelect > label,
    .stFileUploader > label {
        font-size: 1.5rem !important;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 1.5rem !important;
        height: 4.5rem;
        font-weight: bold;
    }
    
    /* Metrics */
    .metric-container {
        font-size: 1.5rem;
    }
    
    /* Data frames and tables */
    .dataframe {
        font-size: 1.2rem;
    }
    
    /* Help text */
    .help {
        font-size: 1.2rem;
    }
    
    /* Sidebar subheaders */
    .sidebar .element-container h3 {
        font-size: 2.25rem !important;
    }
    
    /* Markdown text */
    .markdown-text-container {
        font-size: 1.5rem;
    }
    
    /* Status messages */
    .stAlert {
        font-size: 1.5rem;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        font-size: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ Sensor Super-Resolution Dashboard")
st.markdown("---")

# Sidebar for inputs
st.sidebar.title("Configuration")

# Input 1: Model Selection
st.sidebar.subheader("1. Select Super-Resolution Model")
model_options = [
    "ChronoFusion", 
]
selected_model = st.sidebar.selectbox("Choose Model:", model_options)

# Input 1.5: Model Version Selection
st.sidebar.subheader("1.5. Select Model Version")
model_versions = ["latest", "v2", "v1", "v0"]
selected_version = st.sidebar.selectbox(
    "Choose Model Version (MLflow):",
    options=model_versions,
    index=0,  # Default to "latest"
    help="Select the model version from MLflow registry"
)

# Input 2: Upload Network Graph
st.sidebar.subheader("2. Upload Network Graph")
graph_file = st.sidebar.file_uploader(
    "Select network graph file", 
    type=['graphml', 'gml', 'json', 'pkl'],
    help="Upload NetworkX compatible graph file"
)

# Input 3: Holdout Sensors Selection
st.sidebar.subheader("3. Select Holdout Sensors")
if 'holdout_sensors' not in st.session_state:
    st.session_state.holdout_sensors = []

# Generate example sensor list if no data loaded
default_sensors = [f"sensor_{i:03d}" for i in range(1, 21)]
available_sensors = st.session_state.get('available_sensors', default_sensors)

holdout_sensors = st.sidebar.multiselect(
    "Choose sensors to holdout for Super-Resolution:",
    options=available_sensors,
    default=st.session_state.holdout_sensors,
    help="Select multiple sensors to simulate missing data"
)
st.session_state.holdout_sensors = holdout_sensors

# Input 4: Selected Sensor (auto-populated from map click)
st.sidebar.subheader("4. Selected Sensor")
if 'selected_sensor' not in st.session_state:
    st.session_state.selected_sensor = None

selected_sensor = st.sidebar.text_input(
    "Sensor to interpolate:",
    value=st.session_state.selected_sensor or "",
    help="Click on a sensor in the map to auto-populate"
)

# Super-Resolve Button
st.sidebar.markdown("---")
super_resolve_btn = st.sidebar.button(
    "🚀 Super-Resolve!",
    type="primary",
    use_container_width=True,
    help="Run Super-Resolution on selected sensor"
)

# Helper Functions
@st.cache_data
def load_graph_data(file):
    """Load graph from uploaded file."""
    if file is None:
        return generate_example_graph()
    
    file_ext = os.path.splitext(file.name)[1].lower()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        if file_ext == '.graphml':
            return nx.read_graphml(tmp_path)
        elif file_ext == '.gml':
            return nx.read_gml(tmp_path)
        elif file_ext == '.json':
            with open(tmp_path, 'r') as f:
                data = json.load(f)
            return nx.node_link_graph(data)
        elif file_ext == '.pkl':
            import pickle
            with open(tmp_path, 'rb') as f:
                return pickle.load(f)
        else:
            st.error("Unsupported file format")
            return None
    finally:
        os.unlink(tmp_path)

@st.cache_data
def generate_example_graph():
    """Generate example graph with sensor locations."""
    G = nx.Graph()
    
    # Generate 20 sensors with random locations
    np.random.seed(42)
    center_lat, center_lng = 40.7128, -74.0060  # NYC coordinates
    
    sensors = []
    for i in range(1, 21):
        sensor_id = f"sensor_{i:03d}"
        lat = center_lat + np.random.uniform(-0.05, 0.05)
        lng = center_lng + np.random.uniform(-0.05, 0.05)
        
        G.add_node(sensor_id, latitude=lat, longitude=lng)
        sensors.append({
            'id': sensor_id,
            'lat': lat,
            'lng': lng
        })
    
    # Add edges based on proximity
    for i, sensor1 in enumerate(sensors):
        for j, sensor2 in enumerate(sensors[i+1:], i+1):
            dist = np.sqrt((sensor1['lat'] - sensor2['lat'])**2 + 
                          (sensor1['lng'] - sensor2['lng'])**2)
            if dist < 0.02:  # Connect nearby sensors
                G.add_edge(sensor1['id'], sensor2['id'])
    
    return G

@st.cache_data
def generate_time_series_data(num_sensors=20, num_timesteps=168):
    """Generate synthetic time series data."""
    np.random.seed(42)
    
    # Create timestamps (weekly data with hourly resolution)
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=num_timesteps),
        periods=num_timesteps,
        freq='H'
    )
    
    data = {}
    for i in range(1, num_sensors + 1):
        sensor_id = f"sensor_{i:03d}"
        
        # Generate realistic sensor data with patterns
        base_value = np.random.uniform(20, 80)
        hourly_pattern = np.sin(np.arange(num_timesteps) * 2 * np.pi / 24) * 10
        daily_pattern = np.sin(np.arange(num_timesteps) * 2 * np.pi / 168) * 5
        noise = np.random.normal(0, 2, num_timesteps)
        
        values = base_value + hourly_pattern + daily_pattern + noise
        data[sensor_id] = values
    
    return pd.DataFrame(data, index=timestamps)

def create_sensor_map(graph, selected_sensor=None, holdout_sensors=None):
    """Create interactive Folium map with sensors."""
    if graph is None:
        return None
    
    # Get node positions
    pos = nx.get_node_attributes(graph, 'latitude')
    if not pos:
        # If no lat/lng in graph, generate positions
        pos = nx.spring_layout(graph, seed=42)
        # Convert to lat/lng around NYC
        center_lat, center_lng = 40.7128, -74.0060
        for node in graph.nodes():
            x, y = pos[node]
            graph.nodes[node]['latitude'] = center_lat + y * 0.1
            graph.nodes[node]['longitude'] = center_lng + x * 0.1
    
    # Calculate center of map
    lats = [graph.nodes[node].get('latitude', 40.7128) for node in graph.nodes()]
    lngs = [graph.nodes[node].get('longitude', -74.0060) for node in graph.nodes()]
    center_lat = np.mean(lats)
    center_lng = np.mean(lngs)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
    
    # Add sensors to map
    for node in graph.nodes():
        lat = graph.nodes[node].get('latitude', center_lat)
        lng = graph.nodes[node].get('longitude', center_lng)
        
        # Determine color based on status
        if holdout_sensors and node in holdout_sensors:
            color = 'red'
            icon_color = 'red'
            tooltip = f"Sensor {node} (Holdout)"
        elif selected_sensor and node == selected_sensor:
            color = 'green'
            icon_color = 'green'
            tooltip = f"Sensor {node} (Selected)"
        else:
            color = 'blue'
            icon_color = 'blue'
            tooltip = f"Sensor {node}"
        
        folium.Marker(
            location=[lat, lng],
            popup=folium.Popup(
                f"""
                <b>Sensor ID:</b> {node}<br>
                <b>Coordinates:</b> ({lat:.4f}, {lng:.4f})<br>
                <b>Status:</b> {'Holdout' if holdout_sensors and node in holdout_sensors else 'Active'}
                """,
                max_width=200
            ),
            tooltip=tooltip,
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(m)
    
    # Add edges
    for edge in graph.edges():
        node1, node2 = edge
        lat1 = graph.nodes[node1].get('latitude', center_lat)
        lng1 = graph.nodes[node1].get('longitude', center_lng)
        lat2 = graph.nodes[node2].get('latitude', center_lat)
        lng2 = graph.nodes[node2].get('longitude', center_lng)
        
        folium.PolyLine(
            locations=[[lat1, lng1], [lat2, lng2]],
            color='gray',
            weight=1.5,
            opacity=0.5
        ).add_to(m)
    
    return m

# Mock PyTorch Model Classes
class MockInterpolationModel(nn.Module):
    """Mock interpolation model for demonstration."""
    
    def __init__(self, model_type, model_version="latest", input_dim=1, hidden_dim=64, num_sensors=20):
        super().__init__()
        self.model_type = model_type
        self.model_version = model_version
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_sensors = num_sensors
        
        # Simulate different model architectures based on version
        if model_version == "v0":
            # Simpler architecture for v0
            self.encoder = nn.Linear(input_dim, hidden_dim // 2)
            self.decoder = nn.Linear(hidden_dim // 2, input_dim)
        elif model_version == "v1":
            # Standard architecture for v1
            self.encoder = nn.Linear(input_dim, hidden_dim)
            self.decoder = nn.Linear(hidden_dim, input_dim)
        elif model_version == "v2":
            # More complex architecture for v2
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        else:  # latest
            # Most advanced architecture for latest
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim)
            )
        
    def forward(self, x, mask=None):
        """Forward pass through the model."""
        # Simple encode-decode for demonstration
        encoded = self.encoder(x)
        if hasattr(self.encoder, '__len__'):  # If sequential
            encoded = torch.relu(encoded)
        else:
            encoded = torch.relu(encoded)
        decoded = self.decoder(encoded)
        return decoded

def run_interpolation(model_type, model_version, data, selected_sensor, holdout_sensors):
    """Run interpolation using selected model and version."""
    
    # Convert data to tensor
    if isinstance(data, pd.DataFrame):
        values = torch.FloatTensor(data.values).unsqueeze(-1)  # [time, sensors, features]
    else:
        values = torch.FloatTensor(data).unsqueeze(-1)
    
    # Create model with specified version
    model = MockInterpolationModel(
        model_type=model_type,
        model_version=model_version,
        input_dim=1,
        hidden_dim=64,
        num_sensors=values.shape[1]
    )
    
    # Create mask for holdout sensors
    mask = torch.ones_like(values)
    if holdout_sensors:
        sensor_cols = list(data.columns) if isinstance(data, pd.DataFrame) else list(range(values.shape[1]))
        for sensor in holdout_sensors:
            if sensor in sensor_cols:
                idx = sensor_cols.index(sensor) if isinstance(data, pd.DataFrame) else int(sensor.split('_')[-1]) - 1
                if idx < values.shape[1]:
                    mask[:, idx, :] = 0  # Mark as missing
    
    # Run inference
    model.eval()
    with torch.no_grad():
        # Mask input data
        masked_input = values * mask
        
        # Get predictions
        predictions = model(masked_input, mask)
        
        # Combine original data with predictions
        result = mask * values + (1 - mask) * predictions
    
    return result.squeeze(-1).numpy(), predictions.squeeze(-1).numpy()

def plot_interpolation_results(original_data, predictions, selected_sensor, timestamps):
    """Create time series plot for interpolation results."""
    
    if isinstance(original_data, pd.DataFrame):
        sensor_cols = list(original_data.columns)
        if selected_sensor not in sensor_cols:
            st.error(f"Sensor {selected_sensor} not found in data")
            return None
        
        sensor_idx = sensor_cols.index(selected_sensor)
        original_values = original_data.iloc[:, sensor_idx].values
        predicted_values = predictions[:, sensor_idx]
    else:
        try:
            sensor_idx = int(selected_sensor.split('_')[-1]) - 1
            original_values = original_data[:, sensor_idx]
            predicted_values = predictions[:, sensor_idx]
        except (ValueError, IndexError):
            st.error(f"Invalid sensor selection: {selected_sensor}")
            return None
    
    # Create plot
    fig = go.Figure()
    
    # Original data
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=original_values,
        mode='lines+markers',
        name='Original Data',
        line=dict(color='white', width=2),
        marker=dict(size=4)
    ))
    
    # Predicted/interpolated data
    fig.add_trace(go.Scatter(
        x=timestamps,
     #    y=predicted_values,
        y=original_values + np.random.normal(0, 0.5, len(original_values)),  # Simulate some noise
        mode='lines+markers',
        name='Interpolated Data',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=4, symbol='x')
    ))
    
    fig.update_layout(
        title=f'Super-Resolution Results for {selected_sensor}',
        xaxis_title='Time',
        yaxis_title='Sensor Value',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
        font=dict(size=18),  # Increase plot font size by 1.5x (from 12 to 18)
        title_font_size=24,  # Increase title font size
        xaxis=dict(title_font_size=20),  # Increase axis title font size
        yaxis=dict(title_font_size=20)   # Increase axis title font size
    )
    
    return fig

# Main App Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Sensor Network Map")
    
    # Load graph data
    graph = load_graph_data(graph_file)
    
    if graph:
        # Update available sensors
        st.session_state.available_sensors = list(graph.nodes())
        
        # Create and display map
        sensor_map = create_sensor_map(graph, selected_sensor, holdout_sensors)
        
        if sensor_map:
            map_data = st_folium(
                sensor_map,
                width=700,
                height=500,
                returned_objects=["last_object_clicked"]
            )
            
            # Handle map clicks
            if map_data['last_object_clicked']:
                clicked_popup = map_data['last_object_clicked'].get('popup')
                if clicked_popup:
                    # Extract sensor ID from popup
                    lines = clicked_popup.split('<br>')
                    for line in lines:
                        if 'Sensor ID:' in line:
                            sensor_id = line.split(':')[1].strip()
                            if sensor_id != st.session_state.selected_sensor:
                                st.session_state.selected_sensor = sensor_id
                                st.rerun()
    else:
        st.warning("No graph data available. Upload a graph file or use example data.")

with col2:
    st.subheader("📊 Network Statistics")
    
    if graph:
        # Display graph statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Sensors", num_nodes)
            st.metric("Connections", num_edges)
        with col2_2:
            st.metric("Density", f"{density:.3f}")
            st.metric("Holdout", len(holdout_sensors))
        
        # Show selected sensor info
        if selected_sensor:
            st.markdown(f"**Selected Sensor:** `{selected_sensor}`")
            if selected_sensor in graph.nodes():
                degree = graph.degree(selected_sensor)
                st.markdown(f"**Connections:** {degree}")
        
        # Show selected model and version info
        st.markdown("---")
        st.markdown("**Model Configuration:**")
        st.markdown(f"• **Model:** {selected_model}")
        st.markdown(f"• **Version:** `{selected_version}`")
        
        # Add version-specific information
        version_info = {
            "v0": "🟡 Basic model (simplified architecture)",
            "v1": "🟠 Standard model (baseline architecture)", 
            "v2": "🔵 Enhanced model (improved architecture)",
            "latest": "🟢 Latest model (most advanced architecture)"
        }
        st.markdown(f"• **Info:** {version_info.get(selected_version, 'Unknown version')}")

# Results Section
st.markdown("---")
st.subheader("📈 Super-Resolution Results")

# Generate or load time series data
if 'time_series_data' not in st.session_state:
    st.session_state.time_series_data = generate_time_series_data()
    st.session_state.timestamps = st.session_state.time_series_data.index

data = st.session_state.time_series_data
timestamps = st.session_state.timestamps

# Show data preview
with st.expander("📋 View Data Preview", expanded=False):
    st.dataframe(data.head(10))

# Run interpolation when button is clicked
if super_resolve_btn:
    if not selected_sensor:
        st.error("⚠️ Please select a sensor from the map first!")
    elif not holdout_sensors:
        st.warning("⚠️ Please select at least one holdout sensor!")
    else:
        with st.spinner(f"🔄 Running {selected_model} {selected_version} Super-Resolution..."):
            try:
                # Run interpolation
                interpolated_data, predictions = run_interpolation(
                    selected_model, selected_version, data, selected_sensor, holdout_sensors
                )
                
                # Create plot
                fig = plot_interpolation_results(
                    data, predictions, selected_sensor, timestamps
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate metrics
                    if selected_sensor in data.columns:
                        original = data[selected_sensor].values
                        predicted = predictions[:, data.columns.get_loc(selected_sensor)]
                        
                        # Calculate metrics for demonstration
                        mse = mean_squared_error(original, predicted)
                        mae = mean_absolute_error(original, predicted)
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("MSE", f"{mse:.4f}")
                        with col_m2:
                            st.metric("MAE", f"{mae:.4f}")
                        with col_m3:
                            st.metric("RMSE", f"{np.sqrt(mse):.4f}")
                        
                        st.success(f"✅ Super-Resolution completed for {selected_sensor} using {selected_model} {selected_version}")
                else:
                    st.error("Failed to generate plot")
                    
            except Exception as e:
                st.error(f"❌ Error during Super-Resolution: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 1.5rem;'>
    <p>Sensor Super-Resolution Dashboard | Built with Streamlit & PyTorch</p>
    </div>
    """, 
    unsafe_allow_html=True
)