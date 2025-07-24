# Sensor Interpolation Dashboard

A Streamlit-based interactive dashboard for sensor data interpolation using PyTorch models with leaflet map visualization.

## Features

### 🗺️ Interactive Map Interface
- **Leaflet Map View**: Interactive map showing sensor locations and network connections
- **Click-to-Select**: Click on any sensor on the map to automatically select it for interpolation
- **Visual Status Indicators**: 
  - Blue markers: Active sensors
  - Red markers: Holdout sensors (used for validation)
  - Green markers: Currently selected sensor

### 📊 Four Key Input Controls

1. **Model Selection Dropdown**
   - SAITS (Self-Attention Imputation)
   - BRITS (Bidirectional RNN)
   - ST-Encoder (Spatio-Temporal)
   - BiGRU Spatial Attention
   - Time Series Denoiser
   - IGNNK (Graph Neural Network)

2. **Network Graph Upload**
   - Supports multiple NetworkX formats: `.graphml`, `.gml`, `.json`, `.pkl`
   - Automatically parses sensor locations and connections
   - Falls back to example graph if no file uploaded

3. **Holdout Sensors Selection**
   - Multi-select checkbox interface
   - Choose multiple sensors to simulate missing data
   - Used for validation and testing interpolation accuracy

4. **Selected Sensor (Auto-populated)**
   - Automatically filled when clicking sensors on the map
   - Can also be manually entered
   - Target sensor for interpolation results

### 🚀 Super-Resolve Button
- Executes the selected PyTorch model on the data
- Sends preprocessed dataframes to the model
- Displays interpolation results for the selected sensor

### 📈 Output Visualization
- **Time Series Plot**: Interactive Plotly chart showing:
  - Original sensor data (blue line)
  - Interpolated/predicted values (red dashed line)
- **Performance Metrics**: MSE, MAE, RMSE calculations
- **Network Statistics**: Node count, edge count, network density

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_dashboard.txt
```

2. Run the dashboard:
```bash
streamlit run dashboard.py
```

## Usage

### Quick Start with Example Data
1. Launch the dashboard
2. The system will automatically generate example sensor network data
3. Select holdout sensors from the sidebar
4. Click on a sensor in the map to select it
5. Choose an interpolation model
6. Click "Super-Resolve!" to run interpolation

### Using Your Own Data

#### Graph File Format
Upload a NetworkX-compatible graph file with sensor locations:
```python
# Example: Creating a compatible graph
import networkx as nx

G = nx.Graph()
G.add_node("sensor_001", latitude=40.7128, longitude=-74.0060)
G.add_node("sensor_002", latitude=40.7589, longitude=-73.9851)
G.add_edge("sensor_001", "sensor_002")

# Save as GraphML
nx.write_graphml(G, "sensor_network.graphml")
```

#### Expected Node Attributes
- `latitude`: Float, sensor latitude coordinate
- `longitude`: Float, sensor longitude coordinate
- Node IDs should be string identifiers (e.g., "sensor_001")

### Model Integration

The dashboard provides a framework for integrating PyTorch models. To add your own model:

1. **Inherit from the base model class**:
```python
class YourCustomModel(MockInterpolationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Your model initialization
        
    def forward(self, x, mask=None):
        # Your model forward pass
        return predictions
```

2. **Update the model selection options**:
```python
model_options = [
    # ... existing options
    "Your Custom Model"
]
```

3. **Modify the model instantiation**:
```python
def run_interpolation(model_type, data, selected_sensor, holdout_sensors):
    if model_type == "Your Custom Model":
        model = YourCustomModel(...)
    # ... rest of the function
```

## Architecture

### Data Flow
1. **Input**: Graph file upload or example data generation
2. **Processing**: Parse graph structure and sensor locations
3. **Visualization**: Render interactive map with Folium/Streamlit-Folium
4. **Interaction**: Handle map clicks to update selected sensor
5. **Model Execution**: Run PyTorch model with masked data
6. **Output**: Display time series plot and metrics

### Key Components
- **Graph Management**: NetworkX for graph operations
- **Mapping**: Folium for interactive maps
- **Visualization**: Plotly for time series charts
- **UI Framework**: Streamlit for responsive interface
- **ML Backend**: PyTorch for interpolation models

## File Structure
```
dashboard.py                 # Main Streamlit application
requirements_dashboard.txt   # Python dependencies
README_dashboard.md         # This documentation
```

## Customization

### Adding New Models
1. Implement your PyTorch model class
2. Add model name to the dropdown options
3. Update the model instantiation logic
4. Ensure model returns compatible output format

### Modifying Map Appearance
- Edit the `create_sensor_map()` function
- Customize marker colors, sizes, and popup content
- Adjust map zoom levels and center coordinates

### Extending Metrics
- Modify the `plot_interpolation_results()` function
- Add additional performance metrics calculations
- Create new visualization components

## Troubleshooting

### Common Issues

1. **Map not displaying**: Check if folium and streamlit-folium are properly installed
2. **Graph file not loading**: Ensure file format is supported and properly structured
3. **Model errors**: Verify input data dimensions match model expectations

### Performance Tips

1. **Large Networks**: For networks with >1000 nodes, consider sampling or clustering
2. **Time Series Length**: Long time series may require pagination or windowing
3. **Model Complexity**: Complex models may need GPU acceleration

## Contributing

To extend the dashboard:
1. Fork the repository
2. Add new features following the existing architecture
3. Update documentation
4. Submit a pull request

## License

This dashboard is part of the SAITS project and follows the same licensing terms.
