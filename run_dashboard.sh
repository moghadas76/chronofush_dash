#!/bin/bash

# Sensor Interpolation Dashboard Launcher
# This script sets up and runs the Streamlit dashboard

echo "🗺️ Sensor Interpolation Dashboard"
echo "=================================="

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not running in a virtual environment"
    echo "   Consider creating one with: python -m venv dashboard_env"
    echo "   Then activate with: source dashboard_env/bin/activate"
    echo ""
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python -c "import streamlit, folium, plotly, networkx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements_dashboard.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check your Python environment."
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ All dependencies found"
fi

echo ""
echo "🚀 Starting Streamlit dashboard..."
echo "   The dashboard will open in your browser at http://localhost:8501"
echo "   Press Ctrl+C to stop the server"
echo ""

# Launch the dashboard
streamlit run dashboard.py --server.address localhost --server.port 8501

echo ""
echo "👋 Dashboard stopped. Thanks for using the Sensor Interpolation Dashboard!"
