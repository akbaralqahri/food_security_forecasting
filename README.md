# 🌾 Food Security Forecasting Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Comprehensive interactive dashboard for food security analysis and forecasting using machine learning, featuring advanced geographic visualization and risk assessment capabilities.

## 🚀 Features

### 🤖 **Machine Learning**
- **Random Forest Modeling** with hyperparameter optimization
- **Time Series Cross-Validation** for robust performance assessment
- **Feature Importance Analysis** to identify key drivers
- **Scenario Forecasting** with uncertainty quantification

### 🗺️ **Geographic Visualization**
- **Interactive Risk Maps** with province-level detail
- **Animated Scenario Comparison** across different projections
- **Regional Analysis** with statistical breakdowns
- **Risk Heatmaps** using treemap visualization
- **Export Capabilities** (CSV, JSON, GeoJSON formats)

### 📊 **Advanced Analytics**
- **Real-time Progress Tracking** during analysis
- **Comprehensive Risk Assessment** with actionable insights
- **Data Quality Validation** with automated checks
- **Executive Reporting** with key performance indicators

### 🎯 **User Experience**
- **Enhanced Error Handling** with graceful fallbacks
- **Responsive Design** optimized for various screen sizes
- **Interactive Tooltips** and detailed hover information
- **Export Functionality** for reports and data

## 📁 Project Structure

```
food-security-dashboard/
├── streamlit_dashboard.py          # Main dashboard application
├── src/
│   ├── __init__.py
│   ├── food_security_forecasting.py    # ML models and analysis
│   ├── visualization.py               # Standard visualizations
│   ├── geo_visualization.py           # Geographic mapping functions
│   └── utils.py                       # Utility functions
├── data/
│   ├── sample_data.csv               # Sample dataset
│   └── province_coordinates.json     # Indonesia province coordinates
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT License
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/food-security-dashboard.git
cd food-security-dashboard
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install core packages only
pip install streamlit pandas numpy plotly scikit-learn folium
```

### Step 4: Verify Installation
```bash
# Test streamlit installation
streamlit hello
```

## 🚀 Usage

### Quick Start
```bash
# Run the dashboard
streamlit run streamlit_dashboard.py

# The dashboard will open in your browser at http://localhost:8501
```

### Using Sample Data
1. **Start the application**
2. **Select "Use Sample Data"** in the sidebar
3. **Click "Load Sample Data"**
4. **Run "Full Analysis"** to see all features

### Using Your Own Data

#### Data Format Requirements
Your CSV file should contain these columns:

| Column | Description | Type | Range |
|--------|-------------|------|-------|
| `Tahun` | Year | Integer | 2000-2030 |
| `Provinsi` | Province name | String | Indonesian provinces |
| `Kabupaten` | District/City name | String | - |
| `Kemiskinan (%)` | Poverty rate | Float | 0-100 |
| `Pengeluaran Pangan (%)` | Food expenditure | Float | 0-100 |
| `Tanpa Air Bersih (%)` | Without clean water | Float | 0-100 |
| `Lama Sekolah Perempuan (tahun)` | Female education years | Float | 0-15 |
| `Rasio Tenaga Kesehatan` | Health worker ratio | Float | 0+ |
| `Angka Harapan Hidup (tahun)` | Life expectancy | Float | 50-85 |
| `Komposit` | Composite food security score | Integer | 1-6 |

#### Upload Process
1. **Prepare your CSV** following the format above
2. **Upload via sidebar** "Upload CSV File" option
3. **Review validation results** - fix any errors shown
4. **Configure analysis settings** if needed
5. **Run the analysis**

## 📊 Dashboard Sections

### 1. 📊 Data Overview
- **Data quality metrics** and completeness assessment
- **Missing value analysis** with visualizations
- **Temporal trends** and distribution analysis
- **Interactive data explorer** with filtering

### 2. 🤖 Model Performance
- **Cross-validation results** with detailed metrics
- **Performance visualization** and stability analysis
- **Model diagnostics** and learning curves
- **Recommendation system** based on performance

### 3. 🎯 Feature Analysis
- **Feature importance ranking** with interactive charts
- **Category-based analysis** (Economic, Health, Education)
- **Interactive feature explorer** with filtering
- **Optimization recommendations** for low-impact features

### 4. 🔮 Forecasting
- **Scenario-based predictions** for multiple futures
- **Uncertainty quantification** with confidence intervals
- **Provincial performance ranking** (best/worst)
- **Comparative scenario analysis**

### 5. 🗺️ Geographic Analysis ⭐ **NEW**
- **Interactive Risk Maps**
  - Scatter maps with color-coded risk levels
  - Hover information with detailed province data
  - Zoom and pan capabilities
- **Regional Analysis**
  - Statistics by major regions (Java, Sumatra, etc.)
  - Comparative charts and performance metrics
- **Scenario Comparison**
  - Animated maps showing changes across scenarios
  - Play/pause controls for temporal analysis
- **Data Export**
  - Multiple formats: CSV, JSON, GeoJSON
  - Province-specific data filtering
  - Coordinate data included

### 6. ⚠️ Risk Assessment
- **Risk level distribution** with action priorities
- **Critical alerts** for high-risk situations
- **Geographic risk patterns** and regional analysis
- **Actionable recommendations** by urgency level

### 7. 📋 Reports
- **Executive dashboard** with KPIs
- **Strategic insights** and key findings
- **Policy recommendations** by time horizon
- **Comprehensive export options**

## 🗺️ Geographic Features

### Supported Provinces
The dashboard includes coordinates for all 38 Indonesian provinces:
- **Sumatra Region**: Aceh, Sumatera Utara, Sumatera Barat, Riau, etc.
- **Java Region**: DKI Jakarta, Jawa Barat, Jawa Tengah, Jawa Timur, etc.
- **Kalimantan Region**: All 5 Kalimantan provinces
- **Sulawesi Region**: All 6 Sulawesi provinces
- **Eastern Indonesia**: Papua provinces, Maluku, Nusa Tenggara, Bali

### Map Features
- **Interactive Markers**: Click for province details
- **Color Coding**: Risk-based visualization
- **Size Coding**: Uncertainty-based marker sizing
- **Multiple Layers**: Different base map options
- **Export Options**: Save maps and data

### Regional Analysis
- **Automatic Region Detection**: Groups provinces by major regions
- **Risk Pattern Analysis**: Identifies geographic clustering
- **Performance Comparison**: Best/worst performing regions
- **Infrastructure Insights**: Access and intervention difficulty

## ⚙️ Configuration

### Model Parameters
```python
# In sidebar "Analysis Settings"
- Number of Trees: 50-500 (default: 200)
- Max Tree Depth: None, 5, 10, 15, 20, 30
- CV Folds: 3-10 (default: 5)
```

### Geographic Settings
```python
# Customize in src/geo_visualization.py
INDONESIA_PROVINCES_COORDS = {
    'Province_Name': {
        'lat': latitude,
        'lon': longitude, 
        'code': 'ISO_code'
    }
}
```

## 🔧 Development

### Adding New Features

#### Custom Visualizations
```python
# In src/visualization.py
def create_custom_chart(data):
    # Your visualization code
    return fig

# Import in main dashboard
from src.visualization import create_custom_chart
```

#### New Geographic Analysis
```python
# In src/geo_visualization.py
def create_custom_map(scenario_data, risk_data):
    # Your mapping code
    return fig

# Use in dashboard
fig = create_custom_map(data1, data2)
st.plotly_chart(fig)
```

### Testing
```bash
# Run basic tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_geo_visualization.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 📝 Data Sources

### Required Data Format
- **Time Series**: Multiple years per province
- **Geographic**: Indonesian province names (standard)
- **Indicators**: Poverty, education, health, infrastructure
- **Target**: Food security composite score (1-6)

### Sample Data
The included sample dataset contains:
- **6 years** of data (2018-2023)
- **15 provinces** across major regions
- **Realistic values** based on Indonesian statistics
- **Complete coverage** for all required indicators

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'folium'
pip install folium

# Error: No module named 'src'
# Run from project root directory
cd food-security-dashboard
streamlit run streamlit_dashboard.py
```

#### 2. Data Validation Errors
```
# Missing required columns
- Check CSV column names match exactly
- Ensure no extra spaces in column names
- Verify data types (numeric vs string)

# Invalid data ranges
- Komposit scores should be 1-6
- Percentages should be 0-100
- Years should be reasonable (2000-2030)
```

#### 3. Geographic Visualization Issues
```
# Provinces not showing on map
- Check province name spelling
- Verify names match INDONESIA_PROVINCES_COORDS
- Some provinces may use alternative names (NTB vs Nusa Tenggara Barat)

# Map not loading
- Check internet connection (needed for base maps)
- Try different map types if one fails
- Refresh browser if visualization seems stuck
```

#### 4. Performance Issues
```bash
# Large datasets (>10,000 rows)
- Use data sampling for initial analysis
- Consider upgrading hardware
- Enable performance optimizations:

pip install numba bottleneck
```

### Getting Help

1. **Check the console** for detailed error messages
2. **Review data validation** warnings and suggestions
3. **Try sample data** to verify installation
4. **Check GitHub issues** for known problems
5. **Contact support** with detailed error descriptions

## 🤝 Contributing

### Development Setup
```bash
# Fork the repository
git clone https://github.com/your-username/food-security-dashboard.git

# Create feature branch
git checkout -b feature/new-feature

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Make changes and test
python -m pytest
black src/
flake8 src/

# Submit pull request
```

### Guidelines
- **Follow PEP 8** style guidelines
- **Add tests** for new features
- **Update documentation** as needed
- **Test with sample data** before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the excellent web app framework
- **Plotly** for interactive visualizations
- **Folium** for geographic mapping capabilities
- **Indonesian Government** for food security data standards
- **Open Source Community** for various Python packages

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/food-security-dashboard/issues)
- **Email**: your-email@example.com
- **Documentation**: [Project Wiki](https://github.com/your-username/food-security-dashboard/wiki)

---

**Made with ❤️ for food security analysis and policy decision support**