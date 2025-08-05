# 🚀 Food Security Forecasting - Complete Setup Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Project Structure Setup](#project-structure-setup)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Setup](#advanced-setup)

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Recommended Development Environment
- **Python**: 3.9 or 3.10
- **RAM**: 16GB for large datasets
- **Storage**: 5GB for full development setup
- **IDE**: VS Code, PyCharm, or Jupyter Lab

## Installation Methods

### Method 1: Quick Start (Recommended for Users)

```bash
# 1. Clone or download the project
git clone <repository-url>
cd food-security-forecasting

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the dashboard
streamlit run streamlit_dashboard.py
```

### Method 2: Development Setup (For Contributors)

```bash
# 1. Clone the repository
git clone <repository-url>
cd food-security-forecasting

# 2. Create conda environment (recommended for development)
conda create -n food-security python=3.9
conda activate food-security

# 3. Install dependencies with development tools
pip install -r requirements.txt
pip install -e .  # Install in development mode

# 4. Install additional development tools
pip install pytest pytest-cov black flake8 jupyter

# 5. Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Method 3: Docker Setup (For Production)

```bash
# 1. Build Docker image
docker build -t food-security-forecasting .

# 2. Run container
docker run -p 8501:8501 food-security-forecasting
```

## Project Structure Setup

### Automatic Setup
Run the setup script to create the complete directory structure:

```python
# Run this Python script to set up directories
from src.utils import setup_project_structure
setup_project_structure()
```

### Manual Setup
If automatic setup doesn't work, create these directories manually:

```
food_security_forecasting/
├── src/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── outputs/
│   ├── models/
│   ├── reports/
│   ├── figures/
│   └── predictions/
├── config/
├── notebooks/
├── tests/
├── docs/
├── assets/
│   ├── css/
│   ├── images/
│   └── templates/
└── logs/
```

## Configuration

### 1. Environment Variables (Optional)
Create a `.env` file in the project root:

```bash
# .env file
PYTHONPATH=./src
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
LOG_LEVEL=INFO
```

### 2. Model Configuration
Edit `config/model_config.yaml` to customize model parameters:

```yaml
# Example customization
param_grid:
  n_estimators: [50, 100, 200]  # Reduced for faster training
  max_depth: [10, 20]           # Limited depth
  
n_bootstrap_default: 25  # Reduced for faster uncertainty estimation
```

### 3. Dashboard Configuration
Customize dashboard settings in `streamlit_dashboard.py`:

```python
# Page configuration
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🌾",
    layout="wide"
)
```

## Running the Application

### 1. Basic Dashboard
```bash
# Activate virtual environment first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run dashboard
streamlit run streamlit_dashboard.py
```

The dashboard will open at `http://localhost:8501`

### 2. Using Example Script
```bash
# Run example analysis
python example_usage.py
```

### 3. Jupyter Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks in the notebooks/ directory
```

### 4. Command Line Interface (Advanced)
```bash
# Run analysis from command line
python -c "
from src.food_security_forecasting import FoodSecurityForecaster
from src.utils import create_sample_data

# Create sample data and run analysis
df = create_sample_data()
forecaster = FoodSecurityForecaster()
results = forecaster.run_full_analysis(df)
print('Analysis completed!')
"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:./src"

# Or on Windows:
set PYTHONPATH=%PYTHONPATH%;.\src
```

#### 2. Streamlit Port Issues
```bash
# Error: Port 8501 is already in use
# Solution: Use different port
streamlit run streamlit_dashboard.py --server.port 8502
```

#### 3. Memory Issues with Large Datasets
```python
# Reduce memory usage by modifying config
config = FoodSecurityConfig()
config.N_BOOTSTRAP_DEFAULT = 10  # Reduce bootstrap samples
config.PARAM_GRID = config.PARAM_GRID_FAST  # Use fast parameter grid
```

#### 4. Slow Performance
```python
# Speed up analysis
config = FoodSecurityConfig()
config.PARAM_GRID = {
    'n_estimators': [100],  # Single value
    'max_depth': [20],      # Single value
    # ... reduce other parameters
}
```

#### 5. Data Validation Errors
```python
# Check data format
from src.utils import validate_data_format
validation_results = validate_data_format(your_dataframe)
print(validation_results)
```

### Debug Mode
Enable debug mode for detailed error information:

```bash
# Run with debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# ... run your code
"
```

## Advanced Setup

### 1. Custom Data Sources
To add support for new data sources:

```python
# Extend the DataProcessor class
from src.food_security_forecasting import DataProcessor

class CustomDataProcessor(DataProcessor):
    def load_from_database(self, connection_string):
        # Add database loading logic
        pass
    
    def load_from_api(self, api_endpoint):
        # Add API loading logic
        pass
```

### 2. Custom Models
Add new machine learning models:

```python
# Extend ModelTrainer class
from src.food_security_forecasting import ModelTrainer
from sklearn.ensemble import GradientBoostingRegressor

class ExtendedModelTrainer(ModelTrainer):
    def train_gradient_boosting(self, X, y):
        model = GradientBoostingRegressor()
        # Add training logic
        return model
```

### 3. Custom Visualizations
Add new visualization types:

```python
# Extend FoodSecurityVisualizer
from src.visualization import FoodSecurityVisualizer

class CustomVisualizer(FoodSecurityVisualizer):
    def plot_custom_analysis(self, data):
        # Add custom plotting logic
        pass
```

### 4. Integration with External Systems
```python
# Example: Integration with database
import sqlalchemy

def load_from_database(connection_string, query):
    engine = sqlalchemy.create_engine(connection_string)
    df = pd.read_sql(query, engine)
    return df

# Example: Integration with cloud storage
import boto3

def load_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    # Download and load data
    pass
```

### 5. Automated Scheduling
Set up automated analysis runs:

```python
# Example using schedule library
import schedule
import time

def run_daily_analysis():
    # Load latest data
    # Run analysis
    # Send reports
    pass

schedule.every().day.at("02:00").do(run_daily_analysis)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## Performance Optimization

### 1. For Large Datasets
- Use chunked processing
- Reduce bootstrap samples
- Use feature selection
- Implement data sampling

### 2. For Production Deployment
- Use caching with `@st.cache_data`
- Implement lazy loading
- Use database connections
- Add monitoring and logging

### 3. Memory Management
```python
# Example memory optimization
import gc

def memory_efficient_analysis(df):
    # Process in chunks
    chunk_size = 10000
    results = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        # Process chunk
        result = process_chunk(chunk)
        results.append(result)
        
        # Clean up memory
        del chunk
        gc.collect()
    
    return combine_results(results)
```

## Deployment Options

### 1. Local Deployment
- Run on local machine
- Use for development and testing

### 2. Cloud Deployment
```bash
# Example: Deploy to Streamlit Cloud
# 1. Push code to GitHub
# 2. Connect repository to Streamlit Cloud
# 3. Configure deployment settings
```

### 3. Docker Deployment
```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 4. Server Deployment
```bash
# Example: Deploy to Ubuntu server
# 1. Set up reverse proxy (nginx)
# 2. Use process manager (pm2, supervisor)
# 3. Configure SSL certificate
```

## Support and Maintenance

### Getting Help
1. Check this setup guide
2. Review error messages and logs
3. Check the troubleshooting section
4. Review the main README.md
5. Check the documentation in `docs/`

### Updates and Maintenance
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Run tests after updates
python -m pytest tests/

# Update model configuration if needed
```

### Monitoring
- Check logs in `logs/` directory
- Monitor dashboard performance
- Review analysis results for quality
- Set up alerts for failures

---

🎉 **Congratulations!** You should now have a fully functional Food Security Forecasting system. If you encounter any issues, refer to the troubleshooting section or check the logs for detailed error information.