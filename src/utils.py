# =============================================================================
# CARD COMPONENTS FOR STREAMLIT DASHBOARD
# Reusable card components with guaranteed text visibility
# =============================================================================

def create_metric_card(title, value, description="", icon="📊"):
    """Create a standardized metric card with guaranteed text visibility"""
    return f"""
    <div style="
        background-color: #ffffff; 
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 4px solid #3498db; 
        margin: 0.5rem 0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <h3 style="
            color: #2c3e50 !important; 
            font-size: 0.9rem !important; 
            margin-bottom: 0.5rem !important; 
            font-weight: 600 !important;
            text-align: left;
        ">{icon} {title}</h3>
        <h2 style="
            color: #1f77b4 !important; 
            font-size: 2rem !important; 
            margin: 0.5rem 0 !important; 
            font-weight: 700 !important;
            text-align: center;
        ">{value}</h2>
        <p style="
            color: #495057 !important; 
            font-size: 0.8rem !important; 
            margin: 0.25rem 0 0 0 !important;
            text-align: center;
            opacity: 0.8;
        ">{description}</p>
    </div>
    """

def create_status_card(title, content, card_type="info", icon="ℹ️"):
    """Create status cards with different types"""
    
    type_configs = {
        "info": {
            "bg_color": "#d1ecf1",
            "border_color": "#17a2b8", 
            "text_color": "#0c5460"
        },
        "success": {
            "bg_color": "#d4edda",
            "border_color": "#28a745",
            "text_color": "#155724"
        },
        "warning": {
            "bg_color": "#fff3cd",
            "border_color": "#ffc107",
            "text_color": "#856404"
        },
        "danger": {
            "bg_color": "#f8d7da",
            "border_color": "#dc3545",
            "text_color": "#721c24"
        }
    }
    
    config = type_configs.get(card_type, type_configs["info"])
    
    return f"""
    <div style="
        background-color: {config['bg_color']}; 
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 4px solid {config['border_color']}; 
        margin: 0.5rem 0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-height: 100px;
    ">
        <h4 style="
            color: {config['text_color']} !important; 
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
            font-size: 1rem !important;
        ">{icon} {title}</h4>
        <div style="color: {config['text_color']} !important; font-size: 0.9rem;">
            {content}
        </div>
    </div>
    """

# =============================================================================
# EXISTING UTILITY FUNCTIONS
# =============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from .config import ProjectPaths, FoodSecurityConfig

# Set up logging
logger = logging.getLogger(__name__)

def validate_data_format(df: pd.DataFrame, config: FoodSecurityConfig = None) -> Dict[str, Any]:
    """
    Validate input data format and completeness
    
    Args:
        df: Input dataframe
        config: Configuration object
    
    Returns:
        Dictionary with validation results
    """
    if config is None:
        config = FoodSecurityConfig()
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Required columns
    required_columns = config.PREDICTOR_VARIABLES + [
        config.TARGET_VARIABLE, 'Provinsi', 'Kabupaten', 'Tahun'
    ]
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if 'Tahun' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Tahun']):
            validation_results['errors'].append("'Tahun' column must be numeric")
    
    # Check for missing values in critical columns
    if config.TARGET_VARIABLE in df.columns:
        missing_target = df[config.TARGET_VARIABLE].isnull().sum()
        if missing_target > 0:
            validation_results['warnings'].append(
                f"Target variable has {missing_target} missing values ({missing_target/len(df)*100:.1f}%)"
            )
    
    # Check target variable range
    if config.TARGET_VARIABLE in df.columns:
        target_values = df[config.TARGET_VARIABLE].dropna().unique()
        valid_range = list(config.KOMPOSIT_MAPPING.keys())
        invalid_values = [val for val in target_values if val not in valid_range]
        if invalid_values:
            validation_results['warnings'].append(
                f"Target variable contains unexpected values: {invalid_values}"
            )
    
    # Summary statistics
    validation_results['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'years_range': f"{df['Tahun'].min()}-{df['Tahun'].max()}" if 'Tahun' in df.columns else 'Unknown',
        'provinces_count': df['Provinsi'].nunique() if 'Provinsi' in df.columns else 0,
        'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    return validation_results

def clean_data(df: pd.DataFrame, config: FoodSecurityConfig = None) -> pd.DataFrame:
    """
    Clean and preprocess the input data
    
    Args:
        df: Input dataframe
        config: Configuration object
    
    Returns:
        Cleaned dataframe
    """
    if config is None:
        config = FoodSecurityConfig()
    
    logger.info("Starting data cleaning process...")
    
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Clean string columns
    string_columns = ['Provinsi', 'Kabupaten']
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip()
            df_clean[col] = df_clean[col].str.title()
    
    # Handle numeric columns
    numeric_columns = config.PREDICTOR_VARIABLES + [config.TARGET_VARIABLE]
    for col in numeric_columns:
        if col in df_clean.columns:
            # Convert to numeric, coercing errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove rows with missing target variable
    if config.TARGET_VARIABLE in df_clean.columns:
        before_target_cleaning = len(df_clean)
        df_clean = df_clean.dropna(subset=[config.TARGET_VARIABLE])
        target_missing_removed = before_target_cleaning - len(df_clean)
        if target_missing_removed > 0:
            logger.info(f"Removed {target_missing_removed} rows with missing target variable")
    
    # Handle outliers (optional - using IQR method)
    for col in config.PREDICTOR_VARIABLES:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 3 * IQR  # More conservative than 1.5 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Count outliers but don't remove them automatically
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                logger.info(f"Column '{col}' has {outliers} potential outliers")
    
    logger.info(f"Data cleaning completed. Final dataset: {len(df_clean)} rows")
    
    return df_clean

def save_model(model: Any, model_name: str, metadata: Dict = None) -> str:
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        model_name: Name for the saved model
        metadata: Additional metadata to save
    
    Returns:
        Path to saved model file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = ProjectPaths.MODELS_DIR / filename
    
    # Create models directory if it doesn't exist
    ProjectPaths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    if metadata:
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model saved to: {filepath}")
    return str(filepath)

def load_model(model_path: str) -> Tuple[Any, Dict]:
    """
    Load saved model from disk
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Tuple of (model, metadata)
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata if exists
    metadata_path = Path(model_path).with_suffix('.json')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    logger.info(f"Model loaded from: {model_path}")
    return model, metadata

def save_results(results: Dict, filename: str, file_format: str = 'csv') -> str:
    """
    Save analysis results to disk
    
    Args:
        results: Dictionary containing results
        filename: Name for the output file
        file_format: Format to save ('csv', 'json', 'excel')
    
    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if file_format == 'csv':
        filepath = ProjectPaths.REPORTS_DIR / f"{filename}_{timestamp}.csv"
        if isinstance(results, dict):
            # Convert dict to DataFrame if possible
            df = pd.DataFrame(results)
            df.to_csv(filepath, index=False)
        elif isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
    
    elif file_format == 'json':
        filepath = ProjectPaths.REPORTS_DIR / f"{filename}_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif file_format == 'excel':
        filepath = ProjectPaths.REPORTS_DIR / f"{filename}_{timestamp}.xlsx"
        if isinstance(results, dict):
            with pd.ExcelWriter(filepath) as writer:
                for sheet_name, data in results.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=sheet_name, index=False)
        elif isinstance(results, pd.DataFrame):
            results.to_excel(filepath, index=False)
    
    logger.info(f"Results saved to: {filepath}")
    return str(filepath)

def create_sample_data(n_provinces: int = 10, n_years: int = 6, 
                      districts_per_province: int = 15) -> pd.DataFrame:
    """
    Create sample data for testing and demonstration
    
    Args:
        n_provinces: Number of provinces to generate
        n_years: Number of years of data
        districts_per_province: Average number of districts per province
    
    Returns:
        Sample dataframe
    """
    np.random.seed(42)
    
    provinces = [
        'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Sumatera Utara',
        'Sumatera Barat', 'Sulawesi Selatan', 'Kalimantan Timur', 'Bali', 'NTB',
        'Sumatera Selatan', 'Lampung', 'Kalimantan Selatan', 'Sulawesi Utara', 'Yogyakarta'
    ][:n_provinces]
    
    data = []
    current_year = datetime.now().year
    start_year = current_year - n_years + 1
    
    for year in range(start_year, current_year + 1):
        for province in provinces:
            # Number of districts varies by province
            n_districts = np.random.randint(
                max(1, districts_per_province - 5), 
                districts_per_province + 10
            )
            
            # Province-specific base characteristics
            base_poverty = np.random.uniform(5, 30)
            base_education = np.random.uniform(6, 12)
            base_health_ratio = np.random.uniform(0.5, 3.0)
            base_life_expectancy = np.random.uniform(65, 78)
            
            for i in range(n_districts):
                # Add year trend and random variation
                year_effect = (year - start_year) * 0.1
                
                # Create correlated variables
                poverty = max(0, base_poverty + np.random.normal(-year_effect, 4))
                education = base_education + np.random.normal(year_effect * 0.3, 1.5)
                health_ratio = max(0.1, base_health_ratio + np.random.normal(year_effect * 0.05, 0.4))
                life_expectancy = base_life_expectancy + np.random.normal(year_effect * 0.15, 2)
                
                # Food security composite score (influenced by other variables)
                composite_base = 6 - (poverty / 10) + (education / 3) + (health_ratio / 2)
                composite = max(1, min(6, int(composite_base + np.random.normal(0, 0.8))))
                
                data.append({
                    'Tahun': year,
                    'Provinsi': province,
                    'Kabupaten': f'{province}_Kab_{i+1:02d}',
                    'Kemiskinan (%)': round(poverty, 2),
                    'Pengeluaran Pangan (%)': round(np.random.uniform(35, 75), 2),
                    'Tanpa Air Bersih (%)': round(max(0, np.random.uniform(5, 45) - year_effect), 2),
                    'Lama Sekolah Perempuan (tahun)': round(education, 2),
                    'Rasio Tenaga Kesehatan': round(health_ratio, 3),
                    'Angka Harapan Hidup (tahun)': round(life_expectancy, 1),
                    'Komposit': composite
                })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated sample data: {len(df)} records, {df['Provinsi'].nunique()} provinces, {n_years} years")
    
    return df

def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'max_error': np.max(np.abs(y_true - y_pred)),
        'std_residuals': np.std(y_true - y_pred)
    }
    
    return metrics

def format_number(value: float, format_type: str = 'decimal') -> str:
    """
    Format numbers for display
    
    Args:
        value: Number to format
        format_type: Type of formatting ('decimal', 'percentage', 'currency')
    
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return 'N/A'
    
    if format_type == 'decimal':
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        else:
            return f"{value:.3f}"
    elif format_type == 'percentage':
        return f"{value:.1f}%"
    elif format_type == 'currency':
        return f"Rp {value:,.0f}"
    else:
        return str(value)

def generate_report_summary(forecaster_results: Dict) -> Dict[str, Any]:
    """
    Generate executive summary from forecaster results
    
    Args:
        forecaster_results: Results from FoodSecurityForecaster
    
    Returns:
        Summary dictionary
    """
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'model_performance': {},
        'key_insights': [],
        'recommendations': [],
        'risk_assessment': {},
        'data_quality': {}
    }
    
    # Model performance summary
    if 'cv_results' in forecaster_results:
        cv_results = forecaster_results['cv_results']
        summary['model_performance'] = {
            'mean_r2': cv_results['r2'].mean(),
            'r2_stability': cv_results['r2'].std(),
            'mean_rmse': cv_results['rmse'].mean(),
            'cv_folds': len(cv_results),
            'performance_grade': 'Excellent' if cv_results['r2'].mean() > 0.8 else 
                               'Good' if cv_results['r2'].mean() > 0.6 else 'Moderate'
        }
    
    # Key insights
    if 'feature_importance' in forecaster_results:
        top_feature = forecaster_results['feature_importance'].iloc[0]['Feature']
        summary['key_insights'].append(f"Most important factor: {top_feature}")
    
    # Risk assessment
    if 'risk_assessment' in forecaster_results:
        risk_data = forecaster_results['risk_assessment']
        status_quo = risk_data[risk_data['Scenario'] == 'Status Quo']
        high_risk_count = len(status_quo[status_quo['Risk_Level'].isin(['Very High Risk', 'High Risk'])])
        
        summary['risk_assessment'] = {
            'high_risk_provinces': high_risk_count,
            'total_provinces': len(status_quo),
            'risk_percentage': (high_risk_count / len(status_quo)) * 100
        }
    
    return summary

def export_dashboard_data(forecaster, output_format: str = 'excel') -> str:
    """
    Export all dashboard data to files
    
    Args:
        forecaster: FoodSecurityForecaster instance
        output_format: Export format ('excel', 'csv')
    
    Returns:
        Path to exported file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format == 'excel':
        filepath = ProjectPaths.REPORTS_DIR / f"food_security_analysis_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Model performance
            if forecaster.cv_results is not None:
                forecaster.cv_results.to_excel(writer, sheet_name='Model Performance', index=False)
            
            # Feature importance
            if forecaster.feature_importance is not None:
                forecaster.feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
            
            # Scenario predictions
            if forecaster.scenario_predictions is not None:
                forecaster.scenario_predictions.to_excel(writer, sheet_name='Scenario Predictions', index=False)
            
            # Risk assessment
            if forecaster.risk_assessment is not None:
                status_quo_risk = forecaster.risk_assessment[
                    forecaster.risk_assessment['Scenario'] == 'Status Quo'
                ]
                status_quo_risk.to_excel(writer, sheet_name='Risk Assessment', index=False)
    
    elif output_format == 'csv':
        # Create a directory for CSV files
        csv_dir = ProjectPaths.REPORTS_DIR / f"food_security_analysis_{timestamp}"
        csv_dir.mkdir(exist_ok=True)
        
        # Save individual CSV files
        if forecaster.cv_results is not None:
            forecaster.cv_results.to_csv(csv_dir / 'model_performance.csv', index=False)
        
        if forecaster.feature_importance is not None:
            forecaster.feature_importance.to_csv(csv_dir / 'feature_importance.csv', index=False)
        
        if forecaster.scenario_predictions is not None:
            forecaster.scenario_predictions.to_csv(csv_dir / 'scenario_predictions.csv', index=False)
        
        if forecaster.risk_assessment is not None:
            status_quo_risk = forecaster.risk_assessment[
                forecaster.risk_assessment['Scenario'] == 'Status Quo'
            ]
            status_quo_risk.to_csv(csv_dir / 'risk_assessment.csv', index=False)
        
        filepath = csv_dir
    
    logger.info(f"Dashboard data exported to: {filepath}")
    return str(filepath)

def setup_project_structure():
    """
    Set up complete project directory structure
    """
    logger.info("Setting up project directory structure...")
    
    # Create all directories
    ProjectPaths.create_directories()
    
    # Create additional subdirectories
    additional_dirs = [
        ProjectPaths.ROOT_DIR / "logs",
        ProjectPaths.ASSETS_DIR / "css",
        ProjectPaths.ASSETS_DIR / "images", 
        ProjectPaths.ASSETS_DIR / "templates",
        ProjectPaths.DOCS_DIR / "images"
    ]
    
    for directory in additional_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create essential files
    essential_files = [
        (ProjectPaths.ROOT_DIR / ".gitignore", get_gitignore_content()),
        (ProjectPaths.ROOT_DIR / "setup.py", get_setup_py_content()),
        (ProjectPaths.DATA_DIR / "README.md", get_data_readme_content()),
        (ProjectPaths.CONFIG_DIR / "model_config.yaml", get_model_config_yaml())
    ]
    
    for filepath, content in essential_files:
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
    
    logger.info("Project structure setup completed!")

def get_gitignore_content() -> str:
    """Get .gitignore file content"""
    return """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
outputs/models/*.pkl
outputs/models/*.json
outputs/reports/*.csv
outputs/reports/*.xlsx
outputs/figures/*.png
outputs/figures/*.jpg
outputs/predictions/*.csv
logs/*.log
data/raw/*.csv
data/processed/*.csv
!data/raw/sample_data.csv

# Streamlit
.streamlit/
"""

def get_setup_py_content() -> str:
    """Get setup.py file content"""
    return '''from setuptools import setup, find_packages

setup(
    name="food-security-forecasting",
    version="1.0.0",
    description="Machine Learning System for Food Security Forecasting",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "streamlit>=1.25.0",
        "plotly>=5.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "shap>=0.41.0",
        "pyyaml>=6.0",
        "openpyxl>=3.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
'''

def get_data_readme_content() -> str:
    """Get data folder README content"""
    return """# Data Directory

This directory contains all data files for the Food Security Forecasting project.

## Structure

- `raw/` - Original, unmodified data files
- `processed/` - Cleaned and preprocessed data
- `external/` - External datasets and references

## Data Format Requirements

Your CSV files should contain the following columns:

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| Tahun | Year | Integer | 2023 |
| Provinsi | Province name | String | "Jawa Barat" |
| Kabupaten | District/Regency | String | "Bandung" |
| Kemiskinan (%) | Poverty rate | Float | 12.5 |
| Pengeluaran Pangan (%) | Food expenditure | Float | 45.2 |
| Tanpa Air Bersih (%) | Without clean water | Float | 15.3 |
| Lama Sekolah Perempuan (tahun) | Women's education | Float | 9.8 |
| Rasio Tenaga Kesehatan | Healthcare ratio | Float | 1.8 |
| Angka Harapan Hidup (tahun) | Life expectancy | Float | 69.2 |
| Komposit | Food security score | Integer | 4 |

## Usage

1. Place your raw data files in the `raw/` directory
2. Use the dashboard's data loading feature to upload and validate your data
3. Processed data will be automatically saved to the `processed/` directory
"""

def get_model_config_yaml() -> str:
    """Get model configuration YAML content"""
    return """# Food Security Forecasting Model Configuration

# Random seed for reproducibility
random_state: 42

# Target variable
target_variable: "Komposit"

# Predictor variables
predictor_variables:
  - "Kemiskinan (%)"
  - "Pengeluaran Pangan (%)"
  - "Tanpa Air Bersih (%)"
  - "Lama Sekolah Perempuan (tahun)"
  - "Rasio Tenaga Kesehatan"
  - "Angka Harapan Hidup (tahun)"

# Model hyperparameters
param_grid:
  n_estimators: [100, 200, 300]
  max_depth: [null, 10, 20, 30]
  min_samples_split: [2, 5, 10]
  min_samples_leaf: [1, 2, 4]
  max_features: ["sqrt", "log2", null]
  bootstrap: [true, false]

# Cross-validation settings
min_train_years: 3

# Risk assessment thresholds
risk_thresholds:
  low_security_threshold: 3.0
  uncertainty_threshold: 0.5
  high_uncertainty_threshold: 0.8

# Bootstrap settings
n_bootstrap_default: 50
n_bootstrap_fast: 25

# Visualization settings
plot_config:
  figure_size: [12, 8]
  dpi: 300
  style: "whitegrid"
  palette: "viridis"
  font_scale: 1.2
"""