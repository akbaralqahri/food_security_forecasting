# =============================================================================
# CONFIGURATION MODULE - FOOD SECURITY FORECASTING
# Centralized configuration management
# =============================================================================

import os
from pathlib import Path
from typing import List, Dict, Any
import yaml

class ProjectPaths:
    """Project directory paths"""
    
    # Get project root directory
    ROOT_DIR = Path(__file__).parent.parent
    
    # Source code
    SRC_DIR = ROOT_DIR / "src"
    
    # Data directories
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    
    # Output directories
    OUTPUT_DIR = ROOT_DIR / "outputs"
    MODELS_DIR = OUTPUT_DIR / "models"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    
    # Configuration
    CONFIG_DIR = ROOT_DIR / "config"
    
    # Assets
    ASSETS_DIR = ROOT_DIR / "assets"
    
    # Notebooks
    NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.EXTERNAL_DATA_DIR,
            cls.OUTPUT_DIR, cls.MODELS_DIR, cls.REPORTS_DIR, cls.FIGURES_DIR, cls.PREDICTIONS_DIR,
            cls.CONFIG_DIR, cls.ASSETS_DIR, cls.NOTEBOOKS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files
            gitkeep_file = directory / ".gitkeep"
            if not gitkeep_file.exists():
                gitkeep_file.touch()

class FoodSecurityConfig:
    """Main configuration class for Food Security Forecasting"""
    
    # Random seed for reproducibility
    RANDOM_STATE = 42
    
    # Model configuration
    PREDICTOR_VARIABLES = [
        'Kemiskinan (%)',
        'Pengeluaran Pangan (%)',
        'Tanpa Air Bersih (%)',
        'Lama Sekolah Perempuan (tahun)',
        'Rasio Tenaga Kesehatan',
        'Angka Harapan Hidup (tahun)'
    ]
    
    TARGET_VARIABLE = 'Komposit'
    
    # Food security category mapping
    KOMPOSIT_MAPPING = {
        6: 'Sangat Tahan',
        5: 'Tahan',
        4: 'Agak Tahan',
        3: 'Agak Rentan',
        2: 'Rentan',
        1: 'Sangat Rentan'
    }
    
    # Color mapping for visualizations
    COLOR_MAPPING = {
        'Sangat Tahan': '#2E8B57',      # Sea Green
        'Tahan': '#32CD32',             # Lime Green
        'Agak Tahan': '#FFD700',        # Gold
        'Agak Rentan': '#FFA500',       # Orange
        'Rentan': '#FF6347',            # Tomato
        'Sangat Rentan': '#DC143C'      # Crimson
    }
    
    # Model hyperparameter grid
    PARAM_GRID = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    # Fast parameter grid for development/testing
    PARAM_GRID_FAST = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }
    
    # Time series cross-validation configuration
    MIN_TRAIN_YEARS = 3
    
    # Scenario analysis configuration
    SCENARIO_DEFINITIONS = {
        'conservative_improvement': {
            'description': 'Conservative policy improvements (5% improvement)',
            'poverty_reduction': 0.95,
            'water_access_improvement': 0.95,
            'food_expenditure_reduction': 0.97,
            'education_improvement': 1.02,
            'health_improvement': 1.05,
            'life_expectancy_improvement': 1.01
        },
        'moderate_improvement': {
            'description': 'Moderate policy improvements (10% improvement)',
            'poverty_reduction': 0.90,
            'water_access_improvement': 0.90,
            'food_expenditure_reduction': 0.95,
            'education_improvement': 1.05,
            'health_improvement': 1.10,
            'life_expectancy_improvement': 1.02
        },
        'optimistic_improvement': {
            'description': 'Optimistic policy improvements (15% improvement)',
            'poverty_reduction': 0.85,
            'water_access_improvement': 0.85,
            'food_expenditure_reduction': 0.92,
            'education_improvement': 1.08,
            'health_improvement': 1.15,
            'life_expectancy_improvement': 1.03
        }
    }
    
    # Risk assessment thresholds
    RISK_THRESHOLDS = {
        'low_security_threshold': 3.0,
        'uncertainty_threshold': 0.5,
        'high_uncertainty_threshold': 0.8
    }
    
    # Bootstrap configuration for uncertainty quantification
    N_BOOTSTRAP_DEFAULT = 50
    N_BOOTSTRAP_FAST = 25
    
    # Visualization configuration
    PLOT_CONFIG = {
        'figure_size': (12, 8),
        'dpi': 300,
        'style': 'whitegrid',
        'palette': 'viridis',
        'font_scale': 1.2
    }
    
    @classmethod
    def load_from_yaml(cls, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = ProjectPaths.CONFIG_DIR / "model_config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config_dict = yaml.safe_load(file)
                
            # Update class attributes with YAML values
            for key, value in config_dict.items():
                if hasattr(cls, key.upper()):
                    setattr(cls, key.upper(), value)
    
    @classmethod
    def save_to_yaml(cls, config_path: str = None):
        """Save current configuration to YAML file"""
        if config_path is None:
            config_path = ProjectPaths.CONFIG_DIR / "model_config.yaml"
        
        config_dict = {
            'random_state': cls.RANDOM_STATE,
            'predictor_variables': cls.PREDICTOR_VARIABLES,
            'target_variable': cls.TARGET_VARIABLE,
            'param_grid': cls.PARAM_GRID,
            'min_train_years': cls.MIN_TRAIN_YEARS,
            'scenario_definitions': cls.SCENARIO_DEFINITIONS,
            'risk_thresholds': cls.RISK_THRESHOLDS,
            'n_bootstrap_default': cls.N_BOOTSTRAP_DEFAULT,
            'plot_config': cls.PLOT_CONFIG
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)

class DashboardConfig:
    """Configuration for Streamlit dashboard"""
    
    # Page configuration
    PAGE_CONFIG = {
        'page_title': "Food Security Forecasting Dashboard",
        'page_icon': "🌾",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    }
    
    # Sidebar configuration
    SIDEBAR_CONFIG = {
        'title': "📊 Dashboard Controls",
        'data_section_title': "📁 Data Loading",
        'analysis_section_title': "🔧 Analysis Settings",
        'info_section_title': "ℹ️ Information"
    }
    
    # Color scheme
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#9467bd',
        'light': '#17becf'
    }
    
    # Tab configuration
    TABS = [
        "📊 Data Overview",
        "🤖 Model Performance", 
        "🎯 Feature Analysis",
        "🔮 Scenario Forecasting",
        "⚠️ Risk Assessment",
        "📋 Summary Report"
    ]
    
    # File upload settings
    UPLOAD_CONFIG = {
        'max_file_size': 200,  # MB
        'accepted_types': ['csv'],
        'encoding': 'utf-8'
    }

class LoggingConfig:
    """Logging configuration"""
    
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'
    LOG_FILE = ProjectPaths.ROOT_DIR / 'logs' / 'food_security.log'
    
    @classmethod
    def setup_logging(cls):
        """Set up logging configuration"""
        import logging
        
        # Create logs directory
        cls.LOG_FILE.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )

# Initialize project directories on import
ProjectPaths.create_directories()