# =============================================================================
# FOOD SECURITY FORECASTING - STRUCTURED MAIN MODULE
# Enhanced version with modular design and improved organization
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

# Optional import - SHAP for advanced explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Advanced model explanations will be skipped.")

class FoodSecurityConfig:
    """Configuration class for the food security forecasting project"""
    
    RANDOM_STATE = 42
    
    PREDICTOR_VARIABLES = [
        'Kemiskinan (%)',
        'Pengeluaran Pangan (%)',
        'Tanpa Air Bersih (%)',
        'Lama Sekolah Perempuan (tahun)',
        'Rasio Tenaga Kesehatan',
        'Angka Harapan Hidup (tahun)'
    ]
    
    TARGET_VARIABLE = 'Komposit'
    
    KOMPOSIT_MAPPING = {
        6: 'Sangat Tahan',
        5: 'Tahan', 
        4: 'Agak Tahan',
        3: 'Agak Rentan',
        2: 'Rentan',
        1: 'Sangat Rentan'
    }
    
    PARAM_GRID = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

class DataProcessor:
    """Class for data preprocessing and preparation"""
    
    def __init__(self, config):
        self.config = config
        np.random.seed(config.RANDOM_STATE)
    
    def load_and_validate_data(self, df):
        """Load and validate the input data"""
        print("📥 Loading and validating data...")
        
        # Check available columns
        available_columns = [col for col in self.config.PREDICTOR_VARIABLES if col in df.columns]
        missing_columns = [col for col in self.config.PREDICTOR_VARIABLES if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            self.config.PREDICTOR_VARIABLES = available_columns
        
        print(f"📊 Dataset Overview:")
        print(f"  • Shape: {df.shape}")
        print(f"  • Years: {df['Tahun'].min()} - {df['Tahun'].max()}")
        print(f"  • Provinces: {df['Provinsi'].nunique()}")
        print(f"  • Predictor variables: {len(self.config.PREDICTOR_VARIABLES)}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for modeling"""
        print("⚙️ Preprocessing data...")
        
        # Create working dataset
        required_columns = (self.config.PREDICTOR_VARIABLES + 
                        [self.config.TARGET_VARIABLE, 'Provinsi', 'Kabupaten', 'Tahun'])
        df_model = df[required_columns].copy()
        
        # Remove rows with missing values
        initial_rows = len(df_model)
        df_model = df_model.dropna(subset=self.config.PREDICTOR_VARIABLES + [self.config.TARGET_VARIABLE])
        final_rows = len(df_model)
        
        print(f"  • Data retention: {(final_rows/initial_rows)*100:.1f}%")
        
        # ✅ HAPUS encoding - keep Provinsi for analysis only
        # Province will be used for grouping and validation, not as predictor
        
        # Sort by time
        df_model = df_model.sort_values(['Tahun', 'Provinsi', 'Kabupaten']).reset_index(drop=True)
        
        # ✅ UBAH: Only use meaningful predictors
        X_features = self.config.PREDICTOR_VARIABLES  # Remove + ['Provinsi_encoded']
        X = df_model[X_features].copy()
        y = df_model[self.config.TARGET_VARIABLE].copy()
        
        return df_model, X, y, X_features

class TimeSeriesCV:
    """Custom Time Series Cross-Validator"""
    
    def __init__(self, df_model, years_col='Tahun', min_train_years=3):
        self.df_model = df_model
        self.years_col = years_col
        self.min_train_years = min_train_years
        self.splits = self._create_splits()
    
    def _create_splits(self):
        """Create time series splits"""
        unique_years = sorted(self.df_model[self.years_col].unique())
        splits = []
        
        for i in range(self.min_train_years, len(unique_years)):
            train_years = unique_years[:i]
            test_year = [unique_years[i]]
            
            train_indices = self.df_model[self.df_model[self.years_col].isin(train_years)].index.tolist()
            test_indices = self.df_model[self.df_model[self.years_col].isin(test_year)].index.tolist()
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append({
                    'train_years': train_years,
                    'test_year': test_year[0],
                    'train_indices': train_indices,
                    'test_indices': test_indices
                })
        
        return splits
    
    def split(self, X, y=None, groups=None):
        """Generator for cross-validation splits"""
        for split in self.splits:
            yield split['train_indices'], split['test_indices']
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits"""
        return len(self.splits)

class ModelTrainer:
    """Class for model training and evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.cv_results = None
        self.feature_importance = None
    
    def custom_scoring(self, estimator, X, y):
        """Custom scoring function for time series"""
        y_pred = estimator.predict(X)
        r2 = r2_score(y, y_pred)
        
        if r2 < 0:
            return -1000
        
        # Check for extreme predictions
        pred_range = y_pred.max() - y_pred.min()
        actual_range = y.max() - y.min()
        
        if pred_range > actual_range * 2:
            return r2 * 0.5
        
        return r2
    
    def train_model(self, X, y, ts_cv):
        """Train the model with hyperparameter tuning"""
        print("🎛️ Training model with hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=self.config.RANDOM_STATE, n_jobs=-1),
            self.config.PARAM_GRID,
            cv=ts_cv,
            scoring=self.custom_scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_model = grid_search.best_estimator_
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        return self.best_model
    
    def evaluate_model(self, X, y, ts_splits, df_model):
        """Evaluate model using time series cross-validation"""
        print("📊 Evaluating model with Time Series CV...")
        
        cv_results = []
        
        for i, split in enumerate(ts_splits):
            # Get training and test data
            X_train_cv = X.iloc[split['train_indices']]
            y_train_cv = y.iloc[split['train_indices']]
            X_test_cv = X.iloc[split['test_indices']]
            y_test_cv = y.iloc[split['test_indices']]
            
            # Train model
            self.best_model.fit(X_train_cv, y_train_cv)
            
            # Make predictions
            y_pred_cv = self.best_model.predict(X_test_cv)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_cv, y_pred_cv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_cv, y_pred_cv)
            r2 = r2_score(y_test_cv, y_pred_cv)
            
            cv_results.append({
                'fold': i + 1,
                'test_year': split['test_year'],
                'train_size': len(X_train_cv),
                'test_size': len(X_test_cv),
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
        
        self.cv_results = pd.DataFrame(cv_results)
        return self.cv_results
    
    def calculate_feature_importance(self, X, X_features):
        """Calculate feature importance"""
        print("🎯 Calculating feature importance...")
        
        feature_importance = pd.DataFrame({
            'Feature': X_features,
            'Importance': self.best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.feature_importance = feature_importance
        return feature_importance

class ScenarioAnalyzer:
    """Class for scenario analysis and predictions"""
    
    def __init__(self, config):
        self.config = config
    
    def create_scenarios(self, df_historical, target_year=2025):
        """Create future scenarios"""
        print(f"🔮 Creating scenarios for {target_year}...")
        
        latest_data = df_historical[df_historical['Tahun'] == df_historical['Tahun'].max()].copy()
        
        if len(latest_data) == 0:
            return None
        
        scenarios = []
        
        # Status Quo Scenario
        status_quo = latest_data.copy()
        status_quo['Tahun'] = target_year
        status_quo['Scenario'] = 'Status Quo'
        scenarios.append(status_quo)
        
        # Conservative Improvement (5% improvement)
        conservative = latest_data.copy()
        conservative['Tahun'] = target_year
        conservative['Scenario'] = 'Conservative Improvement'
        
        for col in ['Kemiskinan (%)', 'Tanpa Air Bersih (%)', 'Pengeluaran Pangan (%)']:
            if col in conservative.columns:
                conservative[col] *= 0.95
        
        for col in ['Lama Sekolah Perempuan (tahun)', 'Rasio Tenaga Kesehatan', 'Angka Harapan Hidup (tahun)']:
            if col in conservative.columns:
                conservative[col] *= 1.02
        
        scenarios.append(conservative)
        
        # Moderate Improvement (10% improvement)
        moderate = latest_data.copy()
        moderate['Tahun'] = target_year
        moderate['Scenario'] = 'Moderate Improvement'
        
        for col in ['Kemiskinan (%)', 'Tanpa Air Bersih (%)']:
            if col in moderate.columns:
                moderate[col] *= 0.9
        if 'Pengeluaran Pangan (%)' in moderate.columns:
            moderate['Pengeluaran Pangan (%)'] *= 0.95
        
        for col in ['Lama Sekolah Perempuan (tahun)', 'Rasio Tenaga Kesehatan']:
            if col in moderate.columns:
                moderate[col] *= 1.05
        if 'Angka Harapan Hidup (tahun)' in moderate.columns:
            moderate['Angka Harapan Hidup (tahun)'] *= 1.02
        
        scenarios.append(moderate)
        
        # Optimistic Improvement (15% improvement)
        optimistic = latest_data.copy()
        optimistic['Tahun'] = target_year
        optimistic['Scenario'] = 'Optimistic Improvement'
        
        for col in ['Kemiskinan (%)', 'Tanpa Air Bersih (%)']:
            if col in optimistic.columns:
                optimistic[col] *= 0.85
        if 'Pengeluaran Pangan (%)' in optimistic.columns:
            optimistic['Pengeluaran Pangan (%)'] *= 0.92
        if 'Lama Sekolah Perempuan (tahun)' in optimistic.columns:
            optimistic['Lama Sekolah Perempuan (tahun)'] *= 1.08
        if 'Rasio Tenaga Kesehatan' in optimistic.columns:
            optimistic['Rasio Tenaga Kesehatan'] *= 1.15
        if 'Angka Harapan Hidup (tahun)' in optimistic.columns:
            optimistic['Angka Harapan Hidup (tahun)'] *= 1.03
        
        scenarios.append(optimistic)
        
        return pd.concat(scenarios, ignore_index=True)
    
    def predict_scenarios_with_uncertainty(self, model, scenarios_df, X_features, X, y, n_bootstrap=50):
        """Predict scenarios with uncertainty quantification"""
        print("📊 Generating predictions with uncertainty...")
        
        results = []
        
        for scenario in scenarios_df['Scenario'].unique():
            scenario_data = scenarios_df[scenarios_df['Scenario'] == scenario]
            X_scenario = scenario_data[X_features].copy()
            
            # Bootstrap predictions
            bootstrap_predictions = []
            
            for _ in range(n_bootstrap):
                bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X.iloc[bootstrap_indices]
                y_bootstrap = y.iloc[bootstrap_indices]
                
                bootstrap_model = RandomForestRegressor(**model.get_params())
                bootstrap_model.fit(X_bootstrap, y_bootstrap)
                
                pred_bootstrap = bootstrap_model.predict(X_scenario)
                bootstrap_predictions.append(pred_bootstrap)
            
            # Calculate statistics
            bootstrap_predictions = np.array(bootstrap_predictions)
            mean_pred = np.mean(bootstrap_predictions, axis=0)
            std_pred = np.std(bootstrap_predictions, axis=0)
            lower_ci = np.percentile(bootstrap_predictions, 2.5, axis=0)
            upper_ci = np.percentile(bootstrap_predictions, 97.5, axis=0)
            
            scenario_results = scenario_data.copy()
            scenario_results['Predicted_Komposit'] = mean_pred
            scenario_results['Prediction_Std'] = std_pred
            scenario_results['Lower_CI_95'] = lower_ci
            scenario_results['Upper_CI_95'] = upper_ci
            scenario_results['Uncertainty_Range'] = upper_ci - lower_ci
            
            results.append(scenario_results)
        
        return pd.concat(results, ignore_index=True)

class RiskAssessment:
    """Class for risk assessment and early warning system"""
    
    def __init__(self, config):
        self.config = config
    
    def create_risk_assessment(self, predictions_df, uncertainty_threshold=0.5, low_security_threshold=3.0):
        """Create risk assessment based on predictions"""
        print("⚠️ Creating risk assessment...")
        
        risk_df = predictions_df.copy()
        
        def assign_risk_level(pred, uncertainty, low_threshold, high_uncertainty):
            if pred < low_threshold and uncertainty > high_uncertainty:
                return 'Very High Risk'
            elif pred < low_threshold:
                return 'High Risk'
            elif uncertainty > high_uncertainty:
                return 'Medium Risk (High Uncertainty)'
            elif pred < 4.0:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        risk_df['Risk_Level'] = risk_df.apply(
            lambda row: assign_risk_level(
                row['Predicted_Komposit'],
                row['Uncertainty_Range'],
                low_security_threshold,
                uncertainty_threshold
            ),
            axis=1
        )
        
        return risk_df
    
    def generate_early_warnings(self, risk_df, scenario='Status Quo'):
        """Generate early warning alerts"""
        scenario_data = risk_df[risk_df['Scenario'] == scenario]
        warnings = []
        
        # High risk provinces
        high_risk = scenario_data[scenario_data['Risk_Level'].isin(['Very High Risk', 'High Risk'])]
        if len(high_risk) > 0:
            warnings.append({
                'Type': 'High Risk Alert',
                'Count': len(high_risk),
                'Provinces': high_risk['Provinsi'].tolist(),
                'Message': f'{len(high_risk)} provinces at high risk of food insecurity'
            })
        
        # High uncertainty areas
        high_uncertainty = scenario_data[scenario_data['Uncertainty_Range'] > 0.5]
        if len(high_uncertainty) > 0:
            warnings.append({
                'Type': 'High Uncertainty Alert',
                'Count': len(high_uncertainty),
                'Provinces': high_uncertainty['Provinsi'].tolist(),
                'Message': f'{len(high_uncertainty)} provinces with high prediction uncertainty'
            })
        
        return warnings

class FoodSecurityForecaster:
    """Main class that orchestrates the entire forecasting process"""
    
    def __init__(self, config=None):
        self.config = config or FoodSecurityConfig()
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.scenario_analyzer = ScenarioAnalyzer(self.config)
        self.risk_assessor = RiskAssessment(self.config)
        
        # Data attributes
        self.df_model = None
        self.X = None
        self.y = None
        self.X_features = None
        self.ts_cv = None
        
        # Results attributes
        self.cv_results = None
        self.feature_importance = None
        self.scenario_predictions = None
        self.risk_assessment = None
    
    def run_full_analysis(self, df):
        """Run the complete food security forecasting analysis"""
        print("🚀 Starting Food Security Forecasting Analysis...")
        print("=" * 60)
        
        # 1. Data Processing
        validated_df = self.data_processor.load_and_validate_data(df)
        self.df_model, self.X, self.y, self.X_features = self.data_processor.preprocess_data(validated_df)
        
        # 2. Time Series Cross-Validation Setup
        self.ts_cv = TimeSeriesCV(self.df_model)
        print(f"🕐 Created {self.ts_cv.get_n_splits()} time series CV splits")
        
        # 3. Model Training
        best_model = self.model_trainer.train_model(self.X, self.y, self.ts_cv)
        
        # 4. Model Evaluation
        self.cv_results = self.model_trainer.evaluate_model(self.X, self.y, self.ts_cv.splits, self.df_model)
        
        # 5. Feature Importance
        self.feature_importance = self.model_trainer.calculate_feature_importance(self.X, self.X_features)
        
        # 6. Scenario Analysis
        scenarios_df = self.scenario_analyzer.create_scenarios(self.df_model, target_year=2025)
        if scenarios_df is not None:
            self.scenario_predictions = self.scenario_analyzer.predict_scenarios_with_uncertainty(
                best_model, scenarios_df, self.X_features, self.X, self.y
            )
            
            # 7. Risk Assessment
            self.risk_assessment = self.risk_assessor.create_risk_assessment(self.scenario_predictions)
        
        print("✅ Analysis completed successfully!")
        return self
    
    def get_summary_report(self):
        """Generate summary report"""
        if self.cv_results is None:
            return "Analysis not completed yet. Please run run_full_analysis() first."
        
        report = {
            'model_performance': {
                'mean_r2': self.cv_results['r2'].mean(),
                'r2_stability': self.cv_results['r2'].std(),
                'mean_rmse': self.cv_results['rmse'].mean(),
            },
            'top_features': self.feature_importance.head(3)['Feature'].tolist() if self.feature_importance is not None else [],
            'data_info': {
                'total_records': len(self.df_model),
                'provinces': self.df_model['Provinsi'].nunique(),
                'years_range': f"{self.df_model['Tahun'].min()}-{self.df_model['Tahun'].max()}"
            }
        }
        
        if self.risk_assessment is not None:
            status_quo_risk = self.risk_assessment[self.risk_assessment['Scenario'] == 'Status Quo']
            report['risk_summary'] = {
                'high_risk_provinces': len(status_quo_risk[status_quo_risk['Risk_Level'].isin(['Very High Risk', 'High Risk'])]),
                'total_provinces': len(status_quo_risk)
            }
        
        return report