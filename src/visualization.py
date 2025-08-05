# =============================================================================
# VISUALIZATION MODULE FOR FOOD SECURITY FORECASTING
# Comprehensive visualization functions for dashboard and analysis
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class FoodSecurityVisualizer:
    """Class for creating visualizations for food security analysis"""
    
    def __init__(self, config):
        self.config = config
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#17becf'
        }
    
    def plot_data_overview(self, df):
        """Create data overview visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Distribution by Year', 'Data Distribution by Province', 
                          'Target Variable Distribution', 'Missing Data Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Data by year
        year_counts = df['Tahun'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=year_counts.index, y=year_counts.values, name='Records by Year',
                   marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # 2. Data by province (top 10)
        province_counts = df['Provinsi'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=province_counts.values, y=province_counts.index, 
                   orientation='h', name='Top 10 Provinces',
                   marker_color=self.colors['secondary']),
            row=1, col=2
        )
        
        # 3. Target distribution
        if self.config.TARGET_VARIABLE in df.columns:
            target_dist = df[self.config.TARGET_VARIABLE].value_counts().sort_index()
            labels = [f"{int(val)} ({self.config.KOMPOSIT_MAPPING.get(int(val), 'Unknown')})" 
                     for val in target_dist.index]
            fig.add_trace(
                go.Bar(x=target_dist.index, y=target_dist.values, name='Target Distribution',
                       marker_color=self.colors['success']),
                row=2, col=1
            )
        
        # 4. Missing data
        missing_data = df[self.config.PREDICTOR_VARIABLES].isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        fig.add_trace(
            go.Bar(x=missing_pct.values, y=missing_pct.index, 
                   orientation='h', name='Missing Data %',
                   marker_color=self.colors['warning']),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Data Overview Dashboard",
            showlegend=False
        )
        
        return fig
    
    def plot_model_performance(self, cv_results):
        """Create model performance visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score Over Time', 'RMSE Over Time', 
                          'Performance Metrics Distribution', 'Model Stability'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. R² over time
        fig.add_trace(
            go.Scatter(x=cv_results['test_year'], y=cv_results['r2'],
                      mode='lines+markers', name='R² Score',
                      line=dict(color=self.colors['primary'], width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # 2. RMSE over time
        fig.add_trace(
            go.Scatter(x=cv_results['test_year'], y=cv_results['rmse'],
                      mode='lines+markers', name='RMSE',
                      line=dict(color=self.colors['warning'], width=3),
                      marker=dict(size=8)),
            row=1, col=2
        )
        
        # 3. Metrics distribution
        metrics = ['r2', 'rmse', 'mae']
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Box(y=cv_results[metric], name=metric.upper(),
                      marker_color=list(self.colors.values())[i]),
                row=2, col=1
            )
        
        # 4. Stability metrics
        stability_data = {
            'Mean R²': cv_results['r2'].mean(),
            'Std R²': cv_results['r2'].std(),
            'CV R²': cv_results['r2'].std() / cv_results['r2'].mean()
        }
        
        fig.add_trace(
            go.Bar(x=list(stability_data.keys()), y=list(stability_data.values()),
                   name='Stability Metrics', marker_color=self.colors['info']),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Model Performance Analysis",
            showlegend=False
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance):
        """Create feature importance visualization"""
        # Sort by importance
        sorted_features = feature_importance.sort_values('Importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=sorted_features['Importance'],
            y=sorted_features['Feature'],
            orientation='h',
            marker=dict(
                color=sorted_features['Importance'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title="Feature Importance Analysis",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=500,
            margin=dict(l=200)
        )
        
        return fig
    
    def plot_scenario_comparison(self, scenario_predictions):
        """Create scenario comparison visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Scenario Predictions Distribution', 'Uncertainty Analysis',
                          'Top Provinces by Scenario', 'Risk Level Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Scenario predictions
        scenarios = scenario_predictions['Scenario'].unique()
        colors = px.colors.qualitative.Set3[:len(scenarios)]
        
        for i, scenario in enumerate(scenarios):
            scenario_data = scenario_predictions[scenario_predictions['Scenario'] == scenario]
            fig.add_trace(
                go.Box(y=scenario_data['Predicted_Komposit'], name=scenario,
                      marker_color=colors[i]),
                row=1, col=1
            )
        
        # 2. Uncertainty analysis
        uncertainty_by_scenario = scenario_predictions.groupby('Scenario')['Uncertainty_Range'].mean()
        fig.add_trace(
            go.Bar(x=uncertainty_by_scenario.index, y=uncertainty_by_scenario.values,
                   name='Average Uncertainty', marker_color=self.colors['warning']),
            row=1, col=2
        )
        
        # 3. Top provinces (Moderate Improvement scenario)
        moderate_data = scenario_predictions[scenario_predictions['Scenario'] == 'Moderate Improvement']
        if len(moderate_data) > 0:
            top_provinces = moderate_data.nlargest(10, 'Predicted_Komposit')
            fig.add_trace(
                go.Bar(x=top_provinces['Predicted_Komposit'], 
                       y=top_provinces['Provinsi'],
                       orientation='h', name='Top 10 Provinces',
                       error_x=dict(array=top_provinces['Uncertainty_Range']/2),
                       marker_color=self.colors['success']),
                row=2, col=1
            )
        
        # 4. Risk level distribution
        if 'Risk_Level' in scenario_predictions.columns:
            status_quo_data = scenario_predictions[scenario_predictions['Scenario'] == 'Status Quo']
            risk_dist = status_quo_data['Risk_Level'].value_counts()
            fig.add_trace(
                go.Pie(labels=risk_dist.index, values=risk_dist.values,
                       name='Risk Distribution'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Scenario Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def plot_geographic_heatmap(self, df, value_column, title="Geographic Distribution"):
        """Create geographic heatmap"""
        # Aggregate data by province
        province_data = df.groupby('Provinsi')[value_column].mean().reset_index()
        
        fig = px.choropleth(
            province_data,
            locations='Provinsi',
            color=value_column,
            hover_name='Provinsi',
            color_continuous_scale='RdYlGn',
            title=title
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            )
        )
        
        return fig
    
    def plot_correlation_matrix(self, df):
        """Create correlation matrix heatmap"""
        # Calculate correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def plot_time_series_trends(self, df):
        """Create time series trend analysis"""
        # Calculate yearly averages for key indicators
        yearly_trends = df.groupby('Tahun')[self.config.PREDICTOR_VARIABLES + [self.config.TARGET_VARIABLE]].mean()
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=tuple(self.config.PREDICTOR_VARIABLES),
            vertical_spacing=0.08
        )
        
        for i, var in enumerate(self.config.PREDICTOR_VARIABLES):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_trends.index,
                    y=yearly_trends[var],
                    mode='lines+markers',
                    name=var,
                    line=dict(width=3),
                    marker=dict(size=6)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=900,
            title_text="Time Series Trends of Key Indicators",
            showlegend=False
        )
        
        return fig
    
    def plot_prediction_vs_actual(self, y_true, y_pred, title="Predictions vs Actual"):
        """Create prediction vs actual scatter plot"""
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=y_true,
                colorscale='Viridis',
                size=8,
                opacity=0.7
            )
        ))
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=500
        )
        
        return fig
    
    def plot_residuals_analysis(self, y_true, y_pred):
        """Create residuals analysis plots"""
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Residuals vs Predicted', 'Residuals Distribution')
        )
        
        # Residuals vs predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=6, opacity=0.7)
            ),
            row=1, col=1
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residuals histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residuals Distribution',
                nbinsx=20,
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Residuals Analysis",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_summary_metrics_cards(self, cv_results, feature_importance, scenario_predictions=None):
        """Create summary metrics for dashboard cards"""
        metrics = {
            'Model Performance': {
                'Mean R²': f"{cv_results['r2'].mean():.3f}",
                'RMSE': f"{cv_results['rmse'].mean():.3f}",
                'Model Stability': f"{cv_results['r2'].std():.3f}",
                'CV Folds': len(cv_results)
            },
            'Feature Analysis': {
                'Top Feature': feature_importance.iloc[0]['Feature'],
                'Top Importance': f"{feature_importance.iloc[0]['Importance']:.3f}",
                'Features Used': len(feature_importance),
                'Importance Range': f"{feature_importance['Importance'].min():.3f} - {feature_importance['Importance'].max():.3f}"
            }
        }
        
        if scenario_predictions is not None:
            status_quo = scenario_predictions[scenario_predictions['Scenario'] == 'Status Quo']
            metrics['Scenario Analysis'] = {
                'Provinces Analyzed': len(status_quo),
                'Avg Prediction': f"{status_quo['Predicted_Komposit'].mean():.2f}",
                'Scenarios Created': scenario_predictions['Scenario'].nunique(),
                'Avg Uncertainty': f"{status_quo['Uncertainty_Range'].mean():.3f}"
            }
        
        return metrics
    
    def plot_interactive_map(self, df, color_column, title="Interactive Map"):
        """Create interactive map visualization"""
        # This is a placeholder for interactive mapping
        # In a real implementation, you would use actual geographic data
        province_data = df.groupby('Provinsi')[color_column].mean().reset_index()
        
        fig = px.bar(
            province_data.sort_values(color_column, ascending=False).head(15),
            x=color_column,
            y='Provinsi',
            orientation='h',
            title=f"Top 15 Provinces - {title}",
            color=color_column,
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        
        return fig