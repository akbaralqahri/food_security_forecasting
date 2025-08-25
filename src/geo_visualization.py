"""
Geographic Visualization Module for Food Security Dashboard
==========================================================

This module contains functions for creating interactive geographic visualizations
including maps, regional analysis, and spatial insights for food security data.

Author: Food Security Team
Version: 1.0
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import logging

# Optional imports with error handling
try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logging.warning("Folium not available. Some map features will be limited.")

# Configure logging
logger = logging.getLogger(__name__)

# Indonesia provinces coordinates for mapping
INDONESIA_PROVINCES_COORDS = {
    'Aceh': {'lat': 4.695135, 'lon': 96.749397, 'code': 'ID-AC'},
    'Sumatera Utara': {'lat': 2.1153547, 'lon': 99.5450974, 'code': 'ID-SU'},
    'Sumatera Barat': {'lat': -0.7399397, 'lon': 100.8000051, 'code': 'ID-SB'},
    'Riau': {'lat': 0.2933469, 'lon': 101.7068294, 'code': 'ID-RI'},
    'Kepulauan Riau': {'lat': 3.9456514, 'lon': 108.1428669, 'code': 'ID-KR'},
    'Jambi': {'lat': -1.4851831, 'lon': 102.4380581, 'code': 'ID-JA'},
    'Sumatera Selatan': {'lat': -3.3194374, 'lon': 103.914399, 'code': 'ID-SS'},
    'Bangka Belitung': {'lat': -2.7410513, 'lon': 106.4405872, 'code': 'ID-BB'},
    'Bengkulu': {'lat': -3.5778471, 'lon': 102.3463875, 'code': 'ID-BE'},
    'Lampung': {'lat': -4.5585849, 'lon': 105.4068079, 'code': 'ID-LA'},
    'DKI Jakarta': {'lat': -6.211544, 'lon': 106.845172, 'code': 'ID-JK'},
    'Jawa Barat': {'lat': -6.914744, 'lon': 107.609810, 'code': 'ID-JB'},
    'Jawa Tengah': {'lat': -7.150975, 'lon': 110.140499, 'code': 'ID-JT'},
    'DI Yogyakarta': {'lat': -7.795580, 'lon': 110.369492, 'code': 'ID-YO'},
    'Jawa Timur': {'lat': -7.250445, 'lon': 112.768845, 'code': 'ID-JI'},
    'Banten': {'lat': -6.4058172, 'lon': 106.0640179, 'code': 'ID-BT'},
    'Bali': {'lat': -8.4095178, 'lon': 115.188916, 'code': 'ID-BA'},
    'Nusa Tenggara Barat': {'lat': -8.6529334, 'lon': 117.3616476, 'code': 'ID-NB'},
    'NTB': {'lat': -8.6529334, 'lon': 117.3616476, 'code': 'ID-NB'},
    'Nusa Tenggara Timur': {'lat': -8.6573819, 'lon': 121.0793705, 'code': 'ID-NT'},
    'NTT': {'lat': -8.6573819, 'lon': 121.0793705, 'code': 'ID-NT'},
    'Kalimantan Barat': {'lat': -0.2787808, 'lon': 111.4752851, 'code': 'ID-KB'},
    'Kalimantan Tengah': {'lat': -1.6814878, 'lon': 113.3823545, 'code': 'ID-KT'},
    'Kalimantan Selatan': {'lat': -3.0926415, 'lon': 115.2837585, 'code': 'ID-KS'},
    'Kalimantan Timur': {'lat': 1.6406296, 'lon': 116.419389, 'code': 'ID-KI'},
    'Kalimantan Utara': {'lat': 3.0730929, 'lon': 116.0413889, 'code': 'ID-KU'},
    'Sulawesi Utara': {'lat': 1.2379274, 'lon': 124.8413490, 'code': 'ID-SA'},
    'Sulawesi Tengah': {'lat': -1.4300254, 'lon': 121.4456179, 'code': 'ID-ST'},
    'Sulawesi Selatan': {'lat': -3.6687994, 'lon': 119.9740534, 'code': 'ID-SN'},
    'Sulawesi Tenggara': {'lat': -4.14491, 'lon': 122.174605, 'code': 'ID-SG'},
    'Gorontalo': {'lat': 0.6999372, 'lon': 122.4467238, 'code': 'ID-GO'},
    'Sulawesi Barat': {'lat': -2.8441371, 'lon': 119.2320784, 'code': 'ID-SR'},
    'Maluku': {'lat': -3.2384616, 'lon': 130.1452734, 'code': 'ID-MA'},
    'Maluku Utara': {'lat': 1.5709993, 'lon': 127.8087693, 'code': 'ID-MU'},
    'Papua': {'lat': -4.269928, 'lon': 138.080353, 'code': 'ID-PA'},
    'Papua Barat': {'lat': -1.3361154, 'lon': 133.1747162, 'code': 'ID-PB'},
    'Papua Selatan': {'lat': -7.013056, 'lon': 140.516667, 'code': 'ID-PS'},
    'Papua Tengah': {'lat': -3.9709, 'lon': 136.1127, 'code': 'ID-PT'},
    'Papua Pegunungan': {'lat': -4.0823, 'lon': 138.9568, 'code': 'ID-PP'},
    'Papua Barat Daya': {'lat': -1.7, 'lon': 132.2, 'code': 'ID-PD'}
}

class GeoVisualizationError(Exception):
    """Custom exception for geo visualization errors"""
    pass

def validate_geo_data(scenario_data, risk_data):
    """
    Validate geographic data before visualization
    
    Args:
        scenario_data (pd.DataFrame): Scenario predictions data
        risk_data (pd.DataFrame): Risk assessment data
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    if scenario_data is None or scenario_data.empty:
        errors.append("Scenario data is empty or None")
    
    if risk_data is None or risk_data.empty:
        errors.append("Risk data is empty or None")
    
    if not errors:
        # Check required columns
        required_scenario_cols = ['Scenario', 'Provinsi', 'Predicted_Komposit', 'Uncertainty_Range']
        required_risk_cols = ['Scenario', 'Provinsi', 'Risk_Level']
        
        missing_scenario = [col for col in required_scenario_cols if col not in scenario_data.columns]
        missing_risk = [col for col in required_risk_cols if col not in risk_data.columns]
        
        if missing_scenario:
            errors.append(f"Missing scenario columns: {missing_scenario}")
        if missing_risk:
            errors.append(f"Missing risk columns: {missing_risk}")
    
    return len(errors) == 0, errors

def create_indonesia_folium_map(scenario_data, risk_data, selected_scenario='Status Quo'):
    """
    Create an interactive Folium map for Indonesia with food security data
    
    Args:
        scenario_data (pd.DataFrame): Scenario predictions data
        risk_data (pd.DataFrame): Risk assessment data
        selected_scenario (str): Selected scenario to visualize
        
    Returns:
        folium.Map: Interactive Folium map object
        
    Raises:
        GeoVisualizationError: If folium is not available or data is invalid
    """
    if not FOLIUM_AVAILABLE:
        raise GeoVisualizationError("Folium library is not available. Please install folium.")
    
    # Validate data
    is_valid, errors = validate_geo_data(scenario_data, risk_data)
    if not is_valid:
        raise GeoVisualizationError(f"Data validation failed: {'; '.join(errors)}")
    
    try:
        # Filter data for selected scenario
        scenario_filtered = scenario_data[scenario_data['Scenario'] == selected_scenario]
        risk_filtered = risk_data[risk_data['Scenario'] == selected_scenario]
        
        # Merge the data
        map_data = pd.merge(scenario_filtered, risk_filtered, on=['Provinsi', 'Scenario'], how='left')
        
        # Create base map centered on Indonesia
        indonesia_map = folium.Map(
            location=[-2.5, 118.0],  # Center of Indonesia
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add alternative tile layers
        folium.TileLayer('Stamen Terrain').add_to(indonesia_map)
        folium.TileLayer('cartodb positron').add_to(indonesia_map)
        
        # Create color mapping for risk levels
        risk_colors = {
            'Very High Risk': '#dc3545',  # Red
            'High Risk': '#fd7e14',       # Orange
            'Medium Risk': '#ffc107',     # Yellow  
            'Low Risk': '#28a745'         # Green
        }
        
        # Add markers for each province
        for _, row in map_data.iterrows():
            province = row['Provinsi']
            
            # Get coordinates
            if province in INDONESIA_PROVINCES_COORDS:
                coords = INDONESIA_PROVINCES_COORDS[province]
                
                # Determine marker color based on risk level
                risk_level = row['Risk_Level']
                color = risk_colors.get(risk_level, '#6c757d')
                
                # Create marker size based on predicted score (inverse - lower score = bigger marker)
                marker_size = max(5, min(25, (6 - row['Predicted_Komposit']) * 5))
                
                # Create popup content
                popup_content = f"""
                <div style="width: 250px;">
                    <h4 style="margin: 0 0 10px 0; color: #333;">{province}</h4>
                    <hr style="margin: 10px 0;">
                    <p><strong>Predicted Score:</strong> {row['Predicted_Komposit']:.2f}/6</p>
                    <p><strong>Risk Level:</strong> <span style="color: {color}; font-weight: bold;">{risk_level}</span></p>
                    <p><strong>Confidence Interval:</strong> {row.get('Lower_CI_95', 'N/A'):.2f} - {row.get('Upper_CI_95', 'N/A'):.2f}</p>
                    <p><strong>Uncertainty:</strong> ±{row['Uncertainty_Range']:.3f}</p>
                    <p><strong>Scenario:</strong> {selected_scenario}</p>
                </div>
                """
                
                # Add marker
                folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=marker_size,
                    popup=folium.Popup(popup_content, max_width=300),
                    color='white',
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2,
                    tooltip=f"{province}: {row['Predicted_Komposit']:.2f} ({risk_level})"
                ).add_to(indonesia_map)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: 180px; 
                    background-color: white; border: 2px solid grey; z-index:9999; 
                    font-size: 14px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h4 style="margin: 0 0 10px 0;">Food Security Risk</h4>
        <p style="margin: 2px 0;"><i class="fa fa-circle" style="color: {risk_colors['Very High Risk']}"></i> Very High Risk</p>
        <p style="margin: 2px 0;"><i class="fa fa-circle" style="color: {risk_colors['High Risk']}"></i> High Risk</p>
        <p style="margin: 2px 0;"><i class="fa fa-circle" style="color: {risk_colors['Medium Risk']}"></i> Medium Risk</p>
        <p style="margin: 2px 0;"><i class="fa fa-circle" style="color: {risk_colors['Low Risk']}"></i> Low Risk</p>
        <hr>
        <p style="font-size: 12px; margin: 5px 0 0 0;">Scenario: {selected_scenario}</p>
        <p style="font-size: 12px; margin: 0;">Marker size ∝ Risk level</p>
        </div>
        '''
        indonesia_map.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(indonesia_map)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(indonesia_map)
        
        # Add measure tool
        plugins.MeasureControl().add_to(indonesia_map)
        
        return indonesia_map
        
    except Exception as e:
        logger.error(f"Error creating Folium map: {str(e)}")
        raise GeoVisualizationError(f"Failed to create Folium map: {str(e)}")

def create_choropleth_map_plotly(scenario_data, risk_data, selected_scenario='Status Quo'):
    """
    Create a choropleth map using Plotly for better integration with Streamlit
    REVERTED: All provinces colored normally (no gray areas)
    """
    # Validate data
    is_valid, errors = validate_geo_data(scenario_data, risk_data)
    if not is_valid:
        raise GeoVisualizationError(f"Data validation failed: {'; '.join(errors)}")
    
    try:
        # Filter data for selected scenario
        scenario_filtered = scenario_data[scenario_data['Scenario'] == selected_scenario].copy()
        risk_filtered = risk_data[risk_data['Scenario'] == selected_scenario].copy()
        
        # Clean merge with specific columns to avoid duplicates
        risk_cols = ['Provinsi', 'Scenario', 'Risk_Level']
        if 'Uncertainty_Range' not in scenario_filtered.columns and 'Uncertainty_Range' in risk_filtered.columns:
            risk_cols.append('Uncertainty_Range')
            
        risk_filtered_clean = risk_filtered[risk_cols]
        
        # Merge without duplicate columns
        map_data = pd.merge(scenario_filtered, risk_filtered_clean, on=['Provinsi', 'Scenario'], how='left')
        
        # Add coordinates
        map_data['lat'] = map_data['Provinsi'].map(lambda x: INDONESIA_PROVINCES_COORDS.get(x, {}).get('lat', 0))
        map_data['lon'] = map_data['Provinsi'].map(lambda x: INDONESIA_PROVINCES_COORDS.get(x, {}).get('lon', 0))
        
        # Filter out provinces without coordinates
        map_data = map_data[(map_data['lat'] != 0) | (map_data['lon'] != 0)]
        
        if map_data.empty:
            raise GeoVisualizationError("No provinces found with valid coordinates")
        
        # Handle uncertainty and other columns
        uncertainty_col = None
        size_col = None
        
        if 'Uncertainty_Range' in map_data.columns:
            uncertainty_col = 'Uncertainty_Range'
            size_col = 'Uncertainty_Range'
        elif 'Uncertainty_Range_x' in map_data.columns:
            uncertainty_col = 'Uncertainty_Range_x'
            size_col = 'Uncertainty_Range_x'
        elif 'Uncertainty_Range_y' in map_data.columns:
            uncertainty_col = 'Uncertainty_Range_y' 
            size_col = 'Uncertainty_Range_y'
        else:
            map_data['Uncertainty_Range'] = 0.1
            uncertainty_col = 'Uncertainty_Range'
            size_col = 'Uncertainty_Range'
        
        # Handle other potentially duplicated columns
        predicted_col = 'Predicted_Komposit'
        lower_ci_col = 'Lower_CI_95'
        upper_ci_col = 'Upper_CI_95'
        
        for col_base in ['Predicted_Komposit', 'Lower_CI_95', 'Upper_CI_95']:
            if col_base in map_data.columns:
                continue
            elif f'{col_base}_x' in map_data.columns:
                if col_base == 'Predicted_Komposit':
                    predicted_col = f'{col_base}_x'
                elif col_base == 'Lower_CI_95':
                    lower_ci_col = f'{col_base}_x'  
                elif col_base == 'Upper_CI_95':
                    upper_ci_col = f'{col_base}_x'
        
        # ✅ BACK TO NORMAL: Create scatter mapbox with normal coloring
        fig = px.scatter_mapbox(
            map_data,
            lat='lat',
            lon='lon',
            color=predicted_col,                 # ✅ Normal coloring for all provinces
            size=size_col,                       
            hover_data={
                'Provinsi': True,
                'Risk_Level': True,
                lower_ci_col: ':.2f' if lower_ci_col in map_data.columns else False,
                upper_ci_col: ':.2f' if upper_ci_col in map_data.columns else False,
                'lat': False,
                'lon': False,
                uncertainty_col: ':.3f'
            },
            color_continuous_scale='RdYlGn',     # ✅ Normal color scale
            range_color=[1, 6],
            size_max=20,
            zoom=4,
            center=dict(lat=-2.5, lon=118.0),
            title=f'Food Security Risk Map - {selected_scenario}',
            height=600
        )
        
        # Update layout
        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(r=0, t=50, l=0, b=0),
            coloraxis_colorbar=dict(
                title="Food Security Score",
                titleside="right"
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Plotly choropleth map: {str(e)}")
        raise GeoVisualizationError(f"Failed to create Plotly map: {str(e)}")

def create_risk_distribution_map(scenario_data, risk_data):
    """
    Create a map showing risk distribution across scenarios with animation
    REVERTED: Normal coloring for all provinces
    """
    # Validate data
    is_valid, errors = validate_geo_data(scenario_data, risk_data)
    if not is_valid:
        raise GeoVisualizationError(f"Data validation failed: {'; '.join(errors)}")
    
    try:
        # [Keep all existing aggregation and data processing code]
        # ... existing data preparation code ...
        
        # ✅ REVERTED: Create animated scatter mapbox with normal coloring
        fig = px.scatter_mapbox(
            combined_data,
            lat='lat',
            lon='lon',
            color=predicted_col,                 # ✅ Normal coloring for all provinces
            size=size_col,                      
            animation_frame='Scenario',
            hover_name='Provinsi',
            hover_data={
                'Provinsi': True,
                'Risk_Level': True,
                lower_ci_col: ':.2f' if lower_ci_col in combined_data.columns else False,
                upper_ci_col: ':.2f' if upper_ci_col in combined_data.columns else False,
                'lat': False,
                'lon': False,
                uncertainty_col: ':.3f'
            },
            color_continuous_scale='RdYlGn',     # ✅ Normal color scale
            range_color=[1, 6],
            size_max=20,
            zoom=4,
            center=dict(lat=-2.5, lon=118.0),
            title='Food Security Risk Across Scenarios',
            height=600
        )
        
        # Update layout
        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(r=0, t=50, l=0, b=0),
            coloraxis_colorbar=dict(
                title="Food Security Score",
                titleside="right"
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating animated risk distribution map: {str(e)}")
        raise GeoVisualizationError(f"Failed to create animated map: {str(e)}")
    
def clean_merge_columns(df1, df2, merge_on):
    """
    Clean merge to avoid duplicate column issues
    
    Args:
        df1: First dataframe
        df2: Second dataframe  
        merge_on: Columns to merge on
        
    Returns:
        Merged dataframe without duplicate columns
    """
    # Find overlapping columns (excluding merge keys)
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    merge_keys = set(merge_on)
    
    overlapping = (df1_cols & df2_cols) - merge_keys
    
    if overlapping:
        logger.info(f"Found overlapping columns: {overlapping}")
        # Keep only essential columns from df2
        essential_cols = list(merge_keys) + ['Risk_Level']
        if 'Uncertainty_Range' in df2.columns and 'Uncertainty_Range' not in df1.columns:
            essential_cols.append('Uncertainty_Range')
        
        # Select only non-overlapping columns from df2
        df2_filtered = df2[essential_cols]
        return pd.merge(df1, df2_filtered, on=merge_on, how='left')
    else:
        return pd.merge(df1, df2, on=merge_on, how='left')

def generate_geographic_insights(scenario_predictions, risk_assessment, selected_scenario):
    """
    Generate geographic insights from the forecasting data
    """
    insights = []
    
    try:
        # Filter data for selected scenario
        scenario_data = scenario_predictions[scenario_predictions['Scenario'] == selected_scenario].copy()
        risk_data = risk_assessment[risk_assessment['Scenario'] == selected_scenario].copy()
        
        if scenario_data.empty or risk_data.empty:
            return insights
        
        # ✅ FIX: Aggregate to province level to remove duplicates
        scenario_agg = scenario_data.groupby('Provinsi').agg({
            'Predicted_Komposit': 'mean',
            'Uncertainty_Range': 'mean' if 'Uncertainty_Range' in scenario_data.columns else lambda x: 0
        }).reset_index()
        
        risk_agg = risk_data.groupby('Provinsi').agg({
            'Risk_Level': 'first'  # Take first risk level (should be same for all districts in province)
        }).reset_index()
        
        # Get unique high risk provinces
        high_risk_provinces = risk_agg[
            risk_agg['Risk_Level'].isin(['Very High Risk', 'High Risk'])
        ]['Provinsi'].unique().tolist()  # ✅ FIX: Use unique()
        
        if high_risk_provinces:
            # ✅ FIX: Regional clustering insight with unique provinces
            java_provinces = [p for p in high_risk_provinces if any(x in p for x in ['Jawa', 'Jakarta', 'Banten', 'Yogyakarta'])]
            sumatra_provinces = [p for p in high_risk_provinces if any(x in p for x in ['Sumatera', 'Aceh', 'Lampung', 'Bengkulu', 'Jambi', 'Riau', 'Bangka'])]
            eastern_provinces = [p for p in high_risk_provinces if any(x in p for x in ['Papua', 'Maluku', 'Nusa Tenggara'])]
            sulawesi_provinces = [p for p in high_risk_provinces if any(x in p for x in ['Sulawesi', 'Gorontalo'])]
            kalimantan_provinces = [p for p in high_risk_provinces if 'Kalimantan' in p]
            
            # Find the region with most high-risk provinces
            region_counts = {
                'Java': len(java_provinces),
                'Sumatra': len(sumatra_provinces), 
                'Eastern Indonesia': len(eastern_provinces),
                'Sulawesi': len(sulawesi_provinces),
                'Kalimantan': len(kalimantan_provinces)
            }
            
            max_region = max(region_counts, key=region_counts.get)
            max_count = region_counts[max_region]
            
            if max_count > 0:
                if max_region == 'Java' and java_provinces:
                    insights.append({
                        'title': '🏙️ Java Region Alert',
                        'content': f'Java region shows elevated risk with {len(java_provinces)} provinces affected: {", ".join(java_provinces)}. This requires immediate attention given the region\'s population density.',
                        'type': 'warning',
                        'icon': '⚠️'
                    })
                elif max_region == 'Eastern Indonesia' and eastern_provinces:
                    insights.append({
                        'title': '🌊 Eastern Indonesia Risk Pattern',
                        'content': f'Eastern provinces showing high risk: {", ".join(eastern_provinces)}. Geographic isolation may complicate intervention efforts.',
                        'type': 'danger', 
                        'icon': '🚨'
                    })
                elif max_region == 'Sumatra' and sumatra_provinces:
                    insights.append({
                        'title': '🌴 Sumatra Region Concern',
                        'content': f'Sumatra region shows risk concentration: {", ".join(sumatra_provinces)}. Focus on inter-provincial coordination needed.',
                        'type': 'warning',
                        'icon': '⚠️'
                    })
                elif max_region == 'Sulawesi' and sulawesi_provinces:
                    insights.append({
                        'title': '🏝️ Sulawesi Region Risk',
                        'content': f'Sulawesi region shows elevated risk: {", ".join(sulawesi_provinces)}. Strategic intervention needed for this key agricultural region.',
                        'type': 'warning',
                        'icon': '⚠️'
                    })
                elif max_region == 'Kalimantan' and kalimantan_provinces:
                    insights.append({
                        'title': '🌲 Kalimantan Region Alert',
                        'content': f'Kalimantan provinces at risk: {", ".join(kalimantan_provinces)}. Resource extraction regions require food security attention.',
                        'type': 'warning',
                        'icon': '⚠️'
                    })
        
        # ✅ FIX: Score distribution insight with province-level data
        if not scenario_agg.empty:
            avg_score = scenario_agg['Predicted_Komposit'].mean()
            score_std = scenario_agg['Predicted_Komposit'].std()
            
            if score_std > 1.0:
                insights.append({
                    'title': '📊 High Geographic Variation',
                    'content': f'Significant variation in food security scores across provinces (std: {score_std:.2f}). Targeted interventions needed rather than uniform national policies.',
                    'type': 'info',
                    'icon': '📈'
                })
            else:
                insights.append({
                    'title': '📊 Consistent National Pattern',
                    'content': f'Relatively consistent food security levels across provinces (std: {score_std:.2f}). National-level policies may be effective.',
                    'type': 'success',
                    'icon': '✅'
                })
        
        # ✅ FIX: Uncertainty pattern insight with unique provinces
        if 'Uncertainty_Range' in scenario_agg.columns:
            high_uncertainty_provinces = scenario_agg[
                scenario_agg['Uncertainty_Range'] > scenario_agg['Uncertainty_Range'].mean()
            ]['Provinsi'].unique().tolist()  # ✅ FIX: Use unique()
            
            uncertainty_rate = len(high_uncertainty_provinces) / len(scenario_agg) if len(scenario_agg) > 0 else 0
            
            if uncertainty_rate > 0.3:  # More than 30%
                # ✅ FIX: Limit to first 5 provinces and use unique list
                provinces_display = high_uncertainty_provinces[:5]
                more_text = f" and {len(high_uncertainty_provinces) - 5} others" if len(high_uncertainty_provinces) > 5 else ""
                
                insights.append({
                    'title': '🎲 High Prediction Uncertainty',
                    'content': f'Over 30% of provinces show high prediction uncertainty. Enhanced data collection needed in: {", ".join(provinces_display)}{more_text}.',
                    'type': 'warning',
                    'icon': '📊'
                })
        
        # ✅ FIX: Best/worst performance insight with unique provinces
        if not scenario_agg.empty:
            best_province = scenario_agg.loc[scenario_agg['Predicted_Komposit'].idxmax(), 'Provinsi']
            worst_province = scenario_agg.loc[scenario_agg['Predicted_Komposit'].idxmin(), 'Provinsi']
            best_score = scenario_agg['Predicted_Komposit'].max()
            worst_score = scenario_agg['Predicted_Komposit'].min()
            
            insights.append({
                'title': '🏆 Performance Range',
                'content': f'Best performing: {best_province} ({best_score:.2f}). Most challenging: {worst_province} ({worst_score:.2f}). Consider knowledge transfer from high-performing regions.',
                'type': 'info',
                'icon': '🎯'
            })
        
    except Exception as e:
        logger.error(f"Error generating geographic insights: {str(e)}")
        insights.append({
            'title': '⚠️ Analysis Error',
            'content': f'Unable to generate geographic insights: {str(e)}',
            'type': 'danger',
            'icon': '⚠️'
        })
    
    return insights

def get_provinces_by_region():
    """
    Get provinces grouped by major regions of Indonesia
    
    Returns:
        dict: Dictionary with region names as keys and province lists as values
    """
    regions = {
        'Sumatra': [
            'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Kepulauan Riau',
            'Jambi', 'Sumatera Selatan', 'Bangka Belitung', 'Bengkulu', 'Lampung'
        ],
        'Java': [
            'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'DI Yogyakarta', 
            'Jawa Timur', 'Banten'
        ],
        'Kalimantan': [
            'Kalimantan Barat', 'Kalimantan Tengah', 'Kalimantan Selatan',
            'Kalimantan Timur', 'Kalimantan Utara'
        ],
        'Sulawesi': [
            'Sulawesi Utara', 'Sulawesi Tengah', 'Sulawesi Selatan',
            'Sulawesi Tenggara', 'Gorontalo', 'Sulawesi Barat'
        ],
        'Eastern Indonesia': [
            'Bali', 'Nusa Tenggara Barat', 'NTB', 'Nusa Tenggara Timur', 'NTT',
            'Maluku', 'Maluku Utara', 'Papua', 'Papua Barat', 'Papua Selatan',
            'Papua Tengah', 'Papua Pegunungan', 'Papua Barat Daya'
        ]
    }
    
    return regions

def calculate_regional_statistics(scenario_data, risk_data, selected_scenario):
    """
    Calculate statistics by major regions - FIXED VERSION
    """
    try:
        # Filter data
        scenario_filtered = scenario_data[scenario_data['Scenario'] == selected_scenario].copy()
        risk_filtered = risk_data[risk_data['Scenario'] == selected_scenario].copy()
        
        # ✅ FIX: Aggregate to province level first
        scenario_agg = scenario_filtered.groupby('Provinsi').agg({
            'Predicted_Komposit': 'mean',
            'Uncertainty_Range': 'mean' if 'Uncertainty_Range' in scenario_filtered.columns else lambda x: 0
        }).reset_index()
        
        risk_agg = risk_filtered.groupby('Provinsi').agg({
            'Risk_Level': 'first'
        }).reset_index()
        
        # Get regional mapping
        regions = get_provinces_by_region()
        
        # Calculate regional stats
        regional_stats = []
        
        for region, provinces in regions.items():
            # ✅ FIX: Filter for provinces in this region
            region_scenario = scenario_agg[scenario_agg['Provinsi'].isin(provinces)]
            region_risk = risk_agg[risk_agg['Provinsi'].isin(provinces)]
            
            if not region_scenario.empty:
                # Calculate statistics
                avg_score = region_scenario['Predicted_Komposit'].mean()
                min_score = region_scenario['Predicted_Komposit'].min()
                max_score = region_scenario['Predicted_Komposit'].max()
                std_score = region_scenario['Predicted_Komposit'].std()
                
                # Risk distribution
                high_risk_count = len(region_risk[region_risk['Risk_Level'].isin(['Very High Risk', 'High Risk'])])
                total_provinces = len(region_scenario)
                risk_rate = (high_risk_count / total_provinces) * 100 if total_provinces > 0 else 0
                
                # Uncertainty statistics
                avg_uncertainty = region_scenario['Uncertainty_Range'].mean() if 'Uncertainty_Range' in region_scenario.columns else 0
                
                regional_stats.append({
                    'Region': region,
                    'Provinces_Count': total_provinces,
                    'Avg_Score': round(avg_score, 2),
                    'Min_Score': round(min_score, 2),
                    'Max_Score': round(max_score, 2),
                    'Score_Std': round(std_score, 3) if not pd.isna(std_score) else 0,
                    'High_Risk_Count': high_risk_count,
                    'Risk_Rate_%': round(risk_rate, 1),
                    'Avg_Uncertainty': round(avg_uncertainty, 3)
                })
        
        return pd.DataFrame(regional_stats)
        
    except Exception as e:
        logger.error(f"Error calculating regional statistics: {str(e)}")
        return pd.DataFrame()

def create_regional_comparison_chart(regional_stats):
    """
    Create regional comparison visualization
    
    Args:
        regional_stats (pd.DataFrame): Regional statistics dataframe
        
    Returns:
        plotly.graph_objects.Figure: Regional comparison chart
    """
    try:
        if regional_stats.empty:
            return None
        
        # Create subplot with secondary y-axis
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Food Security Score', 'Risk Rate by Region', 
                          'Score Variation', 'Province Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average scores
        fig.add_trace(
            go.Bar(x=regional_stats['Region'], y=regional_stats['Avg_Score'],
                   name='Avg Score', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Risk rates
        fig.add_trace(
            go.Bar(x=regional_stats['Region'], y=regional_stats['Risk_Rate_%'],
                   name='Risk Rate %', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Score variation
        fig.add_trace(
            go.Bar(x=regional_stats['Region'], y=regional_stats['Score_Std'],
                   name='Score Std Dev', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Province count
        fig.add_trace(
            go.Bar(x=regional_stats['Region'], y=regional_stats['Provinces_Count'],
                   name='Province Count', marker_color='lightyellow'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Regional Food Security Analysis",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating regional comparison chart: {str(e)}")
        return None

def export_geographic_data(scenario_data, risk_data, selected_scenario, export_format='csv'):
    """
    Export geographic data for external analysis
    
    Args:
        scenario_data (pd.DataFrame): Scenario predictions data
        risk_data (pd.DataFrame): Risk assessment data
        selected_scenario (str): Selected scenario for export
        export_format (str): Export format ('csv', 'json', 'geojson')
        
    Returns:
        str: Exported data as string
    """
    try:
        # Filter and merge data
        scenario_filtered = scenario_data[scenario_data['Scenario'] == selected_scenario]
        risk_filtered = risk_data[risk_data['Scenario'] == selected_scenario]
        export_data = pd.merge(scenario_filtered, risk_filtered, on=['Provinsi', 'Scenario'], how='left')
        
        # Add coordinates
        export_data['Latitude'] = export_data['Provinsi'].map(lambda x: INDONESIA_PROVINCES_COORDS.get(x, {}).get('lat', None))
        export_data['Longitude'] = export_data['Provinsi'].map(lambda x: INDONESIA_PROVINCES_COORDS.get(x, {}).get('lon', None))
        export_data['Province_Code'] = export_data['Provinsi'].map(lambda x: INDONESIA_PROVINCES_COORDS.get(x, {}).get('code', None))
        
        # Filter out provinces without coordinates
        export_data = export_data.dropna(subset=['Latitude', 'Longitude'])
        
        if export_format.lower() == 'csv':
            return export_data.to_csv(index=False)
        elif export_format.lower() == 'json':
            return export_data.to_json(orient='records', indent=2)
        elif export_format.lower() == 'geojson':
            # Create GeoJSON format
            features = []
            for _, row in export_data.iterrows():
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row['Longitude'], row['Latitude']]
                    },
                    "properties": {
                        "province": row['Provinsi'],
                        "scenario": row['Scenario'],
                        "predicted_score": row['Predicted_Komposit'],
                        "risk_level": row['Risk_Level'],
                        "uncertainty": row.get('Uncertainty_Range', 0),
                        "province_code": row.get('Province_Code', '')
                    }
                }
                features.append(feature)
            
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            
            import json
            return json.dumps(geojson, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
    except Exception as e:
        logger.error(f"Error exporting geographic data: {str(e)}")
        raise GeoVisualizationError(f"Failed to export data: {str(e)}")

def create_risk_heatmap(scenario_data, risk_data, selected_scenario):
    """
    Create a risk heatmap visualization with Province -> Kabupaten hierarchy
    NEW FEATURE: Click on province to see districts within
    """
    try:
        # Validate inputs
        if scenario_data is None or risk_data is None:
            return None
            
        if scenario_data.empty or risk_data.empty:
            return None
            
        # Filter data for selected scenario
        scenario_filtered = scenario_data[scenario_data['Scenario'] == selected_scenario].copy()
        risk_filtered = risk_data[risk_data['Scenario'] == selected_scenario].copy()
        
        if scenario_filtered.empty or risk_filtered.empty:
            return None
        
        # ✅ NEW: Keep district-level data for hierarchical treemap
        # Clean merge at district level
        risk_cols = ['Provinsi', 'Scenario', 'Risk_Level'] 
        if 'Kabupaten' in risk_filtered.columns:
            risk_cols.append('Kabupaten')
        if 'Uncertainty_Range' not in scenario_filtered.columns and 'Uncertainty_Range' in risk_filtered.columns:
            risk_cols.append('Uncertainty_Range')
            
        risk_filtered_clean = risk_filtered[risk_cols]
        
        # Merge at district level
        merged_data = pd.merge(scenario_filtered, risk_filtered_clean, 
                              on=['Provinsi', 'Scenario'], how='left')
        
        if merged_data.empty:
            return None
        
        # ✅ NEW: Handle kabupaten column from merge
        kabupaten_col = 'Kabupaten'
        if 'Kabupaten_x' in merged_data.columns:
            kabupaten_col = 'Kabupaten_x'
        elif 'Kabupaten_y' in merged_data.columns:
            kabupaten_col = 'Kabupaten_y'
        
        # Create clean kabupaten column
        if kabupaten_col != 'Kabupaten':
            merged_data['Kabupaten'] = merged_data[kabupaten_col]
        
        # Handle predicted score column
        predicted_col = 'Predicted_Komposit'
        if 'Predicted_Komposit_x' in merged_data.columns:
            predicted_col = 'Predicted_Komposit_x'
        elif 'Predicted_Komposit_y' in merged_data.columns:
            predicted_col = 'Predicted_Komposit_y'
        
        # Create clean predicted score column
        if predicted_col != 'Predicted_Komposit':
            merged_data['Predicted_Komposit'] = merged_data[predicted_col]
        
        # Create risk level mapping for sizing
        risk_mapping = {
            'Very High Risk': 4,
            'High Risk': 3,
            'Medium Risk': 2,
            'Low Risk': 1
        }
        
        merged_data['Risk_Numeric'] = merged_data['Risk_Level'].map(risk_mapping)
        
        # ✅ NEW: Get regional mapping and create hierarchy
        regions = get_provinces_by_region()
        
        # Add region information
        province_to_region = {}
        for region, provinces in regions.items():
            for province in provinces:
                province_to_region[province] = region
        
        merged_data['Region'] = merged_data['Provinsi'].map(province_to_region)
        
        # Filter only provinces with region mapping and clean kabupaten names
        merged_data = merged_data.dropna(subset=['Region', 'Kabupaten'])
        merged_data['Kabupaten'] = merged_data['Kabupaten'].astype(str).str.strip()
        
        if merged_data.empty:
            return None
        
        # ✅ NEW: Ensure unique districts per province (in case of duplicates)
        merged_data = merged_data.drop_duplicates(subset=['Provinsi', 'Kabupaten'])
        
        # ✅ NEW: Create hierarchical treemap with Region -> Province -> Kabupaten
        fig = px.treemap(
            merged_data,
            path=['Region', 'Provinsi', 'Kabupaten'],   # ✅ 3-level hierarchy
            values='Risk_Numeric',                       # Size based on risk
            color='Predicted_Komposit',                  # Color based on food security score
            color_continuous_scale='RdYlGn',             # ✅ Green = good, Red = bad
            range_color=[1, 6],                          # Full score range
            title=f'Food Security Risk Heatmap - {selected_scenario}<br><sub>🖱️ Click provinces to see districts | 🎯 Size = Risk Level | 🌈 Color = Food Security Score</sub>',
            hover_data={
                'Risk_Level': True, 
                'Predicted_Komposit': ':.2f'
            },
            color_continuous_midpoint=3.5,
            height=700  # ✅ Increased height for better hierarchy view
        )
        
        # ✅ ENHANCED: Better layout and styling
        fig.update_layout(
            font_size=10,
            title_font_size=16,
            margin=dict(t=80, l=10, r=10, b=10),
            coloraxis_colorbar=dict(
                title="Food Security Score",
                titleside="right",
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5, 6],
                ticktext=['1 (Very Poor)', '2 (Poor)', '3 (Fair)', '4 (Good)', '5 (Very Good)', '6 (Excellent)'],
                len=0.7,
                thickness=15
            )
        )
        
        # ✅ NEW: Update treemap styling for better hierarchy visualization
        fig.update_traces(
            # Better text display
            textinfo="label+value",
            textposition="middle center",
            # Custom hover template
            hovertemplate='<b>%{label}</b><br>' +
                         'Food Security Score: %{color:.2f}<br>' +
                         'Risk Level: %{customdata[0]}<br>' +
                         'Risk Value: %{value}<br>' +
                         '<extra></extra>',
            # Better borders
            marker=dict(
                line=dict(width=2, color='white'),
                colorbar=dict(thickness=15)
            )
        )
        
        # ✅ NEW: Add interactive guide
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=(
                "📊 <b>Interactive Guide:</b><br>"
                "🖱️ <b>Click</b> on regions/provinces to drill down<br>"
                "🎯 <b>Box Size:</b> Risk level (larger = higher risk)<br>"
                "🌈 <b>Color:</b> Food security score<br>"
                "🟢 Green = Good | 🟡 Yellow = Fair | 🔴 Red = Poor"
            ),
            showarrow=False,
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="darkblue",
            borderwidth=1,
            borderpad=8
        )
        
        # ✅ NEW: Add statistics summary
        total_districts = len(merged_data)
        total_provinces = merged_data['Provinsi'].nunique()
        total_regions = merged_data['Region'].nunique()
        high_risk_districts = len(merged_data[merged_data['Risk_Level'].isin(['Very High Risk', 'High Risk'])])
        
        fig.add_annotation(
            x=0.98, y=0.02,
            xref="paper", yref="paper",
            text=(
                f"📈 <b>Coverage Summary:</b><br>"
                f"🏛️ Regions: {total_regions}<br>"
                f"📍 Provinces: {total_provinces}<br>" 
                f"🏘️ Districts: {total_districts}<br>"
                f"🚨 High Risk: {high_risk_districts} ({high_risk_districts/total_districts*100:.1f}%)"
            ),
            showarrow=False,
            font=dict(size=10, color="darkgreen"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="darkgreen",
            borderwidth=1,
            borderpad=6,
            xanchor="right"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating hierarchical risk heatmap: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def validate_coordinates():
    """
    Validate that all provinces in the coordinates dictionary have valid lat/lon
    
    Returns:
        tuple: (is_valid, invalid_provinces)
    """
    invalid_provinces = []
    
    for province, coords in INDONESIA_PROVINCES_COORDS.items():
        lat = coords.get('lat', 0)
        lon = coords.get('lon', 0)
        
        # Check if coordinates are within Indonesia's approximate bounds
        if not (-11 <= lat <= 6) or not (95 <= lon <= 141):
            invalid_provinces.append(province)
    
    return len(invalid_provinces) == 0, invalid_provinces

def get_coordinate_coverage(data):
    """
    Check coordinate coverage for provinces in the data
    
    Args:
        data (pd.DataFrame): Data containing 'Provinsi' column
        
    Returns:
        dict: Coverage statistics
    """
    if 'Provinsi' not in data.columns:
        return {'error': 'No Provinsi column found'}
    
    unique_provinces = data['Provinsi'].unique()
    
    covered_provinces = []
    missing_provinces = []
    
    for province in unique_provinces:
        if province in INDONESIA_PROVINCES_COORDS:
            covered_provinces.append(province)
        else:
            missing_provinces.append(province)
    
    coverage_stats = {
        'total_provinces': len(unique_provinces),
        'covered_provinces': len(covered_provinces),
        'missing_provinces': len(missing_provinces),
        'coverage_percentage': (len(covered_provinces) / len(unique_provinces)) * 100 if unique_provinces.size > 0 else 0,
        'covered_list': covered_provinces,
        'missing_list': missing_provinces
    }
    
    return coverage_stats

# Utility functions for integration with main dashboard
def get_available_map_types():
    """
    Get list of available map visualization types
    
    Returns:
        dict: Available map types with descriptions
    """
    map_types = {
        'interactive_plotly': {
            'name': 'Interactive Plotly Map',
            'description': 'Scatter map with hover information and zoom capabilities',
            'requirements': ['plotly'],
            'available': True
        },
        'animated_scenarios': {
            'name': 'Animated Scenario Map',
            'description': 'Animated map showing changes across different scenarios',
            'requirements': ['plotly'],
            'available': True
        },
        'risk_heatmap': {
            'name': 'Risk Heatmap',
            'description': 'Treemap visualization showing risk distribution by region',
            'requirements': ['plotly'],
            'available': True
        },
        'folium_interactive': {
            'name': 'Folium Interactive Map',
            'description': 'Full-featured interactive map with multiple layers',
            'requirements': ['folium'],
            'available': FOLIUM_AVAILABLE
        }
    }
    
    return map_types

def get_supported_export_formats():
    """
    Get list of supported export formats for geographic data
    
    Returns:
        list: Supported export formats
    """
    return ['csv', 'json', 'geojson']

# Main integration function
def create_geo_visualization_interface(scenario_data, risk_data):
    """
    Main interface function for geographic visualizations in Streamlit
    
    Args:
        scenario_data (pd.DataFrame): Scenario predictions data
        risk_data (pd.DataFrame): Risk assessment data
        
    Returns:
        dict: Status and results of visualization creation
    """
    try:
        # Validate data
        is_valid, errors = validate_geo_data(scenario_data, risk_data)
        if not is_valid:
            return {
                'status': 'error',
                'message': f"Data validation failed: {'; '.join(errors)}",
                'visualizations': {}
            }
        
        # Check coordinate coverage
        coverage = get_coordinate_coverage(scenario_data)
        
        # Get available map types
        available_maps = get_available_map_types()
        
        return {
            'status': 'success',
            'message': 'Geographic visualization ready',
            'coordinate_coverage': coverage,
            'available_maps': available_maps,
            'export_formats': get_supported_export_formats(),
            'visualizations': {
                'scenarios': scenario_data['Scenario'].unique().tolist(),
                'provinces': scenario_data['Provinsi'].unique().tolist(),
                'total_data_points': len(scenario_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in geo visualization interface: {str(e)}")
        return {
            'status': 'error',
            'message': f"Failed to initialize geo visualization: {str(e)}",
            'visualizations': {}
        }