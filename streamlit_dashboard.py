# =============================================================================
# STREAMLIT DASHBOARD FOR FOOD SECURITY FORECASTING
# Comprehensive interactive dashboard for food security analysis
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.food_security_forecasting import FoodSecurityForecaster, FoodSecurityConfig
from src.visualization import FoodSecurityVisualizer
from src.utils import create_metric_card, create_status_card, create_sample_data

# Page configuration
st.set_page_config(
    page_title="Food Security Forecasting Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ffffff;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #3498db !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .metric-card * {
        color: #2c3e50 !important;
    }
    .warning-card {
        background-color: #fff3cd !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #ffc107 !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .warning-card * {
        color: #856404 !important;
    }
    .success-card {
        background-color: #d4edda !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #28a745 !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .success-card * {
        color: #155724 !important;
    }
    .info-card {
        background-color: #d1ecf1 !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #17a2b8 !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .info-card * {
        color: #0c5460 !important;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f4;
    }
    
    /* Light text for dark backgrounds */
    .main-content h1, .main-content h2, .main-content h3, .main-content h4 {
        color: #ffffff !important;
    }
    
    .main-content p, .main-content li {
        color: #e9ecef !important;
    }
    
    /* Override Streamlit's default text colors for dark theme */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
    }
    
    .stMarkdown p {
        color: #e9ecef !important;
    }
    
    .stMarkdown li {
        color: #e9ecef !important;
    }
    
    /* Special styling for welcome screen */
    .welcome-section h2 {
        color: #ffffff !important;
        font-size: 1.5rem !important;
    }
    
    .welcome-section h3 {
        color: #17a2b8 !important;
        font-size: 1.2rem !important;
    }
    
    .welcome-section p, .welcome-section li {
        color: #ced4da !important;
        font-size: 0.95rem !important;
    }
    
    .welcome-section strong {
        color: #ffffff !important;
    }
    
    /* Force dark text in all metric cards */
    div[class*="metric-card"] h3 {
        color: #2c3e50 !important;
    }
    
    div[class*="metric-card"] h2 {
        color: #1f77b4 !important;
    }
    
    div[class*="metric-card"] p {
        color: #495057 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

def load_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    provinces = ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Sumatera Utara', 
                'Sumatera Barat', 'Sulawesi Selatan', 'Kalimantan Timur', 'Bali', 'NTB']
    
    data = []
    for year in range(2018, 2024):
        for province in provinces:
            for i in range(np.random.randint(10, 20)):  # Random number of districts per province
                data.append({
                    'Tahun': year,
                    'Provinsi': province,
                    'Kabupaten': f'{province}_Kab_{i+1}',
                    'Kemiskinan (%)': np.random.uniform(5, 25),
                    'Pengeluaran Pangan (%)': np.random.uniform(40, 70),
                    'Tanpa Air Bersih (%)': np.random.uniform(10, 40),
                    'Lama Sekolah Perempuan (tahun)': np.random.uniform(6, 12),
                    'Rasio Tenaga Kesehatan': np.random.uniform(0.5, 3.0),
                    'Angka Harapan Hidup (tahun)': np.random.uniform(65, 75),
                    'Komposit': np.random.randint(1, 7)
                })
    
    return pd.DataFrame(data)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #ffffff; font-size: 2.5rem; margin-bottom: 0.5rem;">
            🌾 Food Security Forecasting Dashboard
        </h1>
        <p style="color: #ced4da; font-size: 1.1rem; margin: 0;">
            Advanced Machine Learning for Policy Decision Support
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Dashboard Controls")
        
        # Data loading section
        st.markdown("### 📁 Data Loading")
        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV File", "Use Sample Data"],
            help="Upload your own data or use the provided sample dataset"
        )
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                help="Upload CSV file with food security indicators"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        else:
            if st.button("🔄 Load Sample Data"):
                df = load_sample_data()
                st.session_state.uploaded_data = df
                st.success(f"✅ Sample data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Analysis controls
        if st.session_state.uploaded_data is not None:
            st.markdown("### 🔧 Analysis Settings")
            
            # Model parameters
            st.markdown("#### Model Configuration")
            n_estimators = st.slider("Number of Trees", 100, 500, 200, 50)
            max_depth = st.selectbox("Max Depth", [None, 10, 20, 30], index=0)
            
            # Analysis execution
            if st.button("🚀 Run Full Analysis", type="primary"):
                with st.spinner("Running comprehensive food security analysis..."):
                    try:
                        # Initialize config with custom parameters
                        config = FoodSecurityConfig()
                        config.PARAM_GRID['n_estimators'] = [n_estimators]
                        if max_depth is not None:
                            config.PARAM_GRID['max_depth'] = [max_depth]
                        
                        # Run analysis
                        forecaster = FoodSecurityForecaster(config)
                        forecaster.run_full_analysis(st.session_state.uploaded_data)
                        
                        st.session_state.forecaster = forecaster
                        st.session_state.analysis_complete = True
                        st.success("✅ Analysis completed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.exception(e)
        
        # Information section
        st.markdown("### ℹ️ Information")
        st.info("""
        **Dashboard Features:**
        - 📊 Data overview and exploration
        - 🤖 Machine learning model training
        - 📈 Performance evaluation
        - 🔮 Scenario forecasting
        - ⚠️ Risk assessment
        - 📋 Comprehensive reporting
        """)
    
    # Main content area
    if st.session_state.uploaded_data is None:
        # Welcome screen
        st.markdown('<div class="welcome-section">', unsafe_allow_html=True)
        
        st.markdown("## 👋 Welcome to the Food Security Forecasting Dashboard")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 🎯 What this dashboard does:
            
            1. **📊 Data Analysis**: Comprehensive exploration of food security indicators
            2. **🤖 Machine Learning**: Advanced Random Forest modeling with time series validation  
            3. **🔮 Forecasting**: Scenario-based predictions for future food security
            4. **⚠️ Risk Assessment**: Early warning system for vulnerable regions
            5. **📋 Reporting**: Detailed insights and policy recommendations
            
            ### 🚀 Getting Started:
            1. Upload your CSV file or use sample data from the sidebar
            2. Configure analysis settings
            3. Run the full analysis
            4. Explore results across different tabs
            
            ### 📝 Data Requirements:
            Your CSV should include columns for:
            - Tahun (Year)
            - Provinsi (Province) 
            - Kabupaten (District/Regency)
            - Kemiskinan (%) - Poverty rate
            - Pengeluaran Pangan (%) - Food expenditure
            - Tanpa Air Bersih (%) - Without clean water access
            - Lama Sekolah Perempuan (tahun) - Women's education years
            - Rasio Tenaga Kesehatan - Healthcare worker ratio
            - Angka Harapan Hidup (tahun) - Life expectancy
            - Komposit - Food security composite score (1-6)
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Show data overview when data is loaded
    df = st.session_state.uploaded_data
    
    # Create tabs for different sections
    if st.session_state.analysis_complete:
        tabs = st.tabs([
            "📊 Data Overview", 
            "🤖 Model Performance", 
            "🎯 Feature Analysis",
            "🔮 Scenario Forecasting", 
            "⚠️ Risk Assessment",
            "📋 Summary Report"
        ])
    else:
        tabs = st.tabs(["📊 Data Overview"])
    
    # Data Overview Tab
    with tabs[0]:
        show_data_overview(df)
    
    # Model Performance Tab
    if st.session_state.analysis_complete:
        with tabs[1]:
            show_model_performance()
        
        with tabs[2]:
            show_feature_analysis()
        
        with tabs[3]:
            show_scenario_forecasting()
        
        with tabs[4]:
            show_risk_assessment()
        
        with tabs[5]:
            show_summary_report()

def show_data_overview(df):
    """Show data overview section"""
    st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📋 Total Records</h3>
            <h2>{:,}</h2>
            <p>Complete data records in dataset</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🗺️ Provinces</h3>
            <h2>{}</h2>
            <p>Indonesian provinces covered</p>
        </div>
        """.format(df['Provinsi'].nunique()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📅 Years Range</h3>
            <h2>{} - {}</h2>
            <p>Time period of analysis</p>
        </div>
        """.format(df['Tahun'].min(), df['Tahun'].max()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🏢 Districts</h3>
            <h2>{}</h2>
            <p>Districts/regencies included</p>
        </div>
        """.format(df['Kabupaten'].nunique()), unsafe_allow_html=True)
    
    # Visualizations
    config = FoodSecurityConfig()
    visualizer = FoodSecurityVisualizer(config)
    
    # Data distribution plots
    fig_overview = visualizer.plot_data_overview(df)
    st.plotly_chart(fig_overview, use_container_width=True)
    
    # Time series trends
    st.markdown("### 📈 Trends Over Time")
    fig_trends = visualizer.plot_time_series_trends(df)
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Feature Correlations")
    fig_corr = visualizer.plot_correlation_matrix(df)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Data table
    st.markdown("### 📋 Data Sample")
    st.dataframe(df.head(20), use_container_width=True)

def show_model_performance():
    """Show model performance section"""
    st.markdown('<h2 class="section-header">🤖 Model Performance</h2>', unsafe_allow_html=True)
    
    forecaster = st.session_state.forecaster
    cv_results = forecaster.cv_results  # pastikan ini adalah DataFrame
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Display metric cards
    with col1:
        card_html = create_metric_card(
            "Mean R²", 
            f"{cv_results['r2'].mean():.3f}", 
            "Model accuracy score",
            "🎯"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col2:
        card_html = create_metric_card(
            "RMSE", 
            f"{cv_results['rmse'].mean():.3f}", 
            "Root Mean Square Error",
            "📊"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col3:
        card_html = create_metric_card(
            "Stability", 
            f"{cv_results['r2'].std():.3f}", 
            "Model consistency score",
            "🎲"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col4:
        card_html = create_metric_card(
            "CV Folds", 
            str(len(cv_results)), 
            "Cross-validation splits",
            "🔄"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    # Performance visualizations
    config = FoodSecurityConfig()
    visualizer = FoodSecurityVisualizer(config)
    
    fig_performance = visualizer.plot_model_performance(cv_results)
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # Detailed results table
    st.markdown("### 📋 Cross-Validation Results")
    st.dataframe(cv_results.round(4), use_container_width=True)


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

def show_feature_analysis():
    """Show feature analysis section"""
    st.markdown('<h2 class="section-header">🎯 Feature Analysis</h2>', unsafe_allow_html=True)
    
    forecaster = st.session_state.forecaster
    feature_importance = forecaster.feature_importance
    
    # Top features
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🏆 Top 3 Most Important Features")
        for i, (_, row) in enumerate(feature_importance.head(3).iterrows()):
            content = f"<p style='margin: 0;'><strong>Importance: {row['Importance']:.3f}</strong></p>"
            card_html = create_status_card(
                f"#{i+1} {row['Feature']}", 
                content,
                "success",
                "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            )
            st.markdown(card_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Feature Statistics")
        stats_content = f"""
        <p><strong>Total Features:</strong> {len(feature_importance)}</p>
        <p><strong>Importance Range:</strong> {feature_importance['Importance'].min():.3f} - {feature_importance['Importance'].max():.3f}</p>
        <p><strong>Mean Importance:</strong> {feature_importance['Importance'].mean():.3f}</p>
        <p><strong>Top Feature Dominance:</strong> {(feature_importance.iloc[0]['Importance'] / feature_importance['Importance'].sum() * 100):.1f}%</p>
        """
        card_html = create_status_card(
            "Statistical Summary", 
            stats_content,
            "info",
            "📈"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    # Feature importance visualization
    config = FoodSecurityConfig()
    visualizer = FoodSecurityVisualizer(config)
    
    fig_importance = visualizer.plot_feature_importance(feature_importance)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Feature importance table
    st.markdown("### 📋 Complete Feature Ranking")
    st.dataframe(feature_importance.round(4), use_container_width=True)

def show_scenario_forecasting():
    """Show scenario forecasting section"""
    st.markdown('<h2 class="section-header">🔮 Scenario Forecasting (2025)</h2>', unsafe_allow_html=True)
    
    forecaster = st.session_state.forecaster
    
    if forecaster.scenario_predictions is None:
        st.warning("⚠️ Scenario predictions not available. Please re-run the analysis.")
        return
    
    scenario_predictions = forecaster.scenario_predictions
    
    # Scenario overview
    scenarios = scenario_predictions['Scenario'].unique()
    
    st.markdown("### 📊 Scenario Overview")
    cols = st.columns(len(scenarios))
    
    for i, scenario in enumerate(scenarios):
        scenario_data = scenario_predictions[scenario_predictions['Scenario'] == scenario]
        avg_prediction = scenario_data['Predicted_Komposit'].mean()
        
        with cols[i]:
            # Determine card type and icon based on scenario
            if "Optimistic" in scenario:
                card_type, icon = "success", "🟢"
            elif "Moderate" in scenario or "Conservative" in scenario:
                card_type, icon = "info", "🔵" 
            elif "Status Quo" in scenario:
                card_type, icon = "warning", "🟡"
            else:
                card_type, icon = "info", "⚪"
            
            content = f"""
            <h3 style="margin: 0.5rem 0; font-size: 1.2rem;">Avg Score: {avg_prediction:.2f}</h3>
            <p><strong>Provinces:</strong> {len(scenario_data)}</p>
            <p><strong>Range:</strong> {scenario_data['Predicted_Komposit'].min():.2f} - {scenario_data['Predicted_Komposit'].max():.2f}</p>
            <p><strong>Std Dev:</strong> {scenario_data['Predicted_Komposit'].std():.2f}</p>
            """
            
            card_html = create_status_card(scenario, content, card_type, icon)
            st.markdown(card_html, unsafe_allow_html=True)
    
    # Scenario comparison visualization
    config = FoodSecurityConfig()
    visualizer = FoodSecurityVisualizer(config)
    
    fig_scenarios = visualizer.plot_scenario_comparison(scenario_predictions)
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    # Provincial rankings
    st.markdown("### 🏆 Provincial Rankings by Scenario")
    
    selected_scenario = st.selectbox("Select Scenario:", scenarios)
    scenario_data = scenario_predictions[scenario_predictions['Scenario'] == selected_scenario]
    
    top_provinces = scenario_data.nlargest(10, 'Predicted_Komposit')[
        ['Provinsi', 'Predicted_Komposit', 'Lower_CI_95', 'Upper_CI_95', 'Uncertainty_Range']
    ].round(3)
    
    st.dataframe(top_provinces, use_container_width=True)

def show_risk_assessment():
    """Show risk assessment section"""
    st.markdown('<h2 class="section-header">⚠️ Risk Assessment</h2>', unsafe_allow_html=True)
    
    forecaster = st.session_state.forecaster
    
    if forecaster.risk_assessment is None:
        st.warning("⚠️ Risk assessment not available. Please re-run the analysis.")
        return
    
    risk_assessment = forecaster.risk_assessment
    status_quo_risk = risk_assessment[risk_assessment['Scenario'] == 'Status Quo']
    
    # Risk level distribution
    risk_dist = status_quo_risk['Risk_Level'].value_counts()
    
    st.markdown("### 🚨 Risk Level Distribution")
    
    cols = st.columns(len(risk_dist))
    risk_configs = {
        'Very High Risk': {'type': 'danger', 'icon': '🔴', 'action': 'immediate'},
        'High Risk': {'type': 'danger', 'icon': '🟠', 'action': 'urgent'}, 
        'Medium Risk': {'type': 'warning', 'icon': '🟡', 'action': 'monitoring'},
        'Medium Risk (High Uncertainty)': {'type': 'warning', 'icon': '🟡', 'action': 'monitoring'},
        'Low Risk': {'type': 'success', 'icon': '🟢', 'action': 'standard'}
    }
    
    for i, (risk_level, count) in enumerate(risk_dist.items()):
        with cols[i]:
            config = risk_configs.get(risk_level, {'type': 'info', 'icon': '⚪', 'action': 'review'})
            percentage = (count / len(status_quo_risk)) * 100
            
            content = f"""
            <h2 style="text-align: center; margin: 0.5rem 0; font-size: 2rem;">{count}</h2>
            <p style="text-align: center; margin: 0.25rem 0;"><strong>{percentage:.1f}%</strong> of provinces</p>
            <p style="text-align: center; margin: 0.25rem 0; font-size: 0.85rem;">
                Requires <strong>{config['action']}</strong> attention
            </p>
            """
            
            card_html = create_status_card(risk_level, content, config['type'], config['icon'])
            st.markdown(card_html, unsafe_allow_html=True)
    
    # High risk provinces
    high_risk_provinces = status_quo_risk[
        status_quo_risk['Risk_Level'].isin(['Very High Risk', 'High Risk'])
    ]
    
    if len(high_risk_provinces) > 0:
        st.markdown("### 🚨 High Risk Provinces")
        
        # Create alert card for high risk provinces
        provinces_list = high_risk_provinces['Provinsi'].tolist()
        if len(provinces_list) <= 5:
            provinces_text = ', '.join(provinces_list)
        else:
            provinces_text = ', '.join(provinces_list[:5]) + f'... and {len(provinces_list) - 5} more'
        
        alert_content = f"""
        <p><strong>Total High Risk Provinces:</strong> {len(high_risk_provinces)}</p>
        <p><strong>Provinces:</strong> {provinces_text}</p>
        <p><strong>Average Predicted Score:</strong> {high_risk_provinces['Predicted_Komposit'].mean():.2f}</p>
        <p><strong>Average Uncertainty:</strong> {high_risk_provinces['Uncertainty_Range'].mean():.3f}</p>
        """
        
        alert_card = create_status_card(
            "Critical Alert - Immediate Action Required", 
            alert_content, 
            "danger", 
            "🚨"
        )
        st.markdown(alert_card, unsafe_allow_html=True)
        
        st.dataframe(
            high_risk_provinces[['Provinsi', 'Predicted_Komposit', 'Risk_Level', 'Uncertainty_Range']].round(3),
            use_container_width=True
        )
    else:
        # No high risk alert
        success_content = "<p>No provinces currently classified as high risk. Continue monitoring for changes.</p>"
        success_card = create_status_card(
            "No Critical Alerts", 
            success_content, 
            "success", 
            "✅"
        )
        st.markdown(success_card, unsafe_allow_html=True)
    
    # Early warnings
    warnings = forecaster.risk_assessor.generate_early_warnings(risk_assessment, 'Status Quo')
    
    if warnings:
        st.markdown("### 🚨 Early Warning Alerts")
        for warning in warnings:
            warning_content = f"""
            <p>{warning['Message']}</p>
            <p><strong>Affected Provinces:</strong> {', '.join(warning['Provinces'][:5])}{'...' if len(warning['Provinces']) > 5 else ''}</p>
            <p><strong>Total Count:</strong> {warning['Count']} provinces</p>
            """
            
            warning_card = create_status_card(
                warning['Type'], 
                warning_content, 
                "warning", 
                "⚠️"
            )
            st.markdown(warning_card, unsafe_allow_html=True)

def show_summary_report():
    """Show comprehensive summary report"""
    st.markdown('<h2 class="section-header">📋 Summary Report</h2>', unsafe_allow_html=True)
    
    forecaster = st.session_state.forecaster
    
    # Generate summary report
    summary = forecaster.get_summary_report()
    
    # Executive Summary
    st.markdown("### 📊 Executive Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 Model Performance")
        st.markdown(f"""
        <div class="metric-card">
            <p><strong>R² Score:</strong> <span style="color: #1f77b4; font-weight: bold;">{summary['model_performance']['mean_r2']:.3f}</span></p>
            <p><strong>Model Stability:</strong> <span style="color: #28a745; font-weight: bold;">{summary['model_performance']['r2_stability']:.3f}</span></p>
            <p><strong>RMSE:</strong> <span style="color: #dc3545; font-weight: bold;">{summary['model_performance']['mean_rmse']:.3f}</span></p>
            <p><strong>Performance Grade:</strong> <span style="color: #6f42c1; font-weight: bold;">{"Excellent" if summary['model_performance']['mean_r2'] > 0.8 else "Good" if summary['model_performance']['mean_r2'] > 0.6 else "Moderate"}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🎯 Key Factors")
        for i, factor in enumerate(summary['top_features'][:3], 1):
            st.markdown(f"""
            <div class="success-card" style="margin: 0.25rem 0; padding: 0.5rem;">
                <p style="margin: 0;"><strong>{i}. {factor}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Data Overview")
        st.markdown(f"""
        <div class="info-card">
            <p><strong>Total Records:</strong> <span style="color: #0c5460; font-weight: bold;">{summary['data_info']['total_records']:,}</span></p>
            <p><strong>Provinces:</strong> <span style="color: #0c5460; font-weight: bold;">{summary['data_info']['provinces']}</span></p>
            <p><strong>Years:</strong> <span style="color: #0c5460; font-weight: bold;">{summary['data_info']['years_range']}</span></p>
            <p><strong>Data Quality:</strong> <span style="color: #155724; font-weight: bold;">High</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'risk_summary' in summary:
            st.markdown("#### ⚠️ Risk Assessment")
            risk_rate = (summary['risk_summary']['high_risk_provinces']/summary['risk_summary']['total_provinces']*100)
            risk_color = "#721c24" if risk_rate > 30 else "#856404" if risk_rate > 10 else "#155724"
            st.markdown(f"""
            <div class="warning-card">
                <p><strong>High Risk Provinces:</strong> <span style="color: {risk_color}; font-weight: bold;">{summary['risk_summary']['high_risk_provinces']}/{summary['risk_summary']['total_provinces']}</span></p>
                <p><strong>Risk Rate:</strong> <span style="color: {risk_color}; font-weight: bold;">{risk_rate:.1f}%</span></p>
                <p><strong>Status:</strong> <span style="color: {risk_color}; font-weight: bold;">{"Critical" if risk_rate > 30 else "Moderate" if risk_rate > 10 else "Good"}</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Insights
    st.markdown("### 💡 Key Insights")
    
    insights = []
    
    # Model performance insights
    if summary['model_performance']['mean_r2'] > 0.8:
        insights.append("✅ **Excellent Model Performance**: The model shows strong predictive capability with R² > 0.8")
    elif summary['model_performance']['mean_r2'] > 0.6:
        insights.append("✅ **Good Model Performance**: The model shows reliable predictive capability")
    else:
        insights.append("⚠️ **Moderate Model Performance**: Consider additional data sources or feature engineering")
    
    # Stability insights
    if summary['model_performance']['r2_stability'] < 0.1:
        insights.append("✅ **High Model Stability**: Consistent performance across time periods")
    else:
        insights.append("⚠️ **Variable Model Performance**: Performance varies across different time periods")
    
    # Feature insights
    top_feature = summary['top_features'][0] if summary['top_features'] else "Unknown"
    if 'Kemiskinan' in top_feature:
        insights.append("💰 **Poverty Focus**: Poverty reduction should be the primary policy focus")
    elif 'Pendidikan' in top_feature or 'Sekolah' in top_feature:
        insights.append("🎓 **Education Priority**: Educational improvements are critical for food security")
    elif 'Kesehatan' in top_feature:
        insights.append("🏥 **Healthcare Focus**: Healthcare system strengthening is essential")
    
    # Risk insights
    if 'risk_summary' in summary:
        risk_rate = summary['risk_summary']['high_risk_provinces'] / summary['risk_summary']['total_provinces']
        if risk_rate > 0.3:
            insights.append("🚨 **High Risk Alert**: Over 30% of provinces are at high risk - immediate intervention needed")
        elif risk_rate > 0.1:
            insights.append("⚠️ **Moderate Risk**: Significant number of provinces require attention")
        else:
            insights.append("✅ **Low Risk Environment**: Most provinces show good food security outlook")
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Policy Recommendations
    st.markdown("### 🎯 Policy Recommendations")
    
    recommendations = []
    
    # Based on top features
    if summary['top_features']:
        top_3 = summary['top_features'][:3]
        recommendations.append(f"**Priority Focus**: Target the top 3 predictive factors: {', '.join(top_3)}")
    
    # Based on model performance
    if summary['model_performance']['mean_r2'] > 0.7:
        recommendations.append("**Model Deployment**: High model accuracy supports deployment for policy planning")
    
    # Based on risk assessment
    if 'risk_summary' in summary and summary['risk_summary']['high_risk_provinces'] > 0:
        recommendations.append(f"**Immediate Action**: {summary['risk_summary']['high_risk_provinces']} provinces require immediate intervention")
    
    # General recommendations
    recommendations.extend([
        "**Monitoring System**: Establish regular monitoring using this forecasting system",
        "**Data Quality**: Maintain and improve data collection for better predictions",
        "**Scenario Planning**: Use scenario analysis for policy impact assessment",
        "**Early Warning**: Implement early warning system based on risk assessment"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Download Reports
    st.markdown("### 📥 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Model Results"):
            csv_data = forecaster.cv_results.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="model_performance_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🎯 Download Feature Importance"):
            csv_data = forecaster.feature_importance.to_csv(index=False)
            st.download_button(
                label="Download CSV", 
                data=csv_data,
                file_name="feature_importance.csv",
                mime="text/csv"
            )
    
    with col3:
        if forecaster.scenario_predictions is not None:
            if st.button("🔮 Download Predictions"):
                csv_data = forecaster.scenario_predictions.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="scenario_predictions_2025.csv",
                    mime="text/csv"
                )

def create_footer():
    """Create dashboard footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #ced4da; padding: 2rem;'>
        <p style='color: #ffffff; font-size: 1.1rem; margin-bottom: 0.5rem;'>
            🌾 Food Security Forecasting Dashboard | Built with Streamlit & Python
        </p>
        <p style='color: #adb5bd; font-size: 0.9rem;'>
            Advanced Machine Learning for Policy Decision Support
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    create_footer()