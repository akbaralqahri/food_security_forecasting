# =============================================================================
# EXAMPLE USAGE SCRIPT - FOOD SECURITY FORECASTING
# Demonstrates how to use the structured modules
# =============================================================================

import pandas as pd
import numpy as np
from food_security_forecasting import FoodSecurityForecaster, FoodSecurityConfig
from visualization import FoodSecurityVisualizer

def create_sample_data():
    """Create sample data for demonstration"""
    print("📊 Creating sample data...")
    
    np.random.seed(42)
    
    provinces = [
        'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Sumatera Utara',
        'Sumatera Barat', 'Sulawesi Selatan', 'Kalimantan Timur', 'Bali', 'NTB',
        'Sumatera Selatan', 'Lampung', 'Kalimantan Selatan', 'Sulawesi Utara', 'Yogyakarta'
    ]
    
    data = []
    for year in range(2018, 2024):
        for province in provinces:
            # Number of districts varies by province
            n_districts = np.random.randint(8, 25)
            
            # Province-specific characteristics
            base_poverty = np.random.uniform(8, 30)
            base_education = np.random.uniform(7, 12)
            base_health_ratio = np.random.uniform(0.8, 2.5)
            
            for i in range(n_districts):
                # Add some correlation between variables and year trends
                year_effect = (year - 2018) * 0.1
                
                data.append({
                    'Tahun': year,
                    'Provinsi': province,
                    'Kabupaten': f'{province}_District_{i+1}',
                    'Kemiskinan (%)': max(0, base_poverty + np.random.normal(-year_effect, 3)),
                    'Pengeluaran Pangan (%)': np.random.uniform(35, 75),
                    'Tanpa Air Bersih (%)': max(0, np.random.uniform(5, 45) - year_effect),
                    'Lama Sekolah Perempuan (tahun)': base_education + np.random.normal(year_effect*0.5, 1),
                    'Rasio Tenaga Kesehatan': base_health_ratio + np.random.normal(year_effect*0.1, 0.3),
                    'Angka Harapan Hidup (tahun)': np.random.uniform(65, 78) + year_effect*0.2,
                    'Komposit': np.random.randint(1, 7)
                })
    
    df = pd.DataFrame(data)
    print(f"✅ Created dataset with {len(df)} records")
    print(f"   • Years: {df['Tahun'].min()} - {df['Tahun'].max()}")
    print(f"   • Provinces: {df['Provinsi'].nunique()}")
    print(f"   • Districts: {df['Kabupaten'].nunique()}")
    
    return df

def run_basic_analysis():
    """Run basic analysis example"""
    print("\n🚀 RUNNING BASIC ANALYSIS EXAMPLE")
    print("=" * 50)
    
    # 1. Create sample data
    df = create_sample_data()
    
    # 2. Initialize forecaster
    print("\n⚙️ Initializing forecaster...")
    config = FoodSecurityConfig()
    forecaster = FoodSecurityForecaster(config)
    
    # 3. Run full analysis
    print("\n🎯 Running comprehensive analysis...")
    forecaster.run_full_analysis(df)
    
    # 4. Display results
    print("\n📊 ANALYSIS RESULTS")
    print("=" * 30)
    
    # Model performance
    cv_results = forecaster.cv_results
    print(f"Model Performance:")
    print(f"  • Mean R²: {cv_results['r2'].mean():.3f} ± {cv_results['r2'].std():.3f}")
    print(f"  • Mean RMSE: {cv_results['rmse'].mean():.3f}")
    print(f"  • CV Folds: {len(cv_results)}")
    
    # Feature importance
    feature_importance = forecaster.feature_importance
    print(f"\nTop 3 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(3).iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.3f}")
    
    # Scenario predictions
    if forecaster.scenario_predictions is not None:
        scenario_predictions = forecaster.scenario_predictions
        print(f"\nScenario Predictions (2025):")
        for scenario in scenario_predictions['Scenario'].unique():
            scenario_data = scenario_predictions[scenario_predictions['Scenario'] == scenario]
            avg_pred = scenario_data['Predicted_Komposit'].mean()
            print(f"  • {scenario}: {avg_pred:.2f}")
    
    # Risk assessment
    if forecaster.risk_assessment is not None:
        risk_assessment = forecaster.risk_assessment
        status_quo_risk = risk_assessment[risk_assessment['Scenario'] == 'Status Quo']
        risk_dist = status_quo_risk['Risk_Level'].value_counts()
        print(f"\nRisk Assessment:")
        for risk_level, count in risk_dist.items():
            print(f"  • {risk_level}: {count} provinces")
    
    return forecaster

def run_advanced_analysis():
    """Run advanced analysis with custom configuration"""
    print("\n🔬 RUNNING ADVANCED ANALYSIS EXAMPLE")
    print("=" * 50)
    
    # 1. Create custom configuration
    config = FoodSecurityConfig()
    
    # Modify parameters for faster execution
    config.PARAM_GRID = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }
    
    print("⚙️ Using custom configuration:")
    print(f"  • Reduced parameter grid for faster execution")
    print(f"  • Features: {len(config.PREDICTOR_VARIABLES)}")
    
    # 2. Create data and run analysis
    df = create_sample_data()
    forecaster = FoodSecurityForecaster(config)
    forecaster.run_full_analysis(df)
    
    # 3. Advanced analysis
    print("\n📈 Advanced Results Analysis:")
    
    # Model stability analysis
    cv_results = forecaster.cv_results
    r2_cv = cv_results['r2'].std() / cv_results['r2'].mean()
    print(f"Model Stability (CV): {r2_cv:.3f} (lower is better)")
    
    # Feature stability across CV folds
    print(f"\nFeature Importance Analysis:")
    print(forecaster.feature_importance.round(3))
    
    # Best and worst performing provinces
    if forecaster.scenario_predictions is not None:
        moderate_scenario = forecaster.scenario_predictions[
            forecaster.scenario_predictions['Scenario'] == 'Moderate Improvement'
        ]
        
        best_provinces = moderate_scenario.nlargest(3, 'Predicted_Komposit')
        worst_provinces = moderate_scenario.nsmallest(3, 'Predicted_Komposit')
        
        print(f"\nBest Performing Provinces (Moderate Scenario):")
        for _, row in best_provinces.iterrows():
            print(f"  • {row['Provinsi']}: {row['Predicted_Komposit']:.2f}")
        
        print(f"\nWorst Performing Provinces (Moderate Scenario):")
        for _, row in worst_provinces.iterrows():
            print(f"  • {row['Provinsi']}: {row['Predicted_Komposit']:.2f}")
    
    return forecaster

def demonstrate_visualizations():
    """Demonstrate visualization capabilities"""
    print("\n📊 DEMONSTRATING VISUALIZATIONS")
    print("=" * 40)
    
    # Create data and run analysis
    df = create_sample_data()
    config = FoodSecurityConfig()
    forecaster = FoodSecurityForecaster(config)
    forecaster.run_full_analysis(df)
    
    # Initialize visualizer
    visualizer = FoodSecurityVisualizer(config)
    
    print("Creating visualizations...")
    
    # 1. Data overview
    fig_overview = visualizer.plot_data_overview(df)
    print("✅ Data overview plot created")
    
    # 2. Model performance
    fig_performance = visualizer.plot_model_performance(forecaster.cv_results)
    print("✅ Model performance plot created")
    
    # 3. Feature importance
    fig_importance = visualizer.plot_feature_importance(forecaster.feature_importance)
    print("✅ Feature importance plot created")
    
    # 4. Scenario comparison
    if forecaster.scenario_predictions is not None:
        fig_scenarios = visualizer.plot_scenario_comparison(forecaster.scenario_predictions)
        print("✅ Scenario comparison plot created")
    
    # 5. Time series trends
    fig_trends = visualizer.plot_time_series_trends(df)
    print("✅ Time series trends plot created")
    
    print(f"\n📊 All visualizations created successfully!")
    print(f"   In a Jupyter notebook, you can display them with:")
    print(f"   fig_overview.show()")
    
    return visualizer

def save_results_example(forecaster):
    """Example of saving analysis results"""
    print("\n💾 SAVING RESULTS EXAMPLE")
    print("=" * 30)
    
    # Save model performance results
    if forecaster.cv_results is not None:
        forecaster.cv_results.to_csv('model_performance_results.csv', index=False)
        print("✅ Model performance saved to: model_performance_results.csv")
    
    # Save feature importance
    if forecaster.feature_importance is not None:
        forecaster.feature_importance.to_csv('feature_importance.csv', index=False)
        print("✅ Feature importance saved to: feature_importance.csv")
    
    # Save scenario predictions
    if forecaster.scenario_predictions is not None:
        forecaster.scenario_predictions.to_csv('scenario_predictions_2025.csv', index=False)
        print("✅ Scenario predictions saved to: scenario_predictions_2025.csv")
    
    # Save risk assessment
    if forecaster.risk_assessment is not None:
        status_quo_risk = forecaster.risk_assessment[
            forecaster.risk_assessment['Scenario'] == 'Status Quo'
        ]
        status_quo_risk.to_csv('risk_assessment.csv', index=False)
        print("✅ Risk assessment saved to: risk_assessment.csv")

def main():
    """Main example execution"""
    print("🌾 FOOD SECURITY FORECASTING - EXAMPLE USAGE")
    print("=" * 60)
    
    try:
        # 1. Basic analysis
        forecaster_basic = run_basic_analysis()
        
        # 2. Advanced analysis
        forecaster_advanced = run_advanced_analysis()
        
        # 3. Visualizations
        visualizer = demonstrate_visualizations()
        
        # 4. Save results
        save_results_example(forecaster_advanced)
        
        print(f"\n🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 40)
        print("📋 Summary:")
        print("  ✅ Basic analysis completed")
        print("  ✅ Advanced analysis completed") 
        print("  ✅ Visualizations created")
        print("  ✅ Results saved to CSV files")
        print(f"\n🚀 Next steps:")
        print("  • Run 'streamlit run streamlit_dashboard.py' for interactive dashboard")
        print("  • Modify configurations in FoodSecurityConfig for your needs")
        print("  • Add your own data following the required format")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()