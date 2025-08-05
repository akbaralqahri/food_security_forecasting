# =============================================================================
# DASHBOARD COMPONENTS - FOOD SECURITY FORECASTING
# Reusable UI components for Streamlit dashboard
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

def create_alert_banner(message, alert_type="info", dismissible=False):
    """Create alert banners for notifications"""
    
    alert_configs = {
        "info": {"bg": "#d1ecf1", "border": "#17a2b8", "text": "#0c5460", "icon": "ℹ️"},
        "success": {"bg": "#d4edda", "border": "#28a745", "text": "#155724", "icon": "✅"},
        "warning": {"bg": "#fff3cd", "border": "#ffc107", "text": "#856404", "icon": "⚠️"},
        "danger": {"bg": "#f8d7da", "border": "#dc3545", "text": "#721c24", "icon": "🚨"}
    }
    
    config = alert_configs.get(alert_type, alert_configs["info"])
    
    return f"""
    <div style="
        background-color: {config['bg']};
        border: 1px solid {config['border']};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: {config['text']};
    ">
        <strong>{config['icon']} {message}</strong>
    </div>
    """

def create_progress_card(title, current_value, target_value, unit="", description=""):
    """Create progress tracking cards"""
    
    if target_value > 0:
        progress_pct = min((current_value / target_value) * 100, 100)
    else:
        progress_pct = 0
    
    # Determine color based on progress
    if progress_pct >= 80:
        color = "#28a745"  # Green
    elif progress_pct >= 60:
        color = "#ffc107"  # Yellow
    else:
        color = "#dc3545"  # Red
    
    return f"""
    <div style="
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <h4 style="color: #2c3e50; margin-bottom: 0.5rem;">{title}</h4>
        <div style="
            background-color: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 0.5rem 0;
            overflow: hidden;
        ">
            <div style="
                background-color: {color};
                height: 100%;
                width: {progress_pct}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #495057;">
            <span>{current_value} {unit}</span>
            <span>{progress_pct:.1f}%</span>
            <span>{target_value} {unit}</span>
        </div>
        {f'<p style="color: #6c757d; font-size: 0.8rem; margin-top: 0.5rem;">{description}</p>' if description else ''}
    </div>
    """