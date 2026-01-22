# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TimeGPT imports
from nixtla import NixtlaClient

# Set page configuration
st.set_page_config(
    page_title="Fuel Sales Forecasting with TimeGPT",
    page_icon="⛽",
    layout="wide"
)

# Title and description
st.title("⛽ Fuel Sales Forecasting with TimeGPT")
st.markdown("""
This app uses TimeGPT to forecast total network fuel sales across all stations.
The dataset includes sales data for AGO, PMS, Lubricant, Diesel, and LPG across multiple stations.
""")

# Initialize TimeGPT client (you'll need to get an API key)
@st.cache_resource
def init_timegpt():
    try:
        # You'll need to get an API key from https://docs.nixtla.io/
        # For demo purposes, we'll use a placeholder
        # api_key = st.secrets["TIMEGPT_API_KEY"]  # For production, use secrets
        api_key = "nixak-nkiz6AHVez8zFO1d6EMy1lGWAHTBN5QYAg5cKVxmHTXV6SImSU6TmrVV04d0ytTuwlgXiZmbLl14tUl4"  # Replace with your actual API key
        
        nixtla_client = NixtlaClient(api_key=api_key)
        return nixtla_client
    except Exception as e:
        st.error(f"Error initializing TimeGPT: {e}")
        return None

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Load the CSV data
        data = pd.read_csv('timac_fuel_data_500_enhanced.csv')
        
        # Convert Date column to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Create total network sales column
        # Calculate total sales in liters/kg and convert to revenue
        data['Total_Sales_Volume'] = (
            data['AGO_Sales (L)'] + 
            data['PMS_Sales (L)'] + 
            data['Lubricant_Sales (L)'] + 
            data['Diesel_Sales (L)'] + 
            data['LPG_Sales (kg)']
        )
        
        # Calculate total revenue from individual products
        data['Total_Revenue_Calculated'] = (
            data['AGO_Sales (L)'] * data['AGO_Price'] +
            data['PMS_Sales (L)'] * data['PMS_Price'] +
            data['Lubricant_Sales (L)'] * data['Lubricant_Price'] +
            data['Diesel_Sales (L)'] * data['Diesel_Price'] +
            data['LPG_Sales (kg)'] * data['LPG_Price']
        )
        
        # Group by date to get daily totals across all stations
        daily_data = data.groupby('Date').agg({
            'Total_Sales_Volume': 'sum',
            'Total_Revenue_Calculated': 'sum',
            'AGO_Sales (L)': 'sum',
            'PMS_Sales (L)': 'sum',
            'Lubricant_Sales (L)': 'sum',
            'Diesel_Sales (L)': 'sum',
            'LPG_Sales (kg)': 'sum'
        }).reset_index()
        
        # Sort by date
        daily_data = daily_data.sort_values('Date')
        
        return data, daily_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Create time series DataFrame for TimeGPT
def prepare_timegpt_data(daily_data, target_column):
    """Prepare data for TimeGPT forecasting"""
    df_timegpt = daily_data[['Date', target_column]].copy()
    df_timegpt.columns = ['ds', 'y']
    df_timegpt = df_timegpt.sort_values('ds')
    return df_timegpt

# Forecast with TimeGPT
def forecast_with_timegpt(nixtla_client, df_timegpt, horizon, freq='D', level=None):
    """Generate forecast using TimeGPT"""
    try:
        # For demo without API key, return mock forecast
        if nixtla_client is None:
            return create_mock_forecast(df_timegpt, horizon, freq)
        
        # Actual TimeGPT forecast
        if level:
            forecast_df = nixtla_client.forecast(
                df=df_timegpt,
                h=horizon,
                freq=freq,
                level=[level]
            )
        else:
            forecast_df = nixtla_client.forecast(
                df=df_timegpt,
                h=horizon,
                freq=freq
            )
        
        return forecast_df
    except Exception as e:
        st.error(f"Error with TimeGPT forecast: {e}")
        # Return mock forecast as fallback
        return create_mock_forecast(df_timegpt, horizon, freq)

def create_mock_forecast(df_timegpt, horizon, freq):
    """Create a mock forecast for demo purposes"""
    last_date = df_timegpt['ds'].max()
    
    if freq == 'D':
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
    elif freq == 'W':
        future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=horizon, freq='W-MON')
    else:
        future_dates = pd.date_range(start=last_date + timedelta(days=30), periods=horizon, freq='MS')
    
    # Simple trend-based forecast (for demo only)
    last_value = df_timegpt['y'].iloc[-1]
    trend = df_timegpt['y'].diff().mean()
    
    if pd.isna(trend):
        trend = 0
    
    forecast_values = [last_value + (i+1) * trend * 0.5 for i in range(horizon)]
    
    mock_forecast = pd.DataFrame({
        'ds': future_dates,
        'y': forecast_values,
        'TimeGPT': forecast_values
    })
    
    return mock_forecast

# Visualization functions
def plot_forecast(historical_df, forecast_df, title, y_label):
    """Plot historical data and forecast"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['ds'],
        y=historical_df['y'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['TimeGPT'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add confidence interval if available
    if 'TimeGPT-lo-90' in forecast_df.columns and 'TimeGPT-hi-90' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
            y=pd.concat([forecast_df['TimeGPT-hi-90'], forecast_df['TimeGPT-lo-90'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='90% Confidence Interval'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_product_sales(daily_data):
    """Plot individual product sales"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('AGO Sales', 'PMS Sales', 'Lubricant Sales', 
                       'Diesel Sales', 'LPG Sales', 'Total Revenue'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # AGO Sales
    fig.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['AGO_Sales (L)'], 
                  mode='lines', name='AGO', line=dict(color='blue')),
        row=1, col=1
    )
    
    # PMS Sales
    fig.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['PMS_Sales (L)'], 
                  mode='lines', name='PMS', line=dict(color='green')),
        row=1, col=2
    )
    
    # Lubricant Sales
    fig.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['Lubricant_Sales (L)'], 
                  mode='lines', name='Lubricant', line=dict(color='orange')),
        row=1, col=3
    )
    
    # Diesel Sales
    fig.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['Diesel_Sales (L)'], 
                  mode='lines', name='Diesel', line=dict(color='purple')),
        row=2, col=1
    )
    
    # LPG Sales
    fig.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['LPG_Sales (kg)'], 
                  mode='lines', name='LPG', line=dict(color='brown')),
        row=2, col=2
    )
    
    # Total Revenue
    fig.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['Total_Revenue_Calculated'], 
                  mode='lines', name='Total Revenue', line=dict(color='red')),
        row=2, col=3
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        template='plotly_white',
        title_text="Individual Product Sales Analysis"
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Liters", row=1, col=1)
    fig.update_yaxes(title_text="Liters", row=1, col=2)
    fig.update_yaxes(title_text="Liters", row=1, col=3)
    fig.update_yaxes(title_text="Liters", row=2, col=1)
    fig.update_yaxes(title_text="Kilograms", row=2, col=2)
    fig.update_yaxes(title_text="Revenue (₹)", row=2, col=3)
    
    return fig

# Main app
def main():
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Forecasting Parameters")
        
        # Forecasting horizon
        horizon = st.slider(
            "Forecast Horizon (days):",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Number of days to forecast into the future"
        )
        
        # Confidence level
        confidence_level = st.selectbox(
            "Confidence Interval:",
            [None, 80, 90, 95],
            format_func=lambda x: "None" if x is None else f"{x}%",
            help="Confidence interval for the forecast"
        )
        
        # Target column
        target_column = st.selectbox(
            "Target Metric:",
            ['Total_Sales_Volume', 'Total_Revenue_Calculated'],
            format_func=lambda x: "Total Sales Volume" if x == 'Total_Sales_Volume' else "Total Revenue"
        )
        
        # Frequency
        freq = st.selectbox(
            "Frequency:",
            ['D', 'W', 'M'],
            format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x]
        )
        
        # Load data button
        st.header("📊 Data Operations")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # API key input (for demo)
        st.header("🔑 TimeGPT API Key")
        api_key = st.text_input(
            "Enter your TimeGPT API key:",
            type="password",
            help="Get your API key from https://docs.nixtla.io/"
        )
        
        if api_key and api_key != "YOUR_API_KEY_HERE":
            st.success("API key loaded!")
    
    # Load data
    raw_data, daily_data = load_and_preprocess_data()
    
    if daily_data is None:
        st.error("Failed to load data. Please check your CSV file.")
        return
    
    # Display data summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Days of Data",
            f"{len(daily_data)}",
            help="Number of days in the dataset"
        )
    
    with col2:
        avg_daily_sales = daily_data['Total_Sales_Volume'].mean()
        st.metric(
            "Avg Daily Sales Volume",
            f"{avg_daily_sales:,.0f} L/kg",
            help="Average total sales per day"
        )
    
    with col3:
        avg_daily_revenue = daily_data['Total_Revenue_Calculated'].mean()
        st.metric(
            "Avg Daily Revenue",
            f"₹{avg_daily_revenue:,.0f}",
            help="Average revenue per day"
        )
    
    # Data preview tabs
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Data Analysis", "📋 Raw Data"])
    
    with tab1:
        st.header("TimeGPT Forecasting Results")
        
        # Prepare data for TimeGPT
        df_timegpt = prepare_timegpt_data(daily_data, target_column)
        
        # Initialize TimeGPT
        nixtla_client = init_timegpt()
        
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Generating forecast with TimeGPT..."):
                # Generate forecast
                forecast_df = forecast_with_timegpt(
                    nixtla_client,
                    df_timegpt,
                    horizon,
                    freq,
                    confidence_level
                )
                
                # Display forecast results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Forecast Results")
                    forecast_display = forecast_df.copy()
                    if target_column == 'Total_Sales_Volume':
                        forecast_display['TimeGPT'] = forecast_display['TimeGPT'].round(0)
                        y_label = "Sales Volume (L/kg)"
                        title = f"Total Network Sales Volume Forecast ({horizon} days)"
                    else:
                        forecast_display['TimeGPT'] = forecast_display['TimeGPT'].round(2)
                        y_label = "Revenue (₹)"
                        title = f"Total Network Revenue Forecast ({horizon} days)"
                    
                    st.dataframe(
                        forecast_display[['ds', 'TimeGPT']].rename(
                            columns={'ds': 'Date', 'TimeGPT': 'Forecast Value'}
                        ).style.format({
                            'Forecast Value': '{:,.0f}' if target_column == 'Total_Sales_Volume' else '₹{:,.2f}'
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    st.subheader("Forecast Summary")
                    
                    avg_forecast = forecast_df['TimeGPT'].mean()
                    max_forecast = forecast_df['TimeGPT'].max()
                    min_forecast = forecast_df['TimeGPT'].min()
                    
                    st.metric("Average Forecast", 
                             f"{avg_forecast:,.0f} {'L/kg' if target_column == 'Total_Sales_Volume' else '₹'}")
                    st.metric("Maximum Forecast", 
                             f"{max_forecast:,.0f} {'L/kg' if target_column == 'Total_Sales_Volume' else '₹'}")
                    st.metric("Minimum Forecast", 
                             f"{min_forecast:,.0f} {'L/kg' if target_column == 'Total_Sales_Volume' else '₹'}")
                
                # Plot forecast
                fig = plot_forecast(df_timegpt, forecast_df, title, y_label)
                st.plotly_chart(fig, use_container_width=True)
                
                # Download forecast
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast as CSV",
                    data=csv,
                    file_name=f"fuel_sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.header("Data Analysis")
        
        # Plot product sales breakdown
        fig = plot_product_sales(daily_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sales Volume Statistics (L/kg)**")
            volume_stats = daily_data[['Total_Sales_Volume', 'AGO_Sales (L)', 'PMS_Sales (L)', 
                                      'Lubricant_Sales (L)', 'Diesel_Sales (L)', 'LPG_Sales (kg)']].describe()
            st.dataframe(volume_stats.style.format('{:,.0f}'), use_container_width=True)
        
        with col2:
            st.write("**Revenue Statistics (₹)**")
            revenue_stats = pd.DataFrame({
                'Total Revenue': daily_data['Total_Revenue_Calculated'],
                'AGO Revenue': daily_data['AGO_Sales (L)'] * 750,
                'PMS Revenue': daily_data['PMS_Sales (L)'] * 650,
                'Lubricant Revenue': daily_data['Lubricant_Sales (L)'] * 2000,
                'Diesel Revenue': daily_data['Diesel_Sales (L)'] * 820,
                'LPG Revenue': daily_data['LPG_Sales (kg)'] * 850
            }).describe()
            st.dataframe(revenue_stats.style.format('₹{:,.0f}'), use_container_width=True)
    
    with tab3:
        st.header("Raw Data Preview")
        
        # Show raw data with filters
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Select Date Range:",
                value=[daily_data['Date'].min(), daily_data['Date'].max()],
                min_value=daily_data['Date'].min(),
                max_value=daily_data['Date'].max()
            )
        
        with col2:
            rows_to_show = st.slider("Rows to display:", 10, 100, 20)
        
        if len(date_range) == 2:
            filtered_data = daily_data[
                (daily_data['Date'] >= pd.Timestamp(date_range[0])) &
                (daily_data['Date'] <= pd.Timestamp(date_range[1]))
            ]
        else:
            filtered_data = daily_data
        
        st.dataframe(
            filtered_data.head(rows_to_show).style.format({
                'Total_Sales_Volume': '{:,.0f}',
                'Total_Revenue_Calculated': '₹{:,.0f}',
                'AGO_Sales (L)': '{:,.0f}',
                'PMS_Sales (L)': '{:,.0f}',
                'Lubricant_Sales (L)': '{:,.0f}',
                'Diesel_Sales (L)': '{:,.0f}',
                'LPG_Sales (kg)': '{:,.0f}'
            }),
            use_container_width=True
        )
        
        # Download full data
        csv = daily_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Daily Data",
            data=csv,
            file_name="daily_fuel_sales_summary.csv",
            mime="text/csv"
        )
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📚 How to Use This App:
    1. **Get a TimeGPT API key** from [Nixtla](https://docs.nixtla.io/)
    2. **Enter your API key** in the sidebar
    3. **Adjust forecasting parameters** (horizon, confidence level, etc.)
    4. **Click "Generate Forecast"** to get predictions
    5. **Explore different tabs** for analysis and raw data
    
    ### 🔍 About the Data:
    - **Total Sales Volume**: Sum of all product sales in liters/kilograms
    - **Total Revenue**: Calculated from individual product prices
    - **Data aggregated daily** across all stations
    """)

if __name__ == "__main__":
    main()