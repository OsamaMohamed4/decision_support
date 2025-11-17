import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests
import json
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", message="The keyword arguments have been deprecated")

st.set_page_config(
    page_title="Smart Agriculture Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f0f8f5;}
    .stAlert {border-radius: 10px;}
    h1 {color: #2d6a4f; text-align: center;}
    .case-study-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-right: 5px solid #4CAF50;
    }
    .cost-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_historical_data(n_samples=500):
    np.random.seed(42)
    
    data = {
        'temperature': np.random.normal(25, 7, n_samples),
        'humidity': np.random.normal(60, 15, n_samples),
        'rainfall': np.random.exponential(5, n_samples),
        'soil_moisture': np.random.normal(45, 10, n_samples),
        'ph_level': np.random.normal(6.5, 0.8, n_samples),
        'nitrogen': np.random.normal(40, 10, n_samples),
        'phosphorus': np.random.normal(35, 8, n_samples),
        'potassium': np.random.normal(30, 7, n_samples),
        'crop_type': np.random.choice(['Tomato', 'Cucumber', 'Wheat', 'Corn', 'Lettuce'], n_samples),
        'soil_type': np.random.choice(['Clay', 'Sandy', 'Loam'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    df['yield'] = (
        (df['temperature'].clip(15, 35) / 35) * 30 +
        (df['humidity'].clip(30, 80) / 80) * 25 +
        (df['soil_moisture'].clip(20, 70) / 70) * 25 +
        (df['nitrogen'].clip(20, 60) / 60) * 10 +
        (df['phosphorus'].clip(20, 50) / 50) * 5 +
        (df['potassium'].clip(15, 45) / 45) * 5 +
        np.random.normal(0, 5, n_samples)
    ).clip(0, 100)
    
    df['water_need'] = (
        df['temperature'] * 0.5 +
        (100 - df['humidity']) * 0.3 +
        df['soil_moisture'].apply(lambda x: 30 if x < 30 else 20 if x < 50 else 15) +
        np.random.normal(0, 3, n_samples)
    ).clip(10, 50)
    
    return df

@st.cache_resource
def train_ml_models():
    df = generate_historical_data(500)
    
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    
    df['crop_encoded'] = le_crop.fit_transform(df['crop_type'])
    df['soil_encoded'] = le_soil.fit_transform(df['soil_type'])
    
    features = ['temperature', 'humidity', 'rainfall', 'soil_moisture', 
                'ph_level', 'nitrogen', 'phosphorus', 'potassium', 
                'crop_encoded', 'soil_encoded']
    
    X = df[features]
    y_yield = df['yield']
    y_water = df['water_need']
    
    model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
    model_yield.fit(X, y_yield)
    
    model_water = RandomForestRegressor(n_estimators=100, random_state=42)
    model_water.fit(X, y_water)
    
    return model_yield, model_water, le_crop, le_soil

def get_fallback_weather():
    return {
        'temperature': 25.0,
        'humidity': 60.0,
        'rainfall': 0.0,
        'wind_speed': 10.0,
        'description': 'Moderate',
        'pressure': 1013.0,
        'visibility': 10.0
    }

def get_real_weather(city="Cairo", api_key=None):
    if not api_key or api_key.strip() == "":
        st.warning("No API Key entered - Using simulated data")
        weather_data = {
            'temperature': np.random.normal(25, 5),
            'humidity': np.random.normal(60, 10),
            'rainfall': max(0, np.random.exponential(3) if np.random.random() > 0.7 else 0),
            'wind_speed': np.random.uniform(5, 25),
            'description': np.random.choice(['Clear', 'Partly Cloudy', 'Rainy', 'Sunny']),
            'pressure': np.random.uniform(1010, 1020),
            'visibility': np.random.uniform(8, 10)
        }
        return weather_data
    
    try:
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        params = {
            'q': city,
            'appid': api_key.strip(),
            'units': 'metric',
            'lang': 'en'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
     
        if response.status_code == 200:
            data = response.json()
            
            weather_data = {
                'temperature': float(data['main']['temp']),
                'humidity': float(data['main']['humidity']),
                'rainfall': float(data.get('rain', {}).get('1h', 0)),
                'wind_speed': float(data['wind']['speed']) * 3.6,
                'description': data['weather'][0]['description'] if data.get('weather') else 'N/A',
                'pressure': float(data['main']['pressure']),
                'visibility': float(data.get('visibility', 10000)) / 1000
            }
            
            st.success(f"Real weather data fetched successfully from {city}!")
            return weather_data
            
        elif response.status_code == 401:
            st.error("Invalid API Key! Please check your key.")
            st.info("Get a free key from: https://openweathermap.org/api")
            return get_fallback_weather()
            
        elif response.status_code == 404:
            st.error(f"City '{city}' not found! Try another city name.")
            return get_fallback_weather()
            
        else:
            st.error(f"API connection error: {response.status_code}")
            return get_fallback_weather()
            
    except requests.exceptions.Timeout:
        st.error("Connection timeout - Check your internet")
        return get_fallback_weather()
        
    except requests.exceptions.ConnectionError:
        st.error("Internet connection error")
        return get_fallback_weather()
        
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return get_fallback_weather()

def generate_weather_forecast(days=7):
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    base_temp = 25
    temps = [base_temp + np.random.normal(0, 5) + np.sin(i/7*2*np.pi)*3 for i in range(days)]
    humidity = [60 + np.random.normal(0, 10) - i*2 for i in range(days)]
    rainfall = [max(0, np.random.exponential(3) if np.random.random() > 0.6 else 0) for _ in range(days)]
    
    forecast_df = pd.DataFrame({
        'date': dates,
        'temperature': np.round(temps, 1),
        'humidity': np.round(np.clip(humidity, 30, 90), 1),
        'rainfall': np.round(rainfall, 1)
    })
    
    return forecast_df

CROPS_INFO = {
    'Tomato': {
        'growth_days': 80, 
        'min_temp': 18, 
        'max_temp': 30, 
        'ideal_ph': 6.5,
        'cost_per_kg': 2.5,
        'price_per_kg': 5.0,
        'yield_per_m2': 8
    },
    'Cucumber': {
        'growth_days': 60, 
        'min_temp': 20, 
        'max_temp': 32, 
        'ideal_ph': 6.0,
        'cost_per_kg': 2.0,
        'price_per_kg': 4.5,
        'yield_per_m2': 10
    },
    'Wheat': {
        'growth_days': 120, 
        'min_temp': 15, 
        'max_temp': 25, 
        'ideal_ph': 6.5,
        'cost_per_kg': 1.5,
        'price_per_kg': 3.0,
        'yield_per_m2': 5
    },
    'Corn': {
        'growth_days': 90, 
        'min_temp': 18, 
        'max_temp': 35, 
        'ideal_ph': 6.0,
        'cost_per_kg': 1.8,
        'price_per_kg': 3.5,
        'yield_per_m2': 6
    },
    'Lettuce': {
        'growth_days': 45, 
        'min_temp': 12, 
        'max_temp': 20, 
        'ideal_ph': 6.5,
        'cost_per_kg': 3.0,
        'price_per_kg': 6.0,
        'yield_per_m2': 4
    }
}

SOIL_INFO = {
    'Clay': {'retention': 0.8, 'drainage': 0.3, 'nutrients': 0.9},
    'Sandy': {'retention': 0.3, 'drainage': 0.9, 'nutrients': 0.4},
    'Loam': {'retention': 0.6, 'drainage': 0.6, 'nutrients': 0.7}
}

CASE_STUDIES = {
    'Case 1: Small Tomato Farm': {
        'crop': 'Tomato',
        'soil': 'Clay',
        'area': 500,
        'soil_moisture': 55,
        'ph': 6.5,
        'nitrogen': 45,
        'phosphorus': 38,
        'potassium': 32,
        'water': 2000,
        'description': 'Small farm in moderate climate, fertile soil, good water resources'
    },
    'Case 2: Commercial Cucumber Project': {
        'crop': 'Cucumber',
        'soil': 'Loam',
        'area': 1000,
        'soil_moisture': 48,
        'ph': 6.2,
        'nitrogen': 42,
        'phosphorus': 35,
        'potassium': 28,
        'water': 3500,
        'description': 'Medium commercial project, balanced soil, aiming for maximum productivity'
    },
    'Case 3: Wheat Farm in Harsh Environment': {
        'crop': 'Wheat',
        'soil': 'Sandy',
        'area': 2000,
        'soil_moisture': 35,
        'ph': 7.0,
        'nitrogen': 30,
        'phosphorus': 25,
        'potassium': 22,
        'water': 2500,
        'description': 'Large farm in desert environment, challenges with water and nutrient deficiency'
    }
}

def calculate_costs(crop, area, predicted_yield, predicted_water):
    crop_info = CROPS_INFO[crop]
    
    seeds_cost = area * 0.5
    water_cost = predicted_water * 7 * (crop_info['growth_days'] / 7) * 0.02
    fertilizer_cost = area * 2
    labor_cost = area * 1.5
    other_costs = area * 0.8
    
    total_cost = seeds_cost + water_cost + fertilizer_cost + labor_cost + other_costs
    
    expected_yield_kg = area * crop_info['yield_per_m2'] * (predicted_yield / 100)
    revenue = expected_yield_kg * crop_info['price_per_kg']
    
    profit = revenue - total_cost
    roi = (profit / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'seeds_cost': seeds_cost,
        'water_cost': water_cost,
        'fertilizer_cost': fertilizer_cost,
        'labor_cost': labor_cost,
        'other_costs': other_costs,
        'total_cost': total_cost,
        'expected_yield_kg': expected_yield_kg,
        'revenue': revenue,
        'profit': profit,
        'roi': roi
    }

model_yield, model_water, le_crop, le_soil = train_ml_models()

st.markdown("<h1>Smart Agriculture Decision Support System - AI Powered</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Smart decisions powered by AI and real data</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Home", "Case Studies", "Financial Analysis"])

with tab1:
    with st.sidebar:
        st.header("Project Settings")
        
        st.subheader("Weather Settings")
        api_key = st.text_input(
            "OpenWeatherMap API Key",
            type="password",
            placeholder="Enter key here...",
            help="Get a free key from openweathermap.org/api"
        )
        
        if not api_key:
            st.info("To get a free API Key:\n1. Register at openweathermap.org\n2. Go to API Keys\n3. Copy and paste the key here")
        
        city = st.text_input("City", "Cairo", help="Example: Riyadh, Dubai, Jeddah")
        
        if st.button("Refresh Weather Data", use_container_width=True):
            st.rerun()
        
        st.divider()
        
        selected_crop = st.selectbox(
            "Select Crop",
            list(CROPS_INFO.keys())
        )
        
        selected_soil = st.selectbox(
            "Soil Type",
            list(SOIL_INFO.keys())
        )
        
        area = st.number_input("Area (square meters)", 100, 10000, 500, 50)
        
        st.divider()
        st.subheader("Soil Measurements")
        
        soil_moisture = st.slider("Soil Moisture (%)", 10, 80, 45)
        ph_level = st.slider("pH Level", 4.0, 8.0, 6.5, 0.1)
        nitrogen = st.slider("Nitrogen Level", 10, 70, 40)
        phosphorus = st.slider("Phosphorus Level", 10, 60, 35)
        potassium = st.slider("Potassium Level", 10, 50, 30)
        
        st.divider()
        water_available = st.number_input("Available Water (liters/day)", 100, 5000, 1000, 50)

    weather_now = get_real_weather(city, api_key)
    
    input_features = np.array([[
        weather_now['temperature'],
        weather_now['humidity'],
        weather_now['rainfall'],
        soil_moisture,
        ph_level,
        nitrogen,
        phosphorus,
        potassium,
        le_crop.transform([selected_crop])[0],
        le_soil.transform([selected_soil])[0]
    ]])

    predicted_yield = model_yield.predict(input_features)[0]
    predicted_water = model_water.predict(input_features)[0]
    costs = calculate_costs(selected_crop, area, predicted_yield, predicted_water)

    st.markdown("### Current Weather Conditions")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Temperature", f"{weather_now['temperature']:.1f}°C")
    with col2:
        st.metric("Humidity", f"{weather_now['humidity']:.1f}%")
    with col3:
        st.metric("Rainfall", f"{weather_now['rainfall']:.1f} mm")
    with col4:
        st.metric("Wind", f"{weather_now['wind_speed']:.1f} km/h")
    with col5:
        st.metric("Status", weather_now['description'])

    st.divider()

    st.markdown("### AI Predictions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center;'>
            <h2 style='color: white; margin:0;'>{predicted_yield:.1f}%</h2>
            <p style='margin: 5px 0 0 0;'>Expected Productivity</p>
            <small>Compared to average</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center;'>
            <h2 style='color: white; margin:0;'>{predicted_water:.1f} L</h2>
            <p style='margin: 5px 0 0 0;'>Daily Water Need</p>
            <small>Based on current conditions</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        harvest_date = datetime.now() + timedelta(days=CROPS_INFO[selected_crop]['growth_days'])
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center;'>
            <h2 style='color: white; margin:0; font-size: 1.3em;'>{harvest_date.strftime('%d/%m/%Y')}</h2>
            <p style='margin: 5px 0 0 0;'>Expected Harvest Date</p>
            <small>{CROPS_INFO[selected_crop]['growth_days']} days</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Smart Alerts")
    alerts = []
    crop_info = CROPS_INFO[selected_crop]

    if weather_now['temperature'] > crop_info['max_temp']:
        alerts.append(('warning', f"Temperature too high ({weather_now['temperature']:.1f}°C)"))
    elif weather_now['temperature'] < crop_info['min_temp']:
        alerts.append(('error', f"Temperature too low ({weather_now['temperature']:.1f}°C)"))

    if weather_now['rainfall'] > 10:
        alerts.append(('info', f"Heavy rain - Reduce irrigation to {predicted_water*0.5:.1f} L"))

    if soil_moisture < 30:
        alerts.append(('warning', "Low soil moisture - Increase irrigation"))

    if abs(ph_level - crop_info['ideal_ph']) > 1:
        alerts.append(('warning', f"Soil pH not ideal - Required: {crop_info['ideal_ph']}"))

    if water_available < predicted_water * 7:
        alerts.append(('error', "Available water insufficient for next week"))

    if predicted_yield < 50:
        alerts.append(('error', "Conditions not suitable - Recommend postponing planting"))

    if alerts:
        for alert_type, message in alerts:
            if alert_type == 'error':
                st.error(message)
            elif alert_type == 'warning':
                st.warning(message)
            else:
                st.info(message)
    else:
        st.success("All conditions are ideal for planting!")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Weather Forecast (7 Days)")
        forecast = generate_weather_forecast(7)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['temperature'],
                                mode='lines+markers', name='Temperature',
                                line=dict(color='#ff6b6b', width=3)))
        fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['humidity'],
                                mode='lines+markers', name='Humidity',
                                line=dict(color='#4ecdc4', width=3)))
        
        fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Value",
                         hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Influencing Factors Analysis")
        
        factors = pd.DataFrame({
            'Factor': ['Temperature', 'Humidity', 'Nutrients', 'Soil Moisture', 'Soil Type'],
            'Impact': [
                min(100, (weather_now['temperature'] / crop_info['max_temp']) * 100),
                min(100, weather_now['humidity']),
                min(100, (nitrogen + phosphorus + potassium) / 3 * 1.2),
                soil_moisture,
                SOIL_INFO[selected_soil]['retention'] * 100
            ]
        })
        
        fig = go.Figure(go.Bar(x=factors['Impact'], y=factors['Factor'], orientation='h',
                              marker=dict(color=factors['Impact'], colorscale='Viridis', showscale=True)))
        fig.update_layout(height=300, xaxis_title="Suitability (%)", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### Weekly Irrigation Schedule (AI Optimized)")

    weekly_schedule = []
    for i in range(7):
        day_temp = forecast.iloc[i]['temperature']
        day_rain = forecast.iloc[i]['rainfall']
        
        adjusted_water = predicted_water * (1 + (day_temp - 25) / 50)
        if day_rain > 5:
            adjusted_water *= 0.5
        
        morning = adjusted_water * 0.6
        evening = adjusted_water * 0.4
        
        weekly_schedule.append({
            'Day': forecast.iloc[i]['date'].strftime('%A'),
            'Date': forecast.iloc[i]['date'].strftime('%d/%m'),
            'Morning (L)': f"{morning:.1f}",
            'Evening (L)': f"{evening:.1f}",
            'Fertilization': 'Yes' if i % 3 == 0 else 'No',
            'Notes': 'Rainy' if day_rain > 5 else 'Dry'
        })

    schedule_df = pd.DataFrame(weekly_schedule)
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    total_weekly = sum([float(s['Morning (L)']) + float(s['Evening (L)']) for s in weekly_schedule])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Weekly Consumption", f"{total_weekly:.1f} L")
    with col2:
        efficiency = (1 - SOIL_INFO[selected_soil]['drainage']) * 100
        st.metric("Water Use Efficiency", f"{efficiency:.0f}%")
    with col3:
        savings = (water_available * 7 - total_weekly) / (water_available * 7) * 100 if water_available * 7 > 0 else 0
        st.metric("Expected Savings", f"{max(0, savings):.1f}%")

with tab2:
    st.markdown("## Applied Case Studies")
    st.markdown("Analysis of 3 different realistic scenarios")
    
    for case_name, case_data in CASE_STUDIES.items():
        with st.expander(f"{case_name}", expanded=False):
            st.markdown(f"**Description:** {case_data['description']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Crop:** {case_data['crop']}")
                st.write(f"**Soil:** {case_data['soil']}")
                st.write(f"**Area:** {case_data['area']} m²")
            with col2:
                st.write(f"**Soil Moisture:** {case_data['soil_moisture']}%")
                st.write(f"**pH:** {case_data['ph']}")
                st.write(f"**Nitrogen:** {case_data['nitrogen']}")
            with col3:
                st.write(f"**Phosphorus:** {case_data['phosphorus']}")
                st.write(f"**Potassium:** {case_data['potassium']}")
                st.write(f"**Available Water:** {case_data['water']} L/day")
            
            case_input = np.array([[
                25, 60, 0,
                case_data['soil_moisture'],
                case_data['ph'],
                case_data['nitrogen'],
                case_data['phosphorus'],
                case_data['potassium'],
                le_crop.transform([case_data['crop']])[0],
                le_soil.transform([case_data['soil']])[0]
            ]])
            
            case_yield = model_yield.predict(case_input)[0]
            case_water = model_water.predict(case_input)[0]
            case_costs = calculate_costs(case_data['crop'], case_data['area'], case_yield, case_water)
            
            st.divider()
            st.markdown("### Analysis Results:")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Productivity", f"{case_yield:.1f}%")
            with col2:
                st.metric("Daily Irrigation", f"{case_water:.1f} L")
            with col3:
                st.metric("Expected Profit", f"{case_costs['profit']:.0f} SAR")
            with col4:
                st.metric("ROI", f"{case_costs['roi']:.1f}%")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Costs', 'Revenue', 'Profit'],
                y=[case_costs['total_cost'], case_costs['revenue'], case_costs['profit']],
                marker_color=['#e74c3c', '#3498db', '#2ecc71'],
                text=[f"{case_costs['total_cost']:.0f}", 
                      f"{case_costs['revenue']:.0f}", 
                      f"{case_costs['profit']:.0f}"],
                textposition='auto'
            ))
            fig.update_layout(height=300, title="Financial Analysis", template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## Comprehensive Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cost Structure")
        
        costs_data = pd.DataFrame({
            'Item': ['Seeds', 'Water', 'Fertilizers', 'Labor', 'Other'],
            'Cost': [
                costs['seeds_cost'],
                costs['water_cost'],
                costs['fertilizer_cost'],
                costs['labor_cost'],
                costs['other_costs']
            ]
        })
        
        fig = go.Figure(data=[go.Pie(
            labels=costs_data['Item'],
            values=costs_data['Cost'],
            hole=.4,
            marker_colors=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        )])
        fig.update_layout(height=350, title="Cost Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(costs_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Profitability Analysis")
        
        profit_data = pd.DataFrame({
            'Indicator': ['Total Costs', 'Expected Revenue', 'Net Profit'],
            'Value (SAR)': [costs['total_cost'], costs['revenue'], costs['profit']]
        })
        
        fig = go.Figure(data=[go.Bar(
            x=profit_data['Indicator'],
            y=profit_data['Value (SAR)'],
            marker_color=['#e74c3c', '#3498db', '#2ecc71'],
            text=profit_data['Value (SAR)'].round(2),
            textposition='auto'
        )])
        fig.update_layout(height=350, title="Financial Indicators", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(profit_data, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.markdown("### Key Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='cost-card'>
            <h2 style='color: white; margin:0;'>{costs['expected_yield_kg']:.1f} kg</h2>
            <p style='margin: 5px 0 0 0;'>Expected Production</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='cost-card' style='background: linear-gradient(135deg, #  f093fb 0%, #f5576c 100%);'>
            <h2 style='color: white; margin:0;'>{costs['total_cost']:.0f} SAR</h2>
            <p style='margin: 5px 0 0 0;'>Total Costs</p>
        </div>
        """, unsafe_allow_html=True)


    with col3:
        st.markdown(f"""
        <div class='cost-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
            <h2 style='color: white; margin:0;'>{costs['revenue']:.0f} SAR</h2>
            <p style='margin: 5px 0 0 0;'>Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        profit_color = '#2ecc71' if costs['profit'] > 0 else '#e74c3c'
        st.markdown(f"""
        <div class='cost-card' style='background: linear-gradient(135deg, {profit_color} 0%, {profit_color} 100%);'>
            <h2 style='color: white; margin:0;'>{costs['profit']:.0f} SAR</h2>
            <p style='margin: 5px 0 0 0;'>Net Profit</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### Profitability Comparison Between Crops")
    
    comparison_data = []
    for crop_name in CROPS_INFO.keys():
        test_input = input_features.copy()
        test_input[0][8] = le_crop.transform([crop_name])[0]
        
        test_yield = model_yield.predict(test_input)[0]
        test_costs = calculate_costs(crop_name, area, test_yield, predicted_water)
        
        comparison_data.append({
            'Crop': crop_name,
            'Productivity': test_yield,
            'Costs': test_costs['total_cost'],
            'Revenue': test_costs['revenue'],
            'Profit': test_costs['profit'],
            'ROI': test_costs['roi']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Profit', x=comparison_df['Crop'], 
                         y=comparison_df['Profit'], marker_color='#2ecc71'))
    fig.add_trace(go.Scatter(name='ROI %', x=comparison_df['Crop'], 
                             y=comparison_df['ROI'], mode='lines+markers',
                             yaxis='y2', marker_color='#e74c3c', line=dict(width=3)))
    
    fig.update_layout(
        height=400,
        yaxis=dict(title='Profit (SAR)'),
        yaxis2=dict(title='ROI (%)', overlaying='y', side='right'),
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Smart Agriculture Decision Support System</strong></p>
    <p>Powered by Artificial Intelligence | Management Information Systems Department | College of Business Administration</p>
    <p style='font-size: 12px; margin-top: 10px;'>
        AI Models: Random Forest | Data: 500+ Records | Real-time Weather API
    </p>
</div>
""", unsafe_allow_html=True)      