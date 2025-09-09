# src/streamlit_app.py
import streamlit as st
import pandas as pd
from serpapi import Client
from prophet import Prophet
import plotly.express as px
import time
import google.generativeai as genai
import os
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Trend-AI Marketing Dashboard", page_icon="🧠", layout="wide")

# --- Caching & Data Fetching Functions ---
@st.cache_data(ttl=3600)
def fetch_data_from_serpapi(api_key, keywords, timeframe, geo):
    # This function for live data is unchanged
    def _parse_date_string(date_str):
        if '–' in date_str:
            try:
                parts = date_str.split('–'); start_day_month = parts[0].strip(); year = parts[1].split(',')[-1].strip()
                return pd.to_datetime(f"{start_day_month}, {year}")
            except: return pd.to_datetime(date_str.split('–')[0].strip())
        else: return pd.to_datetime(date_str)
            
    params = {"engine": "google_trends", "q": ", ".join(keywords), "date": timeframe, "geo": geo, "api_key": api_key}
    try:
        client = Client(); results = client.search(params)
        all_data = {"interest_over_time": pd.DataFrame(), "interest_by_region": pd.DataFrame()}
        if 'interest_over_time' in results:
            timeline_data = results['interest_over_time']['timeline_data']
            dates = [_parse_date_string(item['date']) for item in timeline_data]
            data = {}
            for i, keyword in enumerate(keywords): data[keyword] = [item['values'][i].get('value', 0) for item in timeline_data]
            all_data["interest_over_time"] = pd.DataFrame(data, index=dates)
        if 'interest_by_region' in results:
            df_region = pd.DataFrame(results['interest_by_region']).set_index('geoName')
            if keywords and len(keywords) > 0:
                 renamed_col = df_region.columns[0]
                 df_region = df_region.rename(columns={renamed_col: 'Interest'})
            all_data["interest_by_region"] = df_region
        return all_data
    except Exception as e:
        st.error(f"An error occurred with the SerpApi request: {e}"); return None

@st.cache_data
def load_all_offline_data(scenario_config):
    """
    UPGRADED: Loads CSVs and cleans the column names to match our keyword list.
    """
    prefix = scenario_config["prefix"]
    keywords = scenario_config["keywords"] # Get the clean keywords for mapping
    all_data = {"interest_over_time": pd.DataFrame(), "interest_by_region": pd.DataFrame()}
    try:
        ot_path = f"data/{prefix}_over_time.csv"
        df_ot = pd.read_csv(ot_path, skiprows=2)
        df_ot.rename(columns={df_ot.columns[0]: 'Date'}, inplace=True)
        
        # --- THIS IS THE KEY FIX ---
        # Get the original messy column names from the CSV
        original_columns = df_ot.columns[1:] 
        # Create a mapping from the messy names to our clean keyword names
        column_map = dict(zip(original_columns, keywords))
        # Rename the columns in the DataFrame
        df_ot.rename(columns=column_map, inplace=True)
        # --- END OF FIX ---

        df_ot['Date'] = pd.to_datetime(df_ot['Date']); df_ot.set_index('Date', inplace=True)
        for col in df_ot.columns: df_ot[col] = pd.to_numeric(df_ot[col], errors='coerce')
        all_data['interest_over_time'] = df_ot
        
        r_path = f"data/{prefix}_by_region.csv"
        df_r = pd.read_csv(r_path, skiprows=1); df_r.rename(columns={df_r.columns[0]: 'Region', df_r.columns[1]: 'Interest'}, inplace=True); df_r['Interest'] = pd.to_numeric(df_r['Interest'], errors='coerce')
        all_data['interest_by_region'] = df_r.set_index('Region')
        return all_data
    except FileNotFoundError as e:
        st.error(f"Offline Data Error: Missing a primary CSV file: {e.filename}"); return None
    except Exception as e:
        st.error(f"Error loading offline data: {e}"); return None

# --- Main Application Logic ---
st.title("📈 Trend-AI: Marketing Intelligence Dashboard")
st.sidebar.header("Dashboard Controls")

# --- Sidebar UI ---
scenarios = {
    "Nike, Adidas, Puma, Asics, Under Armour": {"prefix": "athletic_brands", "keywords": ["Nike", "Adidas", "Puma", "Asics", "Under Armour"], "geo_code": "", "timeframe_key": "Past 5 Years"},
    "Apple iPhone, Samsung Galaxy, Google Pixel": {"prefix": "smartphones", "keywords": ["Apple iPhone", "Samsung Galaxy", "Google Pixel"], "geo_code": "US", "timeframe_key": "Past 5 Years"},
    "Mahindra Scorpio, Maruti Suzuki Brezza, Hyundai Creta": {"prefix": "indian_suvs", "keywords": ["Mahindra Scorpio", "Maruti Suzuki Brezza", "Hyundai Creta"], "geo_code": "IN", "timeframe_key": "Past 5 Years"}
}
scenario_options = [""] + list(scenarios.keys())

# Initialize session state for widgets
if 'keywords_input' not in st.session_state:
    st.session_state.keywords_input = "Nike, Adidas, Puma, Asics, Under Armour"
    st.session_state.geo_selection = ""
    st.session_state.timeframe_selection = "Past 5 Years"
    st.session_state.scenario_selector = "Nike, Adidas, Puma, Asics, Under Armour"

def update_state_from_scenario():
    scenario_key = st.session_state.scenario_selector
    if scenario_key and scenario_key in scenarios:
        config = scenarios[scenario_key]
        st.session_state.keywords_input = scenario_key
        st.session_state.geo_selection = config["geo_code"]
        st.session_state.timeframe_selection = config["timeframe_key"]

keywords_input = st.sidebar.text_input("Enter Keywords", key="keywords_input")
st.sidebar.selectbox("Or, select a popular comparison", options=scenario_options, key='scenario_selector', on_change=update_state_from_scenario, help="Selecting an option will pre-fill the controls and use reliable offline data.")

country_list = ['', 'US', 'GB', 'CA', 'AU', 'DE', 'FR', 'IN', 'JP', 'BR', 'ZA']; country_names = ['Worldwide', 'United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'France', 'India', 'Japan', 'Brazil', 'South Africa']; country_dict = dict(zip(country_list, country_names)); geo_keys = list(country_dict.keys())
try: geo_default_index = geo_keys.index(st.session_state.geo_selection)
except ValueError: geo_default_index = 0
geo = st.sidebar.selectbox("Select Region", options=country_list, format_func=lambda x: country_dict[x], index=geo_default_index)

timeframe_options = {"Past Hour": "now 1-H", "Past 4 Hours": "now 4-H", "Past Day": "now 1-d", "Past 7 Days": "now 7-d", "Past 30 Days": "today 1-m", "Past 90 Days": "today 3-m", "Past 12 Months": "today 12-m", "Past 5 Years": "today 5-y", "All Time (Since 2004)": "all"}
timeframe_keys = list(timeframe_options.keys())
try: timeframe_default_index = timeframe_keys.index(st.session_state.timeframe_selection)
except ValueError: timeframe_default_index = 7
timeframe_key = st.sidebar.selectbox("Select Timeframe", options=timeframe_keys, index=timeframe_default_index)
timeframe = timeframe_options[timeframe_key]

# --- Main Dashboard ---
keywords_str = keywords_input
if keywords_str in scenarios:
    keywords = scenarios[keywords_str]["keywords"]
else:
    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()][:5]

if not keywords:
    st.info("⬅️ Please enter keywords or select a popular comparison from the sidebar.")
else:
    st.header(f"Analysis for: {', '.join(keywords)}")
    tab_names = ["📈 Trend Analysis", "🔮 Future Forecast", "🤖 AI Co-pilot"]
    tab1, tab2, tab3 = st.tabs(tab_names)
    
    trends_data = None
    if keywords_str in scenarios:
        with st.spinner(f"Loading pre-configured analysis for '{keywords_str}'..."):
            trends_data = load_all_offline_data(scenarios[keywords_str])
    else:
        serpapi_key = st.secrets.get("SERPAPI_API_KEY")
        if not serpapi_key: st.error("❌ SerpApi API Key not found in secrets.")
        else:
            with st.spinner(f"Fetching live data for '{keywords_str}'..."):
                trends_data = fetch_data_from_serpapi(serpapi_key, keywords, timeframe, geo)
    
    if trends_data:
        interest_df = trends_data.get("interest_over_time", pd.DataFrame())
        if not interest_df.empty:
            for col in interest_df.columns: 
                if col != 'Date': interest_df[col] = pd.to_numeric(interest_df[col], errors='coerce')
        
        with tab1:
            st.subheader("Search Interest Over Time")
            if not interest_df.empty:
                fig = px.line(interest_df, x=interest_df.index, y=interest_df.columns)
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("📊 Overall Interest Share")
                interest_sum = interest_df.sum().reset_index()
                interest_sum.columns = ['keyword', 'total_interest']
                fig_pie = px.pie(interest_sum, names='keyword', values='total_interest')
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("---")
                st.subheader("🌍 Interest by Region")
                region_df = trends_data.get("interest_by_region")
                if region_df is not None and not region_df.empty:
                    fig_map = px.choropleth(region_df, locations=region_df.index, locationmode='country names', color='Interest', hover_name=region_df.index, color_continuous_scale=px.colors.sequential.Plasma)
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.dataframe(region_df.sort_values(by="Interest", ascending=False), use_container_width=True)
                else: st.warning("No regional data available.")
                st.markdown("---")
                st.subheader("📅 Keyword Seasonality Analysis")
                monthly_df = interest_df.resample('M').mean()
                monthly_df['month'] = monthly_df.index.month
                seasonal_df = monthly_df.groupby('month')[interest_df.columns].mean().reset_index()
                for col in interest_df.columns:
                    if (seasonal_df[col].max() - seasonal_df[col].min()) != 0:
                        seasonal_df[col] = (seasonal_df[col] - seasonal_df[col].min()) / (seasonal_df[col].max() - seasonal_df[col].min()) * 100
                    else: seasonal_df[col] = 0
                seasonal_df['month'] = seasonal_df['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
                seasonal_df.set_index('month', inplace=True)
                fig_heatmap = px.imshow(seasonal_df.T, labels=dict(x="Month", y="Keyword", color="Normalized Interest"), aspect="auto", color_continuous_scale="Viridis")
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown("---")
                st.subheader("🔬 Year-over-Year Growth Analysis")
                keyword_for_yoy = st.selectbox("Select keyword for YoY analysis", options=keywords, key='yoy_select')
                if keyword_for_yoy:
                    monthly_yoy_df = interest_df[[keyword_for_yoy]].resample('M').mean()
                    monthly_yoy_df['YoY Growth (%)'] = monthly_yoy_df[keyword_for_yoy].pct_change(12) * 100
                    st.dataframe(monthly_yoy_df.style.format({'YoY Growth (%)': "{:+.2f}%"}).applymap(lambda v: 'color: green;' if v > 0 else ('color: red;' if v < 0 else ''), subset=['YoY Growth (%)']), use_container_width=True)
                st.markdown("---")
                st.subheader("🔍 Trend Decomposition")
                keyword_to_decompose = st.selectbox("Select keyword to decompose", options=keywords, key='decomp_select')
                if keyword_to_decompose:
                    monthly_decomp_df = interest_df[keyword_to_decompose].resample('M').mean()
                    if len(monthly_decomp_df.dropna()) >= 24:
                        decomposition = seasonal_decompose(monthly_decomp_df.dropna(), model='additive', period=12)
                        fig_decomp = decomposition.plot()
                        fig_decomp.set_size_inches(10, 8)
                        st.pyplot(fig_decomp)
                    else: st.error("❌ Analysis Error: Decomposition requires at least 24 months of data.")
            else:
                st.warning("Could not fetch or load time-series data.")
        with tab2:
            st.subheader("Future Forecast with Prophet")
            keyword_to_forecast = st.selectbox("Select a keyword to forecast", options=keywords)
            if not interest_df.empty and keyword_to_forecast in interest_df.columns:
                if st.button(f"Generate 12-Month Forecast for '{keyword_to_forecast}'"):
                    with st.spinner("Calculating future trends..."):
                        prophet_df = interest_df[[keyword_to_forecast]].reset_index()
                        prophet_df.columns = ['ds', 'y']
                        model = Prophet()
                        model.fit(prophet_df)
                        future = model.make_future_dataframe(periods=365)
                        forecast = model.predict(future)
                        st.success("Forecast generated!")
                        fig_forecast = model.plot(forecast)
                        st.pyplot(fig_forecast)
                        st.subheader("Forecast Data")
                        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
            else: st.warning("Trend data must be loaded first.")
        with tab3:
            st.subheader("AI Marketing Co-pilot")
            google_api_key = st.secrets.get("GOOGLE_API_KEY")
            if not google_api_key: st.info("Please add your Google AI API key to your secrets.")
            else:
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                data_summary = ""
                if trends_data:
                    if not trends_data.get("interest_over_time", pd.DataFrame()).empty: data_summary += "Time-series summary:\n" + trends_data["interest_over_time"].describe().to_string() + "\n\n"
                    if not trends_data.get("interest_by_region", pd.DataFrame()).empty: data_summary += "Top 5 regions:\n" + trends_data["interest_by_region"].head().to_string() + "\n\n"
                if "messages" not in st.session_state: st.session_state.messages = []
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]): st.markdown(message["content"])
                if prompt := st.chat_input("Ask about trends or request a marketing campaign..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    full_prompt = f"You are a marketing analyst AI. Based on this data summary:\n{data_summary}\n\nUser's Question: '{prompt}'"
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        try:
                            response = model.generate_content(full_prompt, stream=True)
                            full_response_text = ""
                            for chunk in response:
                                full_response_text += chunk.text
                                message_placeholder.markdown(full_response_text + "▌")
                            message_placeholder.markdown(full_response_text)
                            st.session_state.messages.append({"role": "assistant", "content": full_response_text})
                        except Exception as e:
                            st.error(f"An error occurred with the AI model: {e}")
    else:
        st.error("Could not load or fetch data.")