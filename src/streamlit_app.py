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

# --- Page Configuration ---
st.set_page_config(
    page_title="Trend-AI Marketing Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_data(ttl=3600)
def fetch_data_from_serpapi(api_key, keywords, timeframe, geo):
    """
    Fetches Google Trends data using the SerpApi service and robustly parses
    all possible date formats (daily, weekly, hourly).
    """
    def _parse_date_string(date_str):
        if '–' in date_str:
            try:
                parts = date_str.split('–')
                start_day_month = parts[0].strip()
                year = parts[1].split(',')[-1].strip()
                return pd.to_datetime(f"{start_day_month}, {year}")
            except:
                return pd.to_datetime(date_str.split('–')[0].strip())
        else:
            return pd.to_datetime(date_str)

    params = {
        "engine": "google_trends",
        "q": ", ".join(keywords),
        "date": timeframe,
        "geo": geo,
        "api_key": api_key
    }

    try:
        client = Client()
        results = client.search(params)

        if 'interest_over_time' in results:
            timeline_data = results['interest_over_time']['timeline_data']
            dates = [_parse_date_string(item['date']) for item in timeline_data]
            data = {}
            for i, keyword in enumerate(keywords):
                data[keyword] = [item['values'][i].get('value', 0) for item in timeline_data]
            
            df = pd.DataFrame(data, index=dates)
            return df
        else:
            st.error(f"SerpApi did not return 'interest_over_time' data. Response: {results.get('error', 'No error message.')}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred with the SerpApi request: {e}")
        return pd.DataFrame()

# --- Main Application Logic ---
st.title("📈 Trend-AI: Marketing Intelligence Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")

keywords_input = st.sidebar.text_input(
    "Enter Keywords to Analyze (comma-separated, max 5)",
    "Nike, Adidas"
)

country_list = ['', 'US', 'GB', 'CA', 'AU', 'DE', 'FR', 'IN', 'JP', 'BR', 'ZA']
country_names = ['Worldwide', 'United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'France', 'India', 'Japan', 'Brazil', 'South Africa']
country_dict = dict(zip(country_list, country_names))

geo = st.sidebar.selectbox("Select Region", options=country_list, format_func=lambda x: country_dict[x], index=0)

timeframe_options = {
    "Past Hour": "now 1-H", "Past 4 Hours": "now 4-H", "Past Day": "now 1-d",
    "Past 7 Days": "now 7-d", "Past 30 Days": "today 1-m", "Past 90 Days": "today 3-m",
    "Past 12 Months": "today 12-m", "Past 5 Years": "today 5-y", "All Time (Since 2004)": "all",
}
timeframe_key = st.sidebar.selectbox("Select Timeframe", options=list(timeframe_options.keys()), index=7)
timeframe = timeframe_options[timeframe_key]

# --- Main Dashboard ---
keywords = [k.strip() for k in keywords_input.split(',') if k.strip()][:5]
serpapi_key = st.secrets.get("SERPAPI_API_KEY")

if not serpapi_key:
    st.error("❌ SerpApi API Key not found. Please add it to your secrets.")
else:
    if not keywords:
        st.info("Please enter at least one keyword in the sidebar to begin analysis.")
    else:
        st.header(f"Analysis for: {', '.join(keywords)}")
        
        tab_names = ["📈 Trend Analysis", "🔮 Future Forecast", "🤖 AI Co-pilot"]
        tab1, tab2, tab3 = st.tabs(tab_names)

        with st.spinner("Fetching reliable data from SerpApi..."):
            interest_df = fetch_data_from_serpapi(serpapi_key, keywords, timeframe, geo)

        # --- THIS IS THE CRITICAL DATA CLEANING FIX ---
        # After fetching, we ensure all keyword columns are converted to numbers.
        # Any non-numeric values (like '<1') will be gracefully replaced with NaN (Not a Number).
        if interest_df is not None and not interest_df.empty:
            for keyword in keywords:
                if keyword in interest_df.columns:
                    interest_df[keyword] = pd.to_numeric(interest_df[keyword], errors='coerce')
        # --- END OF FIX ---

        with tab1:
            st.subheader("Search Interest Over Time")
            if interest_df is not None and not interest_df.empty:
                fig = px.line(interest_df, x=interest_df.index, y=interest_df.columns, title=f'Google Trends for: {", ".join(keywords)}')
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📊 Overall Interest Share")
                interest_sum = interest_df.sum().reset_index()
                interest_sum.columns = ['keyword', 'total_interest']
                fig_pie = px.pie(interest_sum, names='keyword', values='total_interest', title='Share of Search Interest')
                st.plotly_chart(fig_pie, use_container_width=True)

                with st.expander("📅 View Keyword Seasonality Analysis"):
                    monthly_df = interest_df.resample('M').mean() # This line will now work
                    monthly_df['month'] = monthly_df.index.month
                    seasonal_df = monthly_df.groupby('month')[keywords].mean().reset_index()
                    for col in keywords:
                        if (seasonal_df[col].max() - seasonal_df[col].min()) != 0:
                            seasonal_df[col] = (seasonal_df[col] - seasonal_df[col].min()) / (seasonal_df[col].max() - seasonal_df[col].min()) * 100
                        else:
                            seasonal_df[col] = 0
                    seasonal_df['month'] = seasonal_df['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
                    seasonal_df.set_index('month', inplace=True)
                    fig_heatmap = px.imshow(seasonal_df.T, labels=dict(x="Month", y="Keyword", color="Normalized Interest"), aspect="auto", color_continuous_scale="Viridis")
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                with st.expander("🔬 View Year-over-Year Growth Analysis"):
                    keyword_for_yoy = st.selectbox("Select keyword for YoY analysis", options=keywords, key='yoy_select')
                    if keyword_for_yoy:
                        monthly_yoy_df = interest_df[[keyword_for_yoy]].resample('M').mean()
                        monthly_yoy_df['YoY Growth (%)'] = monthly_yoy_df[keyword_for_yoy].pct_change(12) * 100
                        st.dataframe(monthly_yoy_df.style.format({'YoY Growth (%)': "{:+.2f}%"}).applymap(lambda v: 'color: green;' if v > 0 else ('color: red;' if v < 0 else ''), subset=['YoY Growth (%)']), use_container_width=True)
                
                with st.expander("🔍 View Trend Decomposition"):
                    keyword_to_decompose = st.selectbox("Select keyword to decompose", options=keywords, key='decomp_select')
                    if keyword_to_decompose:
                        monthly_decomp_df = interest_df[keyword_to_decompose].resample('M').mean()
                        if len(monthly_decomp_df.dropna()) >= 24:
                            decomposition = seasonal_decompose(monthly_decomp_df.dropna(), model='additive', period=12)
                            fig_decomp = decomposition.plot()
                            fig_decomp.set_size_inches(10, 8)
                            st.pyplot(fig_decomp)
                        else:
                            st.error("❌ Analysis Error: Decomposition requires at least 24 months of data.")
            else:
                st.warning("Could not fetch data.")
        
        with tab2:
            st.subheader("Future Forecast with Prophet")
            keyword_to_forecast = st.selectbox("Select a keyword to forecast", options=keywords)
            if interest_df is not None and not interest_df.empty and keyword_to_forecast in interest_df.columns:
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
            else:
                st.warning("Trend data must be loaded first to run a forecast.")
        
        with tab3:
            st.subheader("AI Marketing Co-pilot")
            google_api_key = st.secrets.get("GOOGLE_API_KEY")
            if not google_api_key:
                st.info("Please add your Google AI API key to your secrets to use the AI Co-pilot.")
            else:
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                data_summary = ""
                if interest_df is not None and not interest_df.empty:
                    data_summary += "Search interest over time summary:\n" + interest_df.describe().to_string()
                else:
                    data_summary = "No data available to summarize."
                
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                if prompt := st.chat_input("Ask about trends or request a marketing campaign..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
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