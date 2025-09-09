# app.py
import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
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
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_pytrends_object():
    return TrendReq(hl='en-US', tz=360)

@st.cache_data(ttl=3600)
def fetch_data_from_trends(_pytrends, keywords, timeframe, geo):
    time.sleep(2) # Be respectful to Google's servers
    try:
        _pytrends.build_payload(kw_list=keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
        return _pytrends.interest_over_time()
    except Exception as e:
        st.error(f"Error fetching data from Google Trends: {e}")
        return None

# --- Main Application Logic ---
st.title("📈 Trend-AI: Marketing Intelligence Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")

keywords_input = st.sidebar.text_input(
    "Enter Keywords to Analyze (comma-separated, max 5)",
    "ceramide, hyaluronic acid, niacinamide"
)

country_list = ['', 'US', 'GB', 'CA', 'AU', 'DE', 'FR', 'IN', 'JP', 'BR', 'ZA']
country_names = ['Worldwide', 'United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'France', 'India', 'Japan', 'Brazil', 'South Africa']
country_dict = dict(zip(country_list, country_names))

geo = st.sidebar.selectbox(
    "Select Region",
    options=country_list,
    format_func=lambda x: country_dict[x],
    index=1 # Default to United States
)

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ['today 5-y', 'today 12-m', 'today 3-m', 'now 7-d'],
    index=0
)

# --- Main Dashboard ---
keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]

if not keywords:
    st.info("Please enter at least one keyword in the sidebar to begin analysis.")
else:
    st.header(f"Analysis for: {', '.join(keywords)}")
    pytrends = get_pytrends_object()
    
    tab_names = ["📈 Trend Analysis", "🔮 Future Forecast", "🤖 AI Co-pilot"]
    tab1, tab2, tab3 = st.tabs(tab_names)

    interest_df = fetch_data_from_trends(pytrends, keywords, timeframe, geo)

    with tab1:
        st.subheader("Search Interest Over Time")
        if interest_df is not None and not interest_df.empty:
            if 'isPartial' in interest_df.columns:
                interest_df.drop(columns=['isPartial'], inplace=True)
            
            # 1. Main Trend Line Chart
            fig = px.line(interest_df, x=interest_df.index, y=interest_df.columns, title=f'Google Trends for: {", ".join(keywords)}')
            st.plotly_chart(fig, use_container_width=True)

            # 2. Overall Interest Share Pie Chart
            st.subheader("📊 Overall Interest Share")
            interest_sum = interest_df.sum().reset_index()
            interest_sum.columns = ['keyword', 'total_interest']
            fig_pie = px.pie(interest_sum, names='keyword', values='total_interest', title='Share of Search Interest')
            st.plotly_chart(fig_pie, use_container_width=True)

            # 3. Keyword Seasonality Analysis
            with st.expander("📅 View Keyword Seasonality Analysis"):
                monthly_df = interest_df.resample('M').mean()
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

            # 4. Year-over-Year Growth Analysis
            with st.expander("🔬 View Year-over-Year Growth Analysis"):
                keyword_for_yoy = st.selectbox("Select keyword for YoY analysis", options=keywords, key='yoy_select')
                if keyword_for_yoy:
                    monthly_yoy_df = interest_df[[keyword_for_yoy]].resample('M').mean()
                    monthly_yoy_df['YoY Growth (%)'] = monthly_yoy_df[keyword_for_yoy].pct_change(12) * 100
                    st.dataframe(monthly_yoy_df.style.format({'YoY Growth (%)': "{:+.2f}%"}).applymap(lambda v: 'color: green;' if v > 0 else ('color: red;' if v < 0 else ''), subset=['YoY Growth (%)']), use_container_width=True)
            
            # 5. Time-Series Decomposition (Corrected)
            with st.expander("🔍 View Trend Decomposition"):
                keyword_to_decompose = st.selectbox("Select keyword to decompose", options=keywords, key='decomp_select')
                if keyword_to_decompose:
                    monthly_decomp_df = interest_df[keyword_to_decompose].resample('M').mean()
                    # Check if there is enough data for decomposition
                    if len(monthly_decomp_df.dropna()) >= 24:
                        decomposition = seasonal_decompose(monthly_decomp_df.dropna(), model='additive', period=12)
                        
                        # Capture the figure returned by the plot() method
                        fig_decomp = decomposition.plot()
                        # Adjust the figure size for better readability
                        fig_decomp.set_size_inches(10, 8)
                        
                        # Display the correct figure in Streamlit
                        st.pyplot(fig_decomp)
                    else:
                        # Display a prominent error if there isn't enough data
                        st.error("❌ Analysis Error: Decomposition requires at least 24 months of data. Please select a longer timeframe like 'today 5-y' from the sidebar.")
        else:
            st.warning("Could not fetch data. This may be due to a temporary issue with Google Trends or the selected keywords having no search volume.")

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
            google_api_key = st.text_input("Enter your Google AI API Key", type="password")
        if not google_api_key:
            st.info("Please add your Google AI API key to use the AI Co-pilot.")
        else:
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel('gemini-pro')
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
                        st.error(f"An error occurred: {e}")