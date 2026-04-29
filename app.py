import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np
import urllib.parse

# ========== ПОДКЛЮЧЕНИЕ К SUPABASE ==========
st.set_page_config(page_title="KTZ Monitoring", page_icon="🚂", layout="wide")

try:
    DB_HOST = st.secrets["DB_HOST"]
    DB_PORT = st.secrets["DB_PORT"]
    DB_NAME = st.secrets["DB_NAME"]
    DB_USER = st.secrets["DB_USER"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    
    DB_PASSWORD_ESCAPED = urllib.parse.quote_plus(DB_PASSWORD)
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD_ESCAPED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
except KeyError as e:
    st.error(f"❌ Missing secret: {e}. Please check your Secrets configuration.")
    st.info("Go to Settings → Secrets and add: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
    st.stop()

@st.cache_resource
def init_connection():
    try:
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 10})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        st.stop()

engine = init_connection()

# ========== ЗАГРУЗКА ДАННЫХ ==========
@st.cache_data(ttl=30)
def load_data():
    query = """
    WITH entry_times AS (
        SELECT 
            e.train_id,
            e.track_section,
            e.event_time AS entry_time,
            e.value AS speed,
            t.train_number
        FROM sensor_events e
        JOIN trains t ON e.train_id = t.train_id
        WHERE e.event_type = 'entry'
    ),
    exit_times AS (
        SELECT 
            e.train_id,
            e.track_section,
            e.event_time AS exit_time
        FROM sensor_events e
        WHERE e.event_type = 'exit'
    )
    SELECT 
        e.train_id,
        e.train_number,
        e.track_section,
        e.entry_time,
        x.exit_time,
        EXTRACT(EPOCH FROM (x.exit_time - e.entry_time)) AS duration_seconds
    FROM entry_times e
    JOIN exit_times x 
        ON e.train_id = x.train_id AND e.track_section = x.track_section
    WHERE x.exit_time > e.entry_time
    ORDER BY e.entry_time DESC
    """
    return pd.read_sql(query, engine)

def detect_anomalies(df, sigma=2.0):
    if len(df) == 0:
        return df
    
    stats = df.groupby('track_section')['duration_seconds'].agg(['mean', 'std']).reset_index()
    stats.columns = ['track_section', 'mean_time', 'std_time']
    stats['threshold'] = stats['mean_time'] + sigma * stats['std_time']
    stats['warning_threshold'] = stats['mean_time'] + 0.5 * stats['std_time']
    
    df_result = df.merge(stats[['track_section', 'mean_time', 'threshold', 'warning_threshold']], on='track_section')
    df_result['status'] = '✅ Normal'
    df_result.loc[df_result['duration_seconds'] > df_result['warning_threshold'], 'status'] = '⚠️ Warning'
    df_result.loc[df_result['duration_seconds'] > df_result['threshold'], 'status'] = '🔴 CRITICAL DELAY'
    
    return df_result

# ========== ИНТЕРФЕЙС ==========
st.title("🚂 KTZ Railway Monitoring System")
st.markdown("*Real-time SCADA Data Analytics*")

# Sidebar
st.sidebar.header("⚙️ Settings")
sigma_threshold = st.sidebar.slider("Detector Sensitivity (σ)", 1.0, 3.0, 2.0, 0.1)
hours_back = st.sidebar.slider("Analysis Period (hours)", 1, 168, 24)

if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Load data
with st.spinner("Loading data from Supabase..."):
    try:
        df_raw = load_data()
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours_back)
        df_raw = df_raw[df_raw['entry_time'] > cutoff_time]
        df = detect_anomalies(df_raw, sigma_threshold)
        st.success(f"✅ Loaded {len(df)} trips")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you have generated data in your database first (run the Colab notebook).")
        st.stop()

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Trains", len(df))
with col2:
    st.metric("✅ Normal", len(df[df['status'] == '✅ Normal']))
with col3:
    st.metric("⚠️ Warning", len(df[df['status'] == '⚠️ Warning']))
with col4:
    critical_count = len(df[df['status'] == '🔴 CRITICAL DELAY'])
    st.metric("🔴 Critical", critical_count)
with col5:
    avg_time = df['duration_seconds'].mean() if len(df) > 0 else 0
    st.metric("Avg Time", f"{avg_time:.0f} sec")

# Scatter plot
if len(df) > 0:
    fig1 = px.scatter(df, x='entry_time', y='duration_seconds', 
                      color='status', hover_data=['train_number', 'track_section'],
                      color_discrete_map={'✅ Normal': 'blue', '⚠️ Warning': 'orange', '🔴 CRITICAL DELAY': 'red'})
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)

# Critical delays table
st.subheader("🚨 Critical Delays")
critical = df[df['status'] == '🔴 CRITICAL DELAY']
if len(critical) > 0:
    st.dataframe(critical[['entry_time', 'train_number', 'track_section', 'duration_seconds']])
else:
    st.success("✅ No critical delays")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
