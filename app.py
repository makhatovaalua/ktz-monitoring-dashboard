import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np

# ========== CONNECTION TO SUPABASE (via Secrets) ==========
# Streamlit automatically pulls variables from Secrets

try:
    # Trying to get data from Secrets
    DB_HOST = st.secrets["DB_HOST"]
    DB_USER = st.secrets["DB_USER"] 
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    DB_PORT = st.secrets.get("DB_PORT", "5432")
    DB_NAME = st.secrets.get("DB_NAME", "postgres")
    
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
except Exception as e:
    st.error(f"❌ Error loading Secrets: {e}")
    st.info("Make sure secrets are added in Streamlit Cloud settings")
    st.stop()

@st.cache_resource
def init_connection():
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        st.error(f"❌ Failed to connect to database: {e}")
        st.stop()

engine = init_connection()

# ========== DATA LOADING ==========
@st.cache_data(ttl=60)
def load_data():
    """Loads data from database"""
    try:
        query = """
        WITH entry_times AS (
            SELECT e.train_id, e.track_section, e.event_time AS entry_time,
                   e.value AS speed, t.train_number
            FROM sensor_events e
            JOIN trains t ON e.train_id = t.train_id
            WHERE e.event_type = 'entry'
        ),
        exit_times AS (
            SELECT e.train_id, e.track_section, e.event_time AS exit_time
            FROM sensor_events e
            WHERE e.event_type = 'exit'
        )
        SELECT e.train_id, e.train_number, e.track_section, e.entry_time,
               x.exit_time, 
               EXTRACT(EPOCH FROM (x.exit_time - e.entry_time)) AS duration_seconds
        FROM entry_times e
        JOIN exit_times x ON e.train_id = x.train_id AND e.track_section = x.track_section
        WHERE x.exit_time > e.entry_time
        ORDER BY e.entry_time DESC
        LIMIT 500
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Check if there is data in sensor_events and trains tables")
        return pd.DataFrame()

# ========== ANOMALY DETECTION FUNCTION ==========
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

# ========== DASHBOARD INTERFACE ==========
st.set_page_config(page_title="KTZ Monitoring", page_icon="🚂", layout="wide")

st.title("🚂 KTZ Delay Monitoring System")
st.markdown("*Analytics based on PostgreSQL + Streamlit*")

# Show connection status
with st.expander("🔧 Connection status", expanded=False):
    st.success(f"✅ Connected to Supabase: {DB_HOST}")

# Loading data
with st.spinner("📡 Loading data from Supabase..."):
    df_raw = load_data()

if len(df_raw) == 0:
    st.warning("⚠️ No data to display")
    st.info("First run the data generator in Google Colab to populate the database")
    st.stop()

# Anomaly analysis
df = detect_anomalies(df_raw)

# ========== KEY METRICS ==========
st.subheader("📊 Key metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total trains", len(df))
with col2:
    normal_count = len(df[df['status'] == '✅ Normal'])
    st.metric("✅ Normal", normal_count)
with col3:
    warning_count = len(df[df['status'] == '⚠️ Warning'])
    st.metric("⚠️ Warning", warning_count)
with col4:
    critical_count = len(df[df['status'] == '🔴 CRITICAL DELAY'])
    st.metric("🔴 Critical", critical_count)

# ========== DATA TABLE ==========
st.subheader("📋 Recent trains")
st.dataframe(df[['entry_time', 'train_number', 'track_section', 'duration_seconds', 'status']].head(20), use_container_width=True)

# ========== CRITICAL DELAYS ==========
st.subheader("🚨 Critical delays")
critical_trips = df[df['status'] == '🔴 CRITICAL DELAY']
if len(critical_trips) > 0:
    st.dataframe(critical_trips[['train_number', 'track_section', 'duration_seconds', 'mean_time']], use_container_width=True)
else:
    st.success("✅ No critical delays")

# ========== CHART ==========
st.subheader("📈 Section travel time")
if len(df) > 0:
    fig = px.scatter(df, x='entry_time', y='duration_seconds', 
                     color='status', title="Real-time monitoring",
                     labels={'duration_seconds': 'Seconds', 'entry_time': 'Entry time'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
