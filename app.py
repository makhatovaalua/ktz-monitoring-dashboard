import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np
import urllib.parse

# ========== ПОДКЛЮЧЕНИЕ К SUPABASE ==========
# Получаем данные из Secrets
try:
    DB_HOST = st.secrets["DB_HOST"]
    DB_PORT = st.secrets["DB_PORT"]
    DB_NAME = st.secrets["DB_NAME"]
    DB_USER = st.secrets["DB_USER"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    
    # Экранируем пароль (на случай спецсимволов)
    DB_PASSWORD_ESCAPED = urllib.parse.quote_plus(DB_PASSWORD)
    
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD_ESCAPED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
except Exception as e:
    st.error(f"Error reading secrets: {e}")
    st.stop()

@st.cache_resource
def init_connection():
    try:
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 10})
        # Проверяем подключение
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

engine = init_connection()
st.success("✅ Connected to Supabase successfully!")

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
    """Детектор аномалий с настраиваемым порогом"""
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

# ========== ИНТЕРФЕЙС ДАШБОРДА ==========
st.set_page_config(page_title="KTZ Monitoring", page_icon="🚂", layout="wide")

st.title("🚂 KTZ Railway Monitoring System")
st.markdown("*Real-time SCADA Data Analytics*")

# Sidebar
st.sidebar.header("⚙️ Settings")
sigma_threshold = st.sidebar.slider("Detector Sensitivity (σ)", 1.0, 3.0, 2.0, 0.1,
                                     help="Lower value = more detections, but more false alarms")
hours_back = st.sidebar.slider("Analysis Period (hours)", 1, 168, 24)

if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Load data
with st.spinner("Loading data from Supabase..."):
    df_raw = load_data()
    # Filter by hours
    cutoff_time = datetime.now() - pd.Timedelta(hours=hours_back)
    df_raw = df_raw[df_raw['entry_time'] > cutoff_time]
    df = detect_anomalies(df_raw, sigma_threshold)

# Key metrics
st.subheader("📊 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Trains", len(df))
with col2:
    normal_count = len(df[df['status'] == '✅ Normal'])
    st.metric("✅ Normal", normal_count)
with col3:
    warning_count = len(df[df['status'] == '⚠️ Warning'])
    st.metric("⚠️ Warning", warning_count)
with col4:
    critical_count = len(df[df['status'] == '🔴 CRITICAL DELAY'])
    st.metric("🔴 Critical", critical_count, 
              delta=f"{critical_count/len(df)*100:.1f}%" if len(df)>0 else None)
with col5:
    avg_time = df['duration_seconds'].mean() if len(df) > 0 else 0
    st.metric("Avg Travel Time", f"{avg_time:.0f} sec")

# Plot 1: Scatter plot
if len(df) > 0:
    st.subheader("📈 Travel Time Monitoring")
    fig1 = px.scatter(df, x='entry_time', y='duration_seconds', 
                      color='status', hover_data=['train_number', 'track_section'],
                      title="Travel Time by Section",
                      color_discrete_map={'✅ Normal': 'blue', '⚠️ Warning': 'orange', '🔴 CRITICAL DELAY': 'red'})
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)

# Two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Travel Time Distribution")
    if len(df) > 0:
        fig2 = px.histogram(df, x='duration_seconds', color='track_section', 
                            nbins=30, title="Histogram",
                            labels={'duration_seconds': 'Seconds', 'count': 'Number of trains'})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("🥧 Problems by Section")
    problem_data = df[df['status'] != '✅ Normal']['track_section'].value_counts()
    if len(problem_data) > 0:
        fig3 = px.pie(values=problem_data.values, names=problem_data.index, title="Problem Distribution")
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("ℹ️ No problems detected")

# Comparison bar chart
st.subheader("🏆 Section Comparison")
if len(df) > 0:
    section_stats = df.groupby('track_section').agg({
        'duration_seconds': ['mean', 'std', 'count']
    }).reset_index()
    section_stats.columns = ['track_section', 'mean_time', 'std_time', 'count']
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=section_stats['track_section'], y=section_stats['mean_time'],
                          name='Average Time', marker_color='green',
                          text=section_stats['mean_time'].round(0), textposition='auto'))
    fig4.add_trace(go.Bar(x=section_stats['track_section'], y=section_stats['std_time'],
                          name='Std Deviation', marker_color='orange',
                          text=section_stats['std_time'].round(0), textposition='auto'))
    fig4.update_layout(title="Section Comparison (Mean ± Std)", height=400, barmode='group')
    st.plotly_chart(fig4, use_container_width=True)

# Critical delays table
st.subheader("🚨 Critical Delays (Requires Attention)")
critical_trips = df[df['status'] == '🔴 CRITICAL DELAY'].copy()
if len(critical_trips) > 0:
    critical_trips['duration_seconds'] = critical_trips['duration_seconds'].round(0).astype(int)
    st.dataframe(
        critical_trips[['entry_time', 'train_number', 'track_section', 'duration_seconds']],
        column_config={
            'entry_time': 'Entry Time',
            'train_number': 'Train',
            'track_section': 'Section',
            'duration_seconds': 'Duration (sec)'
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.success("✅ No critical delays. All trains on schedule!")

# Footer
st.markdown("---")
st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Period: last {hours_back} hours | Detection threshold: {sigma_threshold}σ")
