import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np

# ========== ПОДКЛЮЧЕНИЕ К SUPABASE ==========
st.set_page_config(page_title="KTZ Monitoring", page_icon="🚂", layout="wide")

try:
    DB_CONNECTION = st.secrets["DB_CONNECTION"]
    engine = create_engine(DB_CONNECTION, connect_args={"connect_timeout": 10})
    
    # Проверка подключения
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    st.success("✅ Connected to Supabase")
    
except Exception as e:
    st.error(f"❌ Connection failed: {e}")
    st.stop()

# ========== ЗАГРУЗКА ДАННЫХ ==========
@st.cache_data(ttl=30)
def load_data():
    query = """
    WITH entry_times AS (
        SELECT 
            e.train_id,
            e.track_section,
            e.event_time AS entry_time,
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
        
        if len(df) == 0:
            st.warning("⚠️ No data found. Please run the Colab notebook first to generate data.")
        else:
            st.success(f"✅ Loaded {len(df)} trips")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure tables exist in Supabase. Run the Colab notebook to create them.")
        st.stop()

# Key metrics
if len(df) > 0:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Trains", len(df))
    with col2:
        st.metric("✅ Normal", len(df[df['status'] == '✅ Normal']))
    with col3:
        st.metric("⚠️ Warning", len(df[df['status'] == '⚠️ Warning']))
    with col4:
        st.metric("🔴 Critical", len(df[df['status'] == '🔴 CRITICAL DELAY']))
    with col5:
        avg_time = df['duration_seconds'].mean()
        st.metric("Avg Time", f"{avg_time:.0f} sec")
    
    # Scatter plot
    fig1 = px.scatter(df, x='entry_time', y='duration_seconds', 
                      color='status', hover_data=['train_number', 'track_section'],
                      title="Travel Time Monitoring",
                      color_discrete_map={'✅ Normal': 'blue', '⚠️ Warning': 'orange', '🔴 CRITICAL DELAY': 'red'})
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Two columns
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.histogram(df, x='duration_seconds', color='track_section', 
                            nbins=30, title="Travel Time Distribution")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        problem_data = df[df['status'] != '✅ Normal']['track_section'].value_counts()
        if len(problem_data) > 0:
            fig3 = px.pie(values=problem_data.values, names=problem_data.index, 
                          title="Problems by Section")
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("ℹ️ No problems detected")
    
    # Critical delays table
    st.subheader("🚨 Critical Delays")
    critical = df[df['status'] == '🔴 CRITICAL DELAY']
    if len(critical) > 0:
        st.dataframe(critical[['entry_time', 'train_number', 'track_section', 'duration_seconds']],
                     use_container_width=True)
    else:
        st.success("✅ No critical delays. All trains on schedule!")

# Footer
st.markdown("---")
st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Period: last {hours_back} hours | Threshold: {sigma_threshold}σ")
