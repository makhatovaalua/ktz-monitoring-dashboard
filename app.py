import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import numpy as np

# ========== PAGE SETTINGS ==========
st.set_page_config(
    page_title="KTZ Delay Monitoring",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== DATABASE CONNECTION ==========

DB_HOST = st.secrets["DB_HOST"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_PORT = st.secrets.get("DB_PORT", "5432")
DB_NAME = st.secrets.get("DB_NAME", "postgres")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

@st.cache_resource
def init_connection():
    """Creates and caches database connection"""
    return create_engine(DATABASE_URL)

engine = init_connection()

# ========== DATA LOADING FUNCTIONS ==========
@st.cache_data(ttl=60)
def load_data(hours_back=168):
    """Loads train passage data for the last N hours"""
    query = f"""
    WITH entry_times AS (
        SELECT 
            e.train_id,
            e.track_section,
            e.event_time AS entry_time,
            e.value AS entry_speed,
            t.train_number
        FROM sensor_events e
        JOIN trains t ON e.train_id = t.train_id
        WHERE e.event_type = 'entry'
        AND e.event_time > NOW() - INTERVAL '{hours_back} hours'
    ),
    exit_times AS (
        SELECT 
            e.train_id,
            e.track_section,
            e.event_time AS exit_time,
            e.value AS exit_speed
        FROM sensor_events e
        WHERE e.event_type = 'exit'
        AND e.event_time > NOW() - INTERVAL '{hours_back} hours'
    )
    SELECT 
        e.train_id,
        e.train_number,
        e.track_section,
        e.entry_time,
        x.exit_time,
        EXTRACT(EPOCH FROM (x.exit_time - e.entry_time)) AS duration_seconds,
        e.entry_speed,
        x.exit_speed
    FROM entry_times e
    JOIN exit_times x 
        ON e.train_id = x.train_id AND e.track_section = x.track_section
    WHERE x.exit_time > e.entry_time
    ORDER BY e.entry_time DESC
    """
    
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def detect_anomalies(df, sigma_threshold=2.0):
    """Detects anomalies using sigma rule"""
    if len(df) == 0:
        return df
    
    # Calculate statistics for each section
    stats = df.groupby('track_section')['duration_seconds'].agg(['mean', 'std', 'count']).reset_index()
    stats.columns = ['track_section', 'mean_seconds', 'std_seconds', 'count_trips']
    stats['threshold_anomaly'] = stats['mean_seconds'] + sigma_threshold * stats['std_seconds']
    stats['threshold_warning'] = stats['mean_seconds'] + 0.8 * stats['std_seconds']
    
    # Apply thresholds
    df_result = df.merge(stats[['track_section', 'mean_seconds', 'threshold_warning', 'threshold_anomaly']], on='track_section')
    df_result['status'] = '✅ Normal'
    df_result.loc[df_result['duration_seconds'] > df_result['threshold_warning'], 'status'] = '⚠️ Warning'
    df_result.loc[df_result['duration_seconds'] > df_result['threshold_anomaly'], 'status'] = '🔴 CRITICAL DELAY'
    
    return df_result, stats

# ========== HEADER ==========
st.title("🚂 KTZ Delay Monitoring and Detection System")
st.markdown("*Real-time SACS data analysis*")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Analysis period
    hours_back = st.slider(
        "Analysis period (hours)",
        min_value=1,
        max_value=168,
        value=24,
        help="Data for the last N hours"
    )
    
    # Detector sensitivity
    sigma_threshold = st.slider(
        "Detector sensitivity (σ)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="Lower value = more detections, but higher risk of false alarms"
    )
    
    # Refresh button
    if st.button("🔄 Refresh data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    # Database status
    st.header("🗄️ Database status")
    try:
        with engine.connect() as conn:
            events_count = conn.execute(text("SELECT COUNT(*) FROM sensor_events")).scalar()
            trains_count = conn.execute(text("SELECT COUNT(*) FROM trains")).scalar()
            sensors_count = conn.execute(text("SELECT COUNT(*) FROM sensors")).scalar()
        
        st.metric("📡 Events in DB", events_count)
        st.metric("🚂 Trains", trains_count)
        st.metric("📊 Sensors", sensors_count)
    except Exception as e:
        st.error(f"Connection error: {e}")

# ========== DATA LOADING ==========
with st.spinner("Loading data from database..."):
    df_raw = load_data(hours_back)
    
    if len(df_raw) == 0:
        st.warning("⚠️ No data for selected period. Run the data generator in Colab.")
        st.stop()
    
    df_analysis, stats = detect_anomalies(df_raw, sigma_threshold)
    anomalies = df_analysis[df_analysis['status'] != '✅ Normal']

# ========== KEY METRICS ==========
st.subheader("📊 Key metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🚂 Total trips", len(df_analysis))

with col2:
    normal_count = len(df_analysis[df_analysis['status'] == '✅ Normal'])
    st.metric("✅ Normal", normal_count)

with col3:
    warning_count = len(df_analysis[df_analysis['status'] == '⚠️ Warning'])
    st.metric("⚠️ Warning", warning_count)

with col4:
    critical_count = len(df_analysis[df_analysis['status'] == '🔴 CRITICAL DELAY'])
    st.metric("🔴 Critical", critical_count, 
              delta=f"{critical_count/len(df_analysis)*100:.1f}%" if len(df_analysis) > 0 else None)

with col5:
    avg_time = df_analysis['duration_seconds'].mean()
    st.metric("⏱️ Average time", f"{avg_time:.0f} sec")

# ========== CHART 1: TRAVEL TIME ==========
st.subheader("📈 Section travel time monitoring")

fig1 = go.Figure()

colors = {'✅ Normal': 'blue', '⚠️ Warning': 'orange', '🔴 CRITICAL DELAY': 'red'}

for status, color in colors.items():
    status_data = df_analysis[df_analysis['status'] == status]
    if len(status_data) > 0:
        fig1.add_trace(go.Scatter(
            x=status_data['entry_time'],
            y=status_data['duration_seconds'],
            mode='markers',
            name=status,
            marker=dict(size=12, color=color, 
                       symbol='circle' if status == '✅ Normal' else 'triangle-up'),
            text=status_data['train_number'],
            customdata=status_data['track_section'],
            hovertemplate='<b>Train: %{text}</b><br>' +
                         'Time: %{y:.0f} sec<br>' +
                         'Section: %{customdata}<br>' +
                         '<extra></extra>'
        ))

if len(df_analysis) > 0:
    fig1.add_hline(y=df_analysis['duration_seconds'].mean(), 
                   line_dash="dash", 
                   line_color="green", 
                   line_width=2,
                   annotation_text=f"Average: {df_analysis['duration_seconds'].mean():.0f} sec")

fig1.update_layout(
    title="Section travel time (color = status)",
    xaxis_title="Entry time to section",
    yaxis_title="Travel time (seconds)",
    height=500,
    hovermode='closest',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig1, use_container_width=True)

# ========== CHARTS 2 AND 3 (TWO COLUMNS) ==========
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Time distribution by section")
    
    fig2 = px.box(df_analysis, x='track_section', y='duration_seconds', 
                  color='track_section', 
                  title="Travel time distribution",
                  labels={'track_section': 'Section', 'duration_seconds': 'Seconds'})
    fig2.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("🥧 Problems by section")
    
    if len(anomalies) > 0:
        problem_counts = anomalies[anomalies['status'] != '✅ Normal']['track_section'].value_counts()
        fig3 = px.pie(values=problem_counts.values, 
                      names=problem_counts.index,
                      title="Problem distribution by section",
                      hole=0.3)
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("ℹ️ No problems to display")
        st.metric("All sections", "normal", delta="✅")

# ========== CHART 4: SECTION COMPARISON ==========
st.subheader("🏆 Section comparison and anomaly thresholds")

if len(stats) > 0:
    fig4 = go.Figure()
    
    fig4.add_trace(go.Bar(
        x=stats['track_section'],
        y=stats['mean_seconds'],
        name='Average time',
        marker_color='green',
        text=stats['mean_seconds'].round(0).astype(int),
        textposition='auto'
    ))
    
    fig4.add_trace(go.Bar(
        x=stats['track_section'],
        y=stats['threshold_anomaly'],
        name=f'Anomaly threshold ({sigma_threshold}σ)',
        marker_color='red',
        text=stats['threshold_anomaly'].round(0).astype(int),
        textposition='auto'
    ))
    
    fig4.update_layout(
        title="Comparison of average time with detection threshold",
        xaxis_title="Section",
        yaxis_title="Seconds",
        height=450,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig4, use_container_width=True)

# ========== CRITICAL DELAYS TABLE ==========
st.subheader("🚨 Critical delays (require dispatcher attention)")

critical_trips = df_analysis[df_analysis['status'] == '🔴 CRITICAL DELAY'].copy()

if len(critical_trips) > 0:
    critical_trips['duration_seconds'] = critical_trips['duration_seconds'].round(0).astype(int)
    critical_trips['mean_seconds'] = critical_trips['mean_seconds'].round(0).astype(int)
    critical_trips['excess'] = critical_trips['duration_seconds'] - critical_trips['mean_seconds']
    critical_trips['excess_minutes'] = (critical_trips['excess'] / 60).round(1)
    
    st.dataframe(
        critical_trips[['entry_time', 'train_number', 'track_section', 
                        'duration_seconds', 'mean_seconds', 'excess_minutes']],
        column_config={
            'entry_time': '⏱️ Entry time',
            'train_number': '🚂 Train',
            'track_section': '📍 Section',
            'duration_seconds': '⏱️ Travel time (sec)',
            'mean_seconds': '📊 Normal (sec)',
            'excess_minutes': '⏰ Delay (min)'
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.warning(f"⚠️ Detected {len(critical_trips)} critical delays! Dispatcher intervention required.")
else:
    st.success("✅ No critical delays detected. All trains are on schedule!")

# ========== WARNINGS TABLE ==========
with st.expander("⚠️ View all warnings"):
    warning_trips = df_analysis[df_analysis['status'] == '⚠️ Warning'].copy()
    if len(warning_trips) > 0:
        warning_trips['duration_seconds'] = warning_trips['duration_seconds'].round(0).astype(int)
        warning_trips['mean_seconds'] = warning_trips['mean_seconds'].round(0).astype(int)
        st.dataframe(warning_trips[['entry_time', 'train_number', 'track_section', 'duration_seconds', 'mean_seconds']],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No warnings")

# ========== ANOMALY HISTORY ==========
with st.expander("📜 Critical delays history (from DB)"):
    try:
        history_df = pd.read_sql("SELECT * FROM detected_anomalies ORDER BY detected_at DESC LIMIT 50", engine)
        if len(history_df) > 0:
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No saved history")
    except:
        st.info("History table not found or empty")

# ========== FOOTER ==========
st.divider()
st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"📊 Analysis period: last {hours_back} hours")
st.caption(f"⚙️ Detection threshold: {sigma_threshold}σ (95.4% of data is normal with normal distribution)")   
