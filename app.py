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
st.caption(f"⚙️ Detection threshold: {sigma_threshold}σ (95.4% of data is normal with normal distribution)")    train_id SERIAL PRIMARY KEY,
    train_number VARCHAR(20) NOT NULL,
    direction VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sensors table
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id SERIAL PRIMARY KEY,
    sensor_name VARCHAR(50) NOT NULL,
    track_section VARCHAR(50) NOT NULL,
    sensor_type VARCHAR(30),
    is_active BOOLEAN DEFAULT TRUE
);

-- SACS events table
CREATE TABLE IF NOT EXISTS sensor_events (
    event_id SERIAL PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    train_id INTEGER REFERENCES trains(train_id),
    sensor_id INTEGER REFERENCES sensors(sensor_id),
    track_section VARCHAR(50),
    event_type VARCHAR(20),
    value FLOAT
);

-- Indexes for speed
CREATE INDEX IF NOT EXISTS idx_events_time ON sensor_events(event_time);
CREATE INDEX IF NOT EXISTS idx_events_train ON sensor_events(train_id);
"""

with engine.connect() as conn:
    conn.execute(text(create_tables_sql))  # wrap in text()
    conn.commit()
    print("✅ Tables created")

from sqlalchemy import text


trains_data = [
    ('KTJ-101', 'North'),
    ('KTJ-102', 'South'),
    ('KTJ-103', 'North'),
    ('KTJ-104', 'East')
]

with engine.connect() as conn:
    for train_num, direction in trains_data:

        check = conn.execute(text("SELECT COUNT(*) FROM trains WHERE train_number = :tn"), {"tn": train_num}).scalar()
        if check == 0:
            conn.execute(text("INSERT INTO trains (train_number, direction) VALUES (:tn, :dir)"), {"tn": train_num, "dir": direction})
    conn.commit()
    print("✅ Trains added")


sensors_data = [
    ('Датчик_вход_Алматы1', 'Алматы-1 - Алматы-2', 'entry'),
    ('Датчик_выход_Алматы2', 'Алматы-1 - Алматы-2', 'exit'),
    ('Датчик_вход_Алматы2', 'Алматы-2 - Шелек', 'entry'),
    ('Датчик_выход_Шелек', 'Алматы-2 - Шелек', 'exit')
]

with engine.connect() as conn:
    for name, section, s_type in sensors_data:
        check = conn.execute(text("SELECT COUNT(*) FROM sensors WHERE sensor_name = :sn"), {"sn": name}).scalar()
        if check == 0:
            conn.execute(
                text("INSERT INTO sensors (sensor_name, track_section, sensor_type) VALUES (:name, :section, :type)"),
                {"name": name, "section": section, "type": s_type}
            )
    conn.commit()
    print("✅ Sensors added")

import random
from datetime import datetime, timedelta
from sqlalchemy import text

def generate_train_journey(train_id, track_section, start_time_minutes_ago):
    """Generates entry and exit for one train on the section"""

    entry_time = datetime.now() - timedelta(minutes=start_time_minutes_ago)

    # Normal travel time (60-180 seconds)
    normal_duration = random.uniform(60, 180)

    # Sometimes add a delay (20% probability)
    if random.random() < 0.2:
        normal_duration += random.uniform(120, 300)
        is_delayed = True
    else:
        is_delayed = False

    exit_time = entry_time + timedelta(seconds=normal_duration)

    # Get sensor IDs
    with engine.connect() as conn:
        entry_sensor = conn.execute(
            text("SELECT sensor_id FROM sensors WHERE track_section = :section AND sensor_type = 'entry' LIMIT 1"),
            {"section": track_section}
        ).fetchone()

        exit_sensor = conn.execute(
            text("SELECT sensor_id FROM sensors WHERE track_section = :section AND sensor_type = 'exit' LIMIT 1"),
            {"section": track_section}
        ).fetchone()

    events = [
        (entry_time, train_id, entry_sensor[0], track_section, 'entry', random.uniform(40, 80)),
        (exit_time, train_id, exit_sensor[0], track_section, 'exit', random.uniform(40, 80))
    ]

    return events, normal_duration, is_delayed

# Generate 30 trips
all_events = []
delays_count = 0

for i in range(30):
    train_id = random.randint(1, 4)
    track = random.choice(['Алматы-1 - Алматы-2', 'Алматы-2 - Шелек'])
    minutes_ago = random.randint(1, 120)

    events, duration, is_delayed = generate_train_journey(train_id, track, minutes_ago)
    all_events.extend(events)
    if is_delayed:
        delays_count += 1

# Save to database
with engine.connect() as conn:
    for event in all_events:
        conn.execute(
            text("""
                INSERT INTO sensor_events (event_time, train_id, sensor_id, track_section, event_type, value)
                VALUES (:event_time, :train_id, :sensor_id, :track_section, :event_type, :value)
            """),
            {
                "event_time": event[0],
                "train_id": event[1],
                "sensor_id": event[2],
                "track_section": event[3],
                "event_type": event[4],
                "value": event[5]
            }
        )
    conn.commit()

print(f"✅ Generated {len(all_events)} events (30 trips)")
print(f"⚠️ Of which delays: {delays_count}")

query = """
WITH entry_times AS (
    SELECT
        train_id,
        track_section,
        event_time AS entry_time,
        value AS speed
    FROM sensor_events
    WHERE event_type = 'entry'
),
exit_times AS (
    SELECT
        train_id,
        track_section,
        event_time AS exit_time
    FROM sensor_events
    WHERE event_type = 'exit'
)
SELECT
    e.train_id,
    t.train_number,
    e.track_section,
    e.entry_time,
    x.exit_time,
    EXTRACT(EPOCH FROM (x.exit_time - e.entry_time)) AS duration_seconds,
    e.speed AS entry_speed
FROM entry_times e
JOIN exit_times x
    ON e.train_id = x.train_id AND e.track_section = x.track_section
JOIN trains t ON e.train_id = t.train_id
WHERE x.exit_time > e.entry_time
ORDER BY e.entry_time DESC
"""

df = pd.read_sql(query, engine)
print(f"📊 Analyzed {len(df)} passages\n")
df.head(10)

stats = df.groupby('track_section')['duration_seconds'].agg(['mean', 'std', 'count']).reset_index()
stats.columns = ['track_section', 'mean_seconds', 'std_seconds', 'count_trips']
stats['threshold_anomaly'] = stats['mean_seconds'] + 2 * stats['std_seconds']
stats['threshold_warning'] = stats['mean_seconds'] + stats['std_seconds']


df_analysis = df.merge(stats[['track_section', 'mean_seconds', 'threshold_warning', 'threshold_anomaly']], on='track_section')
df_analysis['status'] = '✅ Normal'


df_analysis.loc[df_analysis['duration_seconds'] > df_analysis['threshold_warning'], 'status'] = '⚠️ Warning'
df_analysis.loc[df_analysis['duration_seconds'] > df_analysis['threshold_anomaly'], 'status'] = '🔴 CRITICAL DELAY'


anomalies = df_analysis[df_analysis['status'] != '✅ Normal']
print(f"\n{'='*60}")
print(f"📊 STATISTICS:")
print(f"   Total trips: {len(df_analysis)}")
print(f"   Normal: {len(df_analysis[df_analysis['status']=='✅ Normal'])}")
print(f"   Warning: {len(df_analysis[df_analysis['status']=='⚠️ Warning'])}")
print(f"   🔴 Critical delays: {len(df_analysis[df_analysis['status']=='🔴 CRITICAL DELAY'])}")
print(f"{'='*60}\n")

if len(anomalies) > 0:
    print("⚠️ Detected problems:")
    anomalies[['train_number', 'track_section', 'duration_seconds', 'mean_seconds', 'status']]
else:
    print("✅ All trips are normal")

anomalies = df_analysis[df_analysis['status'] != '✅ Normal'].copy()

if len(anomalies) > 0:
    print("="*80)
    print("🚨 DETECTED PROBLEMS (full list):")
    print("="*80)


    anomalies['severity'] = anomalies['status'].map({'🔴 CRITICAL DELAY': 0, '⚠️ Warning': 1})
    anomalies = anomalies.sort_values('severity')


    display_cols = ['train_number', 'track_section', 'duration_seconds', 'mean_seconds', 'status']
    display(anomalies[display_cols])

    print("\n" + "="*80)
    print("📊 DETAILED STATISTICS BY ANOMALIES:")
    print("="*80)

    for section in anomalies['track_section'].unique():
        section_anomalies = anomalies[anomalies['track_section'] == section]
        print(f"\n📍 Section: {section}")
        print(f"   - Total problems: {len(section_anomalies)}")
        print(f"   - Of which critical: {len(section_anomalies[section_anomalies['status'] == '🔴 CRITICAL DELAY'])}")
        print(f"   - Average time (normal): {section_anomalies['mean_seconds'].iloc[0]:.1f} sec")
        print(f"   - Maximum delay: {section_anomalies['duration_seconds'].max():.1f} sec")
else:
    print("✅ All trips are normal, no problems detected")

print("\n" + "="*80)
print("✅ EXAMPLES OF NORMAL TRIPS (for comparison):")
print("="*80)

normal_trips = df_analysis[df_analysis['status'] == '✅ Normal'].head(10)
display(normal_trips[['train_number', 'track_section', 'duration_seconds', 'mean_seconds']])

print(f"\n📈 Brief statistics for normal trips:")
print(f"   - Minimum time: {normal_trips['duration_seconds'].min():.1f} sec")
print(f"   - Maximum time: {normal_trips['duration_seconds'].max():.1f} sec")
print(f"   - Average time: {normal_trips['duration_seconds'].mean():.1f} sec")

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(16, 12))


ax1 = axes[0, 0]
colors = {'✅ Normal': 'blue', '⚠️ Warning': 'orange', '🔴 CRITICAL DELAY': 'red'}

for status in colors.keys():
    status_data = df_analysis[df_analysis['status'] == status]
    ax1.scatter(status_data['entry_time'], status_data['duration_seconds'],
               c=colors[status], alpha=0.6, s=50, label=status,
               marker='o' if status == '✅ Normal' else '^')

ax1.axhline(y=df_analysis['mean_seconds'].mean(), color='green', linestyle='--', linewidth=2, label='Network average')
ax1.set_xlabel('Entry time to section')
ax1.set_ylabel('Travel time (seconds)')
ax1.set_title('Monitoring of KTZ section travel time')
ax1.legend()
ax1.grid(True, alpha=0.3)


ax2 = axes[0, 1]
sections = stats['track_section']
x = np.arange(len(sections))
width = 0.35

bars1 = ax2.bar(x - width/2, stats['mean_seconds'], width, label='Average time', color='green', alpha=0.7)
bars2 = ax2.bar(x + width/2, stats['threshold_anomaly'], width, label='Anomaly threshold (mean+2σ)', color='red', alpha=0.7)


for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}', ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}', ha='center', va='bottom')

ax2.set_xlabel('Section')
ax2.set_ylabel('Seconds')
ax2.set_title('Comparison of average time with anomaly threshold')
ax2.set_xticks(x)
ax2.set_xticklabels(sections, rotation=15, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')


ax3 = axes[1, 0]
for section in df_analysis['track_section'].unique():
    section_data = df_analysis[df_analysis['track_section'] == section]
    ax3.hist(section_data['duration_seconds'], alpha=0.5, label=section, bins=15)
ax3.set_xlabel('Travel time (seconds)')
ax3.set_ylabel('Number of trains')
ax3.set_title('Travel time distribution by section')
ax3.legend()
ax3.grid(True, alpha=0.3)


ax4 = axes[1, 1]
problem_counts = anomalies.groupby('track_section').size()
if len(problem_counts) > 0:
    ax4.pie(problem_counts.values, labels=problem_counts.index, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Problem distribution by section')
else:
    ax4.text(0.5, 0.5, 'No problems', ha='center', va='center', fontsize=20)
    ax4.set_title('Problem distribution by section')

plt.tight_layout()
plt.show()


print("\n" + "="*80)
print("📊 KEY FINDINGS FROM ANALYSIS RESULTS:")
print("="*80)
print(f"1. Total trains analyzed: {len(df_analysis)}")
print(f"2. Percentage of anomalies: {len(anomalies)/len(df_analysis)*100:.1f}%")
print(f"3. Critical delays: {len(anomalies[anomalies['status']=='🔴 CRITICAL DELAY'])}")
print(f"4. Most problematic section: {anomalies['track_section'].mode()[0] if len(anomalies)>0 else 'none'}")

from sqlalchemy import text
from datetime import datetime


df_analysis['detected_at'] = datetime.now()


with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS detected_anomalies (
            id SERIAL PRIMARY KEY,
            train_id INTEGER,
            track_section VARCHAR(50),
            duration_seconds FLOAT,
            mean_seconds FLOAT,
            status VARCHAR(50),
            detected_at TIMESTAMPTZ
        )
    """))
    conn.commit()
    print("✅ Table detected_anomalies created (or already exists)")


critical_to_save = df_analysis[df_analysis['status'] == '🔴 CRITICAL DELAY']

if len(critical_to_save) > 0:
    with engine.connect() as conn:
        for _, row in critical_to_save.iterrows():
            conn.execute(text("""
                INSERT INTO detected_anomalies (train_id, track_section, duration_seconds, mean_seconds, status, detected_at)
                VALUES (:train_id, :track_section, :duration_seconds, :mean_seconds, :status, :detected_at)
            """), {
                "train_id": int(row['train_id']),
                "track_section": row['track_section'],
                "duration_seconds": float(row['duration_seconds']),
                "mean_seconds": float(row['mean_seconds']),
                "status": 'critical',
                "detected_at": row['detected_at']
            })
        conn.commit()
    print(f"✅ Saved {len(critical_to_save)} anomalies to detected_anomalies table")


    print("\n📝 Saved critical delays:")
    display(critical_to_save[['train_number', 'track_section', 'duration_seconds', 'mean_seconds']])
else:
    print("ℹ️ No critical delays to save")


print("\n" + "="*60)
print("📋 Contents of detected_anomalies table:")
print("="*60)

try:
    df_anomalies_db = pd.read_sql("SELECT * FROM detected_anomalies ORDER BY detected_at DESC", engine)
    if len(df_anomalies_db) > 0:
        display(df_anomalies_db)
    else:
        print("Table is empty")
except Exception as e:
    print(f"Error reading table: {e}")
