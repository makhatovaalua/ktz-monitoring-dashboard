import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np

# ========== ПОДКЛЮЧЕНИЕ К SUPABASE ==========

DB_HOST = "aws-1-us-east-1.pooler.supabase.com"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres.qnyrcvrnhtzzkypsrfls"
DB_PASSWORD = "qomkag-3xyPca-gupder"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

@st.cache_resource
def init_connection():
    return create_engine(DATABASE_URL)

engine = init_connection()

# ========== ЗАГРУЗКА ДАННЫХ ==========
@st.cache_data(ttl=30)
def load_data():
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
           x.exit_time, EXTRACT(EPOCH FROM (x.exit_time - e.entry_time)) AS duration_seconds
    FROM entry_times e
    JOIN exit_times x ON e.train_id = x.train_id AND e.track_section = x.track_section
    WHERE x.exit_time > e.entry_time
    ORDER BY e.entry_time DESC
    """
    return pd.read_sql(query, engine)

# ========== ИНТЕРФЕЙС ==========
st.set_page_config(page_title="КТЖ Мониторинг", page_icon="🚂", layout="wide")
st.title("🚂 Система мониторинга задержек КТЖ")

with st.spinner("Загрузка данных..."):
    df = load_data()

st.metric("Всего поездов", len(df))
st.dataframe(df.head(10))
