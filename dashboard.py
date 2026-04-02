import streamlit as st
import pandas as pd
from database import SessionLocal, EquipmentTelemetry
import time
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="EagleVision Analytics", layout="wide")

st.title("EagleVision: Equipment Monitoring Dashboard")
st.markdown("---")

def get_data():
    db = SessionLocal()
    data = db.query(EquipmentTelemetry).order_by(EquipmentTelemetry.db_timestamp.desc()).limit(100).all()
    db.close()
    return pd.DataFrame([vars(t) for t in data]).drop(columns=['_sa_instance_state'], errors='ignore')

# Sidebar
st.sidebar.title("Dashboard Controls")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)

# Placeholders for dynamic content
video_placeholder = st.empty()
metrics_placeholder = st.empty()
tables_placeholder = st.empty()
charts_placeholder = st.empty()

while True:
    # 1. Update Video Feed Frame
    with video_placeholder.container():
        st.subheader("Live Processed Video Feed")
        if os.path.exists("out/current_frame.jpg"):
            try:
                st.image("out/current_frame.jpg", width='stretch')
            except Exception:
                pass # Skip drawing this frame if OpenCV is actively writing to it
        else:
            st.info("Awaiting video stream...")

    # 2. Update Telemetry Data
    df = get_data()
    
    if not df.empty:
        # Group by latest entry per equipment_id
        latest_status = df.groupby('equipment_id').first()
        
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            total_active = (latest_status['current_state'] == 'ACTIVE').sum()
            total_inactive = (latest_status['current_state'] == 'INACTIVE').sum()
            avg_util = latest_status['utilization_percent'].mean()
            
            col1.metric("Active Equipment", total_active)
            col2.metric("Idle Equipment", total_inactive)
            col3.metric("Avg. Utilization", f"{round(avg_util, 1) if pd.notnull(avg_util) else 0}%")
            col4.metric("Extracted Frame Processed", df['frame_id'].max())

        with tables_placeholder.container():
            st.subheader("Live Status & Utilization Tracker")
            # Present tracking info side-by-side
            display_df = latest_status[['equipment_class', 'current_state', 'current_activity', 'total_active_seconds', 'total_idle_seconds', 'utilization_percent']].copy()
            display_df.rename(columns={
                'equipment_class': 'Class',
                'current_state': 'State', 
                'current_activity': 'Activity',
                'total_active_seconds': 'Working Time (s)',
                'total_idle_seconds': 'Idle Time (s)',
                'utilization_percent': 'Utilization (%)'
            }, inplace=True)
            st.table(display_df)

        with charts_placeholder.container():
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Utilization Trend")
                group_data = df.groupby(['equipment_id', 'id'])['utilization_percent'].mean().unstack(level=0)
                st.line_chart(group_data)

            with col_chart2:
                st.subheader("Activity Distribution")
                activity_count = df['current_activity'].value_counts()
                fig, ax = plt.subplots()
                activity_count.plot.pie(autopct='%1.1f%%', ax=ax, textprops={'color':"w"})
                fig.patch.set_alpha(0)  # transparent background
                ax.set_ylabel('')
                st.pyplot(fig)
                plt.close(fig)  # Clear figure from memory
    else:
        with metrics_placeholder.container():
            st.info("No data received yet. Run 'main.py' and the 'consumer.py' to pipe data into the dashboard.")

    time.sleep(refresh_rate)
