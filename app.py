
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AI Network Anomaly Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ AI Network Anomaly Detector")
st.write(
    "This machine-learning project identifies unusual patterns "
    "in simulated network traffic using Isolation Forest."
)

# Generate simulated typical network activity
np.random.seed(42)
n = 500

normal_data = pd.DataFrame({
    "traffic_volume": np.random.normal(500, 100, n).clip(50, 1000),
    "connection_duration": np.random.normal(30, 8, n).clip(1, 60),
    "packet_count": np.random.normal(300, 60, n).clip(20, 600),
    "failed_connections": np.random.poisson(1, n)
})

features = [
    "traffic_volume",
    "connection_duration",
    "packet_count",
    "failed_connections"
]

# Train the model on typical simulated activity
model = IsolationForest(
    contamination=0.05,
    random_state=42
)
model.fit(normal_data[features])

st.subheader("Test network activity")
st.write("Adjust the values, then click Analyze.")

with st.form("network_test"):
    traffic = st.number_input(
        "Traffic volume", min_value=0, max_value=10000, value=500
    )
    duration = st.number_input(
        "Connection duration", min_value=0, max_value=3600, value=30
    )
    packets = st.number_input(
        "Packet count", min_value=0, max_value=10000, value=300
    )
    failed = st.number_input(
        "Failed connections", min_value=0, max_value=1000, value=1
    )
    submitted = st.form_submit_button("Analyze Network Activity")

if submitted:
    sample = pd.DataFrame([{
        "traffic_volume": traffic,
        "connection_duration": duration,
        "packet_count": packets,
        "failed_connections": failed
    }])

    result = model.predict(sample[features])[0]

    if result == -1:
        st.warning("Unusual network activity detected!")
    else:
        st.success("Network activity appears typical.")

    st.caption(
        "This is an educational demonstration using simulated data, "
        "not a validated real-world cyberattack detector."
    )

st.divider()
st.subheader("Explore the simulated data")

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(
    normal_data["traffic_volume"],
    normal_data["packet_count"],
    alpha=0.6
)
ax.set_xlabel("Traffic Volume")
ax.set_ylabel("Packet Count")
ax.set_title("Simulated Typical Network Activity")
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.subheader("How it works")
st.markdown("""
1. **Data generation:** Creates 500 simulated examples of typical network activity.
2. **Machine learning:** Trains an Isolation Forest anomaly-detection model.
3. **Interactive testing:** Lets users enter network measurements and receive a prediction.
4. **Visualization:** Displays a graph of the simulated data.
""")

st.caption("Created as a student project exploring AI and networking.")
