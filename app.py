import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Road Safety Hotspot AI", page_icon="🚦", layout="wide")
st.title("🚦 AI-Based Road Accident Hotspot Prediction")
st.caption("Interactive mini-project prototype | Demonstration data is synthetic, not official accident records.")

@st.cache_data
def make_demo_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(26.8, 28.9, n)
    lon = rng.uniform(77.0, 79.2, n)
    traffic = rng.integers(1, 11, n)
    road_quality = rng.integers(1, 6, n)  # 1 poor, 5 good
    visibility = rng.integers(1, 6, n)    # 1 poor, 5 good
    night = rng.integers(0, 2, n)
    weather = rng.integers(0, 3, n)       # 0 clear, 1 rain, 2 fog
    # Synthetic risk score for a demonstrable ML workflow
    score = (traffic * 0.42 + (6-road_quality) * 0.65 +
             (6-visibility) * 0.55 + night * 1.0 +
             (weather == 1) * 0.65 + (weather == 2) * 1.15 +
             rng.normal(0, 0.8, n))
    q1, q2 = np.quantile(score, [0.55, 0.82])
    risk = np.where(score >= q2, "High", np.where(score >= q1, "Medium", "Low"))
    return pd.DataFrame({
        "Latitude": lat, "Longitude": lon, "Traffic Level (1-10)": traffic,
        "Road Quality (1-5)": road_quality, "Visibility (1-5)": visibility,
        "Night (1=yes)": night, "Weather (0=clear,1=rain,2=fog)": weather,
        "Risk Level": risk
    })

df = make_demo_data()
features = ["Traffic Level (1-10)", "Road Quality (1-5)", "Visibility (1-5)", "Night (1=yes)", "Weather (0=clear,1=rain,2=fog)"]
X, y = df[features], df["Risk Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

m1, m2, m3 = st.columns(3)
m1.metric("Demo records", f"{len(df):,}")
m2.metric("High-risk demo locations", int((df["Risk Level"] == "High").sum()))
m3.metric("Test accuracy", f"{acc:.1%}")

st.warning("Academic prototype: records and coordinates are generated for demonstration. Do not use these predictions for real-world safety decisions.")

tab1, tab2, tab3 = st.tabs(["🗺️ Hotspot Map", "🔎 Predict Risk", "📊 Data & Model"])
with tab1:
    st.subheader("Illustrative road-risk map")
    risk_filter = st.multiselect("Show risk levels", ["Low", "Medium", "High"], default=["Medium", "High"])
    map_df = df[df["Risk Level"].isin(risk_filter)].sample(min(500, len(df[df["Risk Level"].isin(risk_filter)])), random_state=42)
    fig = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude", color="Risk Level",
                            hover_data=features, zoom=5, height=520,
                            mapbox_style="open-street-map",
                            color_discrete_map={"Low":"#2ca25f","Medium":"#f0ad4e","High":"#d62728"})
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.subheader("Try a sample road condition")
    c1, c2 = st.columns(2)
    with c1:
        traffic = st.slider("Traffic level", 1, 10, 7)
        road = st.slider("Road quality", 1, 5, 2, help="1 = poor, 5 = good")
        visibility = st.slider("Visibility", 1, 5, 2, help="1 = poor, 5 = good")
    with c2:
        night = st.selectbox("Time", ["Day", "Night"])
        weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"])
    row = pd.DataFrame([{
        "Traffic Level (1-10)": traffic, "Road Quality (1-5)": road,
        "Visibility (1-5)": visibility, "Night (1=yes)": int(night=="Night"),
        "Weather (0=clear,1=rain,2=fog)": {"Clear":0,"Rain":1,"Fog":2}[weather]
    }])
    if st.button("Predict Risk Level", type="primary"):
        pred = model.predict(row)[0]
        probs = model.predict_proba(row)[0]
        st.subheader(f"Estimated demo risk: {pred}")
        st.progress(float(max(probs)))
        st.write("Model confidence by class (synthetic demo):")
        st.dataframe(pd.DataFrame({"Risk level":model.classes_, "Model score":probs}).sort_values("Model score", ascending=False), use_container_width=True)
with tab3:
    st.subheader("Sample dataset")
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("Feature importance")
    imp = pd.DataFrame({"Feature":features, "Importance":model.feature_importances_}).sort_values("Importance", ascending=True)
    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Random Forest feature importance"), use_container_width=True)
    st.write(f"Test accuracy on synthetic held-out data: **{acc:.1%}**")
    st.caption("Accuracy is only a software demonstration metric; it does not represent real-world accident prediction performance.")
