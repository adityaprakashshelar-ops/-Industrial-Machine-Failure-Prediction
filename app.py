import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

st.title("Industrial Machine Failure Risk Predictor")

ptype = st.selectbox("Product Type", ["L", "M", "H"])
air_temp = st.number_input("Air temperature [K]", 295.0, 305.0, 300.0)
process_temp = st.number_input("Process temperature [K]", 305.0, 315.0, 310.0)
speed = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500)
torque = st.number_input("Torque [Nm]", 0.0, 80.0, 40.0)
tool_wear = st.number_input("Tool wear [min]", 0, 260, 100)

if st.button("Predict"):
    row = pd.DataFrame([{                       # RAW row — pipeline handles encoding
        "Type": ptype,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])
    row = pd.get_dummies(row, columns=["Type"])            # encode, NO drop_first
    row = row.reindex(columns=model.feature_names_in_, fill_value=0)  # match training cols

    proba = model.predict_proba(row)[0, list(model.classes_).index(1)]
    st.metric("Failure risk", f"{proba:.1%}")
    st.write("⚠️ Likely failure" if proba > 0.5 else "✅ Normal operation")
