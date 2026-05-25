import streamlit as st
import requests, os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("🐱 Cat vs Dog Classifier")
uploaded = st.file_uploader("Choose image", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded, width=300)
    if st.button("Predict"):
        try:
            r = requests.post(f"{API_URL}/predict", files={"file": uploaded})
            if r.status_code == 200:
                st.success(r.json()["prediction"])
            else:
                st.error("Prediction failed")
        except:
            st.error("Connection error")
