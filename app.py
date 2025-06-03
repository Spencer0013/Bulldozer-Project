import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Add src to Python path so utils can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import extract_date_features, build_preprocessor

# Streamlit page config
st.set_page_config(page_title="Bulldozer Price Prediction", page_icon="🚜")
st.title("🚜 Bulldozer Price Prediction")
st.write("""
Upload a CSV file containing raw bulldozer features (including the 'saledate' column), and get price predictions.
""")

# File uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type="csv")

if not uploaded_file:
    st.info("⬅️ Upload a CSV file to start predictions.")
    st.stop()

# Load uploaded CSV
try:
    df = pd.read_csv(uploaded_file, parse_dates=["saledate"], low_memory=False)
except Exception as e:
    st.error(f"❌ Failed to read CSV: {e}")
    st.stop()

# Extract date features
try:
    df = extract_date_features(df)
    df.drop(columns=["saledate"], inplace=True, errors="ignore")
except Exception as e:
    st.error(f"❌ Error processing 'saledate': {e}")
    st.stop()

# Load model
try:
    pipeline = joblib.load("artifacts/model.pkl")
except Exception as e:
    st.error(f"❌ Could not load model pipeline: {e}")
    st.stop()

# Apply preprocessing and predict
try:
    preprocessor = build_preprocessor(df)
    X_transformed = preprocessor.fit_transform(df)  # or .transform() if columns are fixed
    preds = pipeline.predict(X_transformed)

    df["PredictedPrice"] = preds
    st.success("✅ Predictions generated successfully!")
    st.dataframe(df[["PredictedPrice"] + df.columns[:5].tolist()])

    # Download predictions
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

except Exception as e:
    st.error(f"❌ Prediction failed: {e}")



