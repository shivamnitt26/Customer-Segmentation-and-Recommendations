# --- Import Necessary Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Helper Function to Load Models ---
@st.cache_data
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# --- Load All Required Files ---
try:
    kmeans_model = load_model('kmeans_model.pkl')
    scaler = load_model('scaler.pkl')
    segment_map = load_model('segment_map.pkl')
    item_sim_df = load_model('item_similarity_matrix.pkl')
    description_to_stockcode = load_model('description_to_stockcode.pkl')
    stockcode_to_description = {v: k for k, v in description_to_stockcode.items()}
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}. Please ensure all .pkl files are in the 'My Project' folder.")
    st.stop()

# --- Recommendation Function ---
def get_product_recommendations(description, top_n=5):
    try:
        stock_code = description_to_stockcode[description]
        sim_scores = item_sim_df[stock_code].sort_values(ascending=False)
        top_products_stockcodes = sim_scores.iloc[1:top_n+1].index
        recommendations = [stockcode_to_description.get(code, "Unknown Product") for code in top_products_stockcodes]
        return recommendations
    except KeyError:
        return ["Product not found in the dataset."]
    except Exception as e:
        return [f"An error occurred: {e}"]

# --- Segmentation Prediction Function ---
def predict_segment(recency, frequency, monetary):
    try:
        input_data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})
        input_log = np.log1p(input_data)
        input_scaled = scaler.transform(input_log)
        cluster_label = kmeans_model.predict(input_scaled)[0]
        segment = segment_map.get(cluster_label, "Unknown Segment")
        return segment
    except Exception as e:
        return f"Error in prediction: {e}"

# --- Streamlit App UI ---
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛒 Shopper Spectrum: Customer Segmentation & Recommendations")
st.markdown("This application helps analyze customer behavior and recommend products.")

col1, col2 = st.columns(2)

with col1:
    st.header("🎯 Product Recommendation")
    product_list = list(description_to_stockcode.index)
    selected_product = st.selectbox("Select a Product", options=product_list, index=100)
    if st.button("Get Recommendations"):
        if selected_product:
            recommendations = get_product_recommendations(selected_product)
            st.subheader("Recommended Products:")
            for i, product in enumerate(recommendations):
                st.success(f"{i+1}. {product}")

with col2:
    st.header("🔍 Customer Segmentation")
    recency = st.number_input("Recency (days)", min_value=1, value=30)
    frequency = st.number_input("Frequency (purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend $)", min_value=1.0, value=500.0)
    if st.button("Predict Cluster"):
        segment = predict_segment(recency, frequency, monetary)
        st.subheader("Predicted Customer Segment:")
        if "Error" in segment:
            st.error(segment)
        elif segment == "High-Value":
            st.success(f"**{segment}** - This is a top customer! 🎉")
        elif segment == "Regular":
            st.info(f"**{segment}** - This is a loyal, consistent customer.")
        elif segment == "Occasional":
            st.warning(f"**{segment}** - This customer makes infrequent purchases.")
        elif segment == "At-Risk":
            st.error(f"**{segment}** - This customer is at risk. 😟")

st.markdown("---")