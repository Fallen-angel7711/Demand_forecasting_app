import streamlit as st
import pandas as pd
import cloudpickle
import matplotlib.pyplot as plt

# Load trained model
with open("model.pkl", "rb") as f:
    model = cloudpickle.load(f)

st.title("🧠 AI Demand Forecasting App")
st.write("Upload your weekly store-SKU data and get demand predictions (units sold).")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    expected_cols = ['record_ID', 'store_id', 'sku_id', 'week', 'total_price', 'units_sold']
    if not all(col in df.columns for col in expected_cols):
        st.error("Uploaded CSV does not have the required columns.")
        st.stop()

    # Preprocessing
    df['week'] = pd.to_datetime(df['week'], format='%y/%m/%d')
    df['year'] = df['week'].dt.year
    df['month'] = df['week'].dt.month
    df['day_of_week'] = df['week'].dt.dayofweek
    df.drop(['record_ID', 'week', 'total_price'], axis=1, inplace=True)

    # Drop target if exists
    if 'units_sold' in df.columns:
        X = df.drop('units_sold', axis=1)
    else:
        X = df

    y_pred = model.predict(X)

    st.subheader("📊 Predicted Units Sold")
    df['predicted_units_sold'] = y_pred
    st.dataframe(df[['store_id', 'sku_id', 'predicted_units_sold']])

    st.subheader("📈 Forecast Visualization")
    plt.figure(figsize=(10, 5))
    plt.plot(y_pred, label="Predicted Units Sold", color='blue')
    plt.legend()
    st.pyplot(plt.gcf())
