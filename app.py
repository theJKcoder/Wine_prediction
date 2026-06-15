import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("wine_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# Header
st.title("🍷 Wine Quality Prediction")
st.markdown(
    """
    Predict wine quality using Machine Learning.
    
    Enter the most important wine characteristics below and click **Predict Quality**.
    """
)

st.divider()

# Two Columns
col1, col2 = st.columns(2)

with col1:

    st.subheader("🍇 Chemical Properties")

    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        value=9.4,
        step=0.1
    )

    sulphates = st.number_input(
        "Sulphates",
        min_value=0.0,
        value=0.56,
        step=0.01
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity",
        min_value=0.0,
        value=0.70,
        step=0.01
    )

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide",
        min_value=0.0,
        value=34.0,
        step=1.0
    )

with col2:

    st.subheader("⚗️ Additional Properties")

    density = st.number_input(
        "Density",
        min_value=0.0,
        value=0.9978,
        format="%.4f"
    )

    chlorides = st.number_input(
        "Chlorides",
        min_value=0.0,
        value=0.076,
        format="%.3f"
    )

    ph = st.number_input(
        "pH",
        min_value=0.0,
        value=3.51,
        step=0.01
    )

st.divider()

if st.button("🍷 Predict Quality", use_container_width=True):

    # Default values for hidden features
    fixed_acidity = 7.4
    citric_acid = 0.00
    residual_sugar = 1.9
    free_sulfur_dioxide = 11.0

    # Create dataframe in EXACT training order
    input_data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]], columns=[
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ])

    prediction = model.predict(input_data)[0]

    st.divider()

    st.metric(
        label="Predicted Wine Quality",
        value=str(prediction)
    )

    if prediction >= 7:
        st.success(
            f"🌟 Excellent Quality Wine (Quality Score: {prediction})"
        )

    elif prediction >= 6:
        st.info(
            f"🍷 Good Quality Wine (Quality Score: {prediction})"
        )

    else:
        st.warning(
            f"⚠️ Average Quality Wine (Quality Score: {prediction})"
        )

st.divider()

st.caption(
    "Built with Streamlit • Wine Quality Prediction using Machine Learning"
)