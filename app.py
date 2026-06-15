import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("wine_model.pkl")

# Page config
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷"
)

# Title
st.title("🍷 Wine Quality Prediction")
st.markdown(
    """
    Predict wine quality using Machine Learning.

    Enter the 5 most important wine characteristics and click **Predict Quality**.
    """
)

st.divider()

# User inputs (Top 5 important features)
alcohol = st.number_input(
    "Alcohol",
    min_value=0.0,
    value=10.5,
    step=0.1
)

sulphates = st.number_input(
    "Sulphates",
    min_value=0.0,
    value=0.65,
    step=0.01
)

volatile_acidity = st.number_input(
    "Volatile Acidity",
    min_value=0.0,
    value=0.53,
    step=0.01
)

total_sulfur_dioxide = st.number_input(
    "Total Sulfur Dioxide",
    min_value=0.0,
    value=46.0,
    step=1.0
)

density = st.number_input(
    "Density",
    min_value=0.0,
    value=0.9967,
    format="%.4f"
)

st.divider()

if st.button("🍷 Predict Quality", use_container_width=True):

    # Default average values for remaining features
    fixed_acidity = 8.3
    citric_acid = 0.27
    residual_sugar = 2.5
    chlorides = 0.087
    free_sulfur_dioxide = 15.9
    ph = 3.31

    # IMPORTANT:
    # Feature order must match training order exactly
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
    "Machine Learning Project • Wine Quality Prediction"
)