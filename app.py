import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("wine_model.pkl")

# Page Config
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# Header
st.title("🍷 Wine Quality Prediction System")
st.markdown(
    """
    Predict the quality of wine based on its physicochemical properties.
    Enter the wine characteristics below and click **Predict Quality**.
    """
)

st.divider()

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Chemical Properties")

    fixed_acidity = st.number_input(
        "Fixed Acidity",
        value=7.4,
        step=0.1
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity",
        value=0.70,
        step=0.01
    )

    citric_acid = st.number_input(
        "Citric Acid",
        value=0.00,
        step=0.01
    )

    residual_sugar = st.number_input(
        "Residual Sugar",
        value=1.9,
        step=0.1
    )

    chlorides = st.number_input(
        "Chlorides",
        value=0.076,
        step=0.001,
        format="%.3f"
    )

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide",
        value=11.0,
        step=1.0
    )

with col2:
    st.subheader("⚗️ Additional Properties")

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide",
        value=34.0,
        step=1.0
    )

    density = st.number_input(
        "Density",
        value=0.9978,
        step=0.0001,
        format="%.4f"
    )

    ph = st.number_input(
        "pH",
        value=3.51,
        step=0.01
    )

    sulphates = st.number_input(
        "Sulphates",
        value=0.56,
        step=0.01
    )

    alcohol = st.number_input(
        "Alcohol",
        value=9.4,
        step=0.1
    )

st.divider()

if st.button("🍷 Predict Quality", use_container_width=True):

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
    "Built with Streamlit • Machine Learning Wine Quality Prediction Project"
)