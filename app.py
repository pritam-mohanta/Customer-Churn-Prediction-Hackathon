
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings

from pathlib import Path

warnings.filterwarnings("ignore")


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "DataSet" / "customer_churn_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# =====================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# 3. LOAD MODEL, SCALER AND DATASET
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    return model, scaler


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


try:
    model, scaler = load_model()
    df = load_dataset()

except Exception as e:
    st.error(f"Application loading error: {e}")
    st.stop()


# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("📂 Navigation")

app_mode = st.sidebar.radio(
    "Go to",
    [
        "📌 Project Overview",
        "🔍 Data Exploration",
        "🔮 Churn Prediction"
    ]
)


# =====================================================
# 5. PROJECT OVERVIEW
# =====================================================

if app_mode == "📌 Project Overview":

    st.title("📌 Customer Churn Prediction")
    st.subheader("Machine Learning Based Customer Churn Analysis")

    st.markdown("""
    ### Problem Statement

    Predict whether a customer will **Churn** or **Not Churn**
    based on the following customer attributes:

    - Age
    - Gender
    - Tenure
    - Monthly Charges

    The objective is to identify potential churn customers
    and help businesses improve customer retention.
    """)

    st.divider()

    st.markdown("### 🤖 Machine Learning Models and Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 1. Logistic Regression")

        st.write("Accuracy: `0.875`")
        st.write("Precision: `0.889`")
        st.write("Recall: `0.994`")
        st.write("F1-Score: `0.938`")

        st.divider()

        st.markdown("#### 2. Support Vector Classifier (SVC)")

        st.success("Selected Final Model")

        st.write("Accuracy: `0.885`")
        st.write("Precision: `0.885`")
        st.write("Recall: `1.000`")
        st.write("F1-Score: `0.939`")
        st.write("Best Parameters: `C=0.01, kernel=linear`")

    with col2:

        st.markdown("#### 3. Decision Tree")

        st.write("Accuracy: `0.855`")
        st.write("Precision: `0.9157`")
        st.write("Recall: `0.9209`")
        st.write("F1-Score: `0.9183`")

        st.write(
            "Best Parameters: `criterion=entropy, "
            "max_depth=None, min_samples_leaf=4, "
            "min_samples_split=2, splitter=random`"
        )

        st.divider()

        st.markdown("#### 4. Random Forest")

        st.write("Accuracy: `0.860`")
        st.write("Precision: `0.890`")
        st.write("Recall: `0.960`")
        st.write("F1-Score: `0.9239`")

        st.write(
            "Best Parameters: `bootstrap=True, "
            "max_features=2, n_estimators=64`"
        )

    st.divider()

    st.markdown("### Model Performance Comparison")

    performance_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "SVC",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [0.875, 0.885, 0.855, 0.860],
        "Precision": [0.889, 0.885, 0.9157, 0.890],
        "Recall": [0.994, 1.000, 0.9209, 0.960],
        "F1-Score": [0.938, 0.939, 0.9183, 0.9239]
    })

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    ### Final Model Selection

    **Support Vector Classifier (SVC)** is selected as the
    final model because it provides the highest recall and
    F1-score among the evaluated models.

    High recall is particularly important for identifying
    potential churn customers.
    """)


# =====================================================
# 6. DATA EXPLORATION
# =====================================================

elif app_mode == "🔍 Data Exploration":

    st.title("🔍 Data Exploration")

    st.markdown(
        "Explore customer information, churn patterns, "
        "and relationships between numerical and categorical variables."
    )

    # ---------------------------------------------
    # Correlation Heatmap
    # ---------------------------------------------

    st.markdown("### 🔥 Correlation Heatmap")

    corr_df = df.select_dtypes(
        include=["number"]
    ).corr()

    fig_corr = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Between Numerical Features"
    )

    fig_corr.update_layout(height=500)

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.info(
        "'Tenure' and 'TotalCharges' are highly correlated. "
        "'TotalCharges' was excluded from the model."
    )

    st.divider()

    # ---------------------------------------------
    # Churn Distribution
    # ---------------------------------------------

    st.markdown("### 🥧 Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    fig_pie = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        color=churn_counts.index,
        color_discrete_map={
            "Yes": "salmon",
            "No": "skyblue"
        },
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    st.divider()

    # ---------------------------------------------
    # Gender and Monthly Charges vs Churn
    # ---------------------------------------------

    st.markdown("### 👥 Gender and Monthly Charges vs Churn")

    grouped_data = (
        df.groupby(
            ["Churn", "Gender"]
        )["MonthlyCharges"]
        .mean()
        .reset_index()
    )

    st.dataframe(
        grouped_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------------------
    # Average Monthly Charges by Contract Type
    # ---------------------------------------------

    st.markdown("### 📊 Average Monthly Charges by Contract Type")

    avg_by_contract = (
        df.groupby("ContractType")["MonthlyCharges"]
        .mean()
        .reset_index()
    )

    fig_bar = px.bar(
        avg_by_contract,
        x="ContractType",
        y="MonthlyCharges",
        title="Average Monthly Charges by Contract Type",
        color="MonthlyCharges",
        color_continuous_scale="Teal"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    st.divider()

    # ---------------------------------------------
    # Monthly Charges Histogram
    # ---------------------------------------------

    st.markdown("### 💵 Distribution of Monthly Charges")

    fig_hist_mc = px.histogram(
        df,
        x="MonthlyCharges",
        nbins=30,
        title="Monthly Charges Distribution",
        color_discrete_sequence=["orange"]
    )

    fig_hist_mc.update_traces(
        marker_line_color="black",
        marker_line_width=1
    )

    st.plotly_chart(
        fig_hist_mc,
        use_container_width=True
    )

    st.divider()

    # ---------------------------------------------
    # Tenure Histogram
    # ---------------------------------------------

    st.markdown("### ⌛ Distribution of Tenure")

    fig_hist_tenure = px.histogram(
        df,
        x="Tenure",
        nbins=30,
        title="Customer Tenure Distribution",
        color_discrete_sequence=["purple"]
    )

    fig_hist_tenure.update_traces(
        marker_line_color="black",
        marker_line_width=1
    )

    st.plotly_chart(
        fig_hist_tenure,
        use_container_width=True
    )


# =====================================================
# 7. CUSTOMER CHURN PREDICTION
# =====================================================

elif app_mode == "🔮 Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.markdown("""
    Enter the customer details below and click the Predict
    button to identify the customer's churn status.
    """)

    st.divider()

    # ---------------------------------------------
    # Customer Input Fields
    # ---------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Enter Age",
            min_value=10,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Select Gender",
            ["Male", "Female"]
        )

    with col2:

        tenure = st.number_input(
            "Enter Tenure",
            min_value=0,
            max_value=130,
            value=10
        )

        monthly_charge = st.number_input(
            "Enter Monthly Charge",
            min_value=30,
            max_value=150,
            value=30
        )

    gender_selected = 1 if gender == "Female" else 0

    st.divider()

    # ---------------------------------------------
    # Prediction Button
    # ---------------------------------------------

    if st.button(
        "🚀 Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing customer information..."):

            try:

                # Prepare input features
                x = np.array([
                    age,
                    gender_selected,
                    tenure,
                    monthly_charge
                ]).reshape(1, -1)

                # Apply feature scaling
                x_scaled = scaler.transform(x)

                # Generate prediction
                prediction = model.predict(x_scaled)[0]

                st.divider()

                if prediction == 1:

                    st.error(
                        "Prediction Result: Customer is likely to CHURN."
                    )

                    st.warning(
                        "This customer may require a retention strategy."
                    )

                else:

                    st.success(
                        "Prediction Result: Customer is likely NOT TO CHURN."
                    )

                    st.info(
                        "This customer is predicted to remain with the service."
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed. Please check the model input format. Error: {e}"
                )


# =====================================================
# 8. FOOTER
# =====================================================

st.sidebar.divider()

st.sidebar.caption(
    "Customer Churn Prediction Dashboard"
)

st.sidebar.caption(
    "Python | Machine Learning | Streamlit"
)

