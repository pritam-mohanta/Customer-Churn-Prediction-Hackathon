import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')
df = pd.read_csv("customer_churn_data.csv")

# Set Streamlit layout
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Sidebar navigation
st.sidebar.title("📂 Navigation")
app_mode = st.sidebar.radio("Go to", ["📌 Project Overview", "🔍 Data Exploration", "🔮 Churn Prediction"])

# -------------------------------------
# 📌 Project Overview
# -------------------------------------
if app_mode == "📌 Project Overview":
    st.title("📌 Project Overview")
    
    st.markdown("""
    ### 🧠 Problem Statement:
    Predict whether a customer will **Churn** or **Not Churn** based on:
    - Age
    - Gender
    - Tenure
    - Monthly Charges

    ---

    ### 🤖 ML Models Used & Their Performances

    **1️⃣ Logistic Regression**
    - Accuracy: `0.875`
    - Precision: `0.889`
    - Recall: `0.994`
    - F1-Score: `0.938`

    **2️⃣ Support Vector Classifier (SVC)** ✅ *(Final Model)*
    - Accuracy: `0.885`
    - Precision: `0.885`
    - Recall: `1.000`
    - F1-Score: `0.939`
    - Best Params: `C=0.01`, `kernel=linear`

    **3️⃣ Decision Tree**
    - Accuracy: `0.855`
    - Precision: `0.9157`
    - Recall: `0.9209`
    - F1-Score: `0.9183`
    - Best Params: `criterion=entropy`, `max_depth=None`, `min_samples_leaf=4`, `min_samples_split=2`, `splitter=random`

    **4️⃣ Random Forest**
    - Accuracy: `0.860`
    - Precision: `0.890`
    - Recall: `0.960`
    - F1-Score: `0.9239`
    - Best Params: `bootstrap=True`, `max_features=2`, `n_estimators=64`

    ---

    ### ✅ Final Model Selection:
    **SVC** is selected as it provides the highest **Recall** and **F1-score**, which is critical for identifying potential churn customers.
    """)

# -------------------------------------
# 🔍 Data Exploration
# -------------------------------------
elif app_mode == "🔍 Data Exploration":
    st.title("🔍 Data Exploration")

    st.markdown("### 🔥 Correlation Heatmap")
    corr_df = df.select_dtypes(include=['number']).corr()
    fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r')
    fig_corr.update_layout(width=800, height=500)
    st.plotly_chart(fig_corr, use_container_width=False)
    st.info("📌 'Tenure' and 'TotalCharges' are highly correlated → 'TotalCharges' excluded from model.")

    st.markdown("### 🥧 Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    fig_pie = px.pie(values=churn_counts.values, names=churn_counts.index,
                     color=churn_counts.index,
                     color_discrete_map={'Yes': 'salmon', 'No': 'skyblue'},
                     title="Churn (Yes/No)")
    fig_pie.update_layout(width=800, height=500)
    st.plotly_chart(fig_pie, use_container_width=False)

    st.markdown("### 👥 Gender & Monthly Charges vs Churn")
    grouped_data = df.groupby(['Churn', 'Gender'])['MonthlyCharges'].mean().reset_index()
    st.dataframe(grouped_data)

    st.markdown("### 📊 Avg Monthly Charges by Contract Type")
    avg_by_contract = df.groupby('ContractType')['MonthlyCharges'].mean().reset_index()
    fig_bar = px.bar(avg_by_contract, x='ContractType', y='MonthlyCharges',
                     title='Avg Monthly Charges by Contract Type',
                     color='MonthlyCharges', color_continuous_scale='Teal')
    fig_bar.update_layout(width=800, height=500)
    st.plotly_chart(fig_bar, use_container_width=False)

    st.markdown("### 💵 Histogram of Monthly Charges")
    fig_hist_mc = px.histogram(df, x='MonthlyCharges', nbins=30,
                               title="Distribution of Monthly Charges",
                               color_discrete_sequence=['orange'])
    fig_hist_mc.update_traces(marker_line_color='black', marker_line_width=1)
    fig_hist_mc.update_layout(width=800, height=500)
    st.plotly_chart(fig_hist_mc, use_container_width=False)

    st.markdown("### ⌛ Histogram of Tenure")
    fig_hist_tenure = px.histogram(df, x='Tenure', nbins=30,
                                   title="Distribution of Tenure",
                                   color_discrete_sequence=['purple'])
    fig_hist_tenure.update_traces(marker_line_color='black', marker_line_width=1)
    fig_hist_tenure.update_layout(width=800, height=500)
    st.plotly_chart(fig_hist_tenure, use_container_width=False)

# -------------------------------------
# 🔮 Churn Prediction
# -------------------------------------
elif app_mode == "🔮 Churn Prediction":
    st.title("🔮 Customer Churn Prediction")

    st.markdown("#### 📥 Please enter the values and hit the Predict Button to get the prediction")

    # Input fields
    age = st.number_input('Enter Age', min_value=10, max_value=100, value=30)
    gender = st.selectbox('Select Gender', ['Male', 'Female'])
    tenure = st.number_input('Enter Tenure', min_value=0, max_value=130, value=10)
    monthly_charge = st.number_input('Enter Monthly Charge', min_value=30, max_value=150, value=30)

    gender_selected = 1 if gender == 'Female' else 0

    if st.button("🚀 Predict"):
        with st.spinner('🔍 Analyzing... Getting Prediction...'):
            x = [age, gender_selected, tenure, monthly_charge]
            x_array = scaler.transform([np.array(x)])
            prediction = model.predict(x_array)[0]
            result = 'Churn' if prediction == 1 else 'Not Churn'
            st.success(f"🎯 Prediction Result: Customer will **{result}**")
