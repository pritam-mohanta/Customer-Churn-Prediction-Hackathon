# Customer Churn Prediction Dashboard

**Python | Machine Learning | SVC | Streamlit | Data Analytics**

An interactive machine learning application that predicts customer churn risk using a Support Vector Classifier (SVC) model with **88.5% accuracy**.

The application provides real-time churn predictions, interactive data exploration, visualization, and machine learning model comparisons through a user-friendly Streamlit dashboard.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Business Problem](#business-problem)
- [Project Objectives](#project-objectives)
- [Project Structure](#project-structure)
- [Project Workflow](#project-workflow)
- [Key Features](#key-features)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance](#model-performance)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Dashboard Screenshots](#dashboard-screenshots)
- [Technical Implementation](#technical-implementation)
- [Installation and Setup](#installation-and-setup)
- [Deployment](#deployment)
- [Tools and Technologies](#tools-and-technologies)
- [Skills Demonstrated](#skills-demonstrated)
- [Author](#author)

---

## About the Project

This project focuses on predicting customer churn using machine learning techniques and developing an interactive web application with Streamlit.

Customer churn prediction helps businesses identify customers who are likely to discontinue their services. By identifying potential churn risks, businesses can take proactive steps to improve customer retention.

The application uses a trained Support Vector Classifier (SVC) model to generate churn predictions and provides interactive tools for exploring customer data and understanding model performance.

<badge color="success">88.5% Model Accuracy</badge>

**Live Application:** [Open Customer Churn Prediction Dashboard](https://customer-churn-prediction-hackathon-ldij34jss6ya74h5g8lhwq.streamlit.app/)

---

## Business Problem

Customer churn is a major challenge for businesses because losing existing customers can affect revenue, profitability, and long-term growth.

The objective of this project is to build a machine learning solution that helps businesses:

- Identify customers who are likely to leave.
- Predict customer churn using customer-related features.
- Understand customer behavior through data visualization.
- Compare different machine learning algorithms.
- Support data-driven customer retention strategies.

---

## Project Objectives

The primary objectives of this project are:

- Perform exploratory data analysis on customer churn data.
- Understand relationships between customer features and churn.
- Visualize customer distribution and behavioral patterns.
- Train and compare multiple machine learning algorithms.
- Select an appropriate model based on performance.
- Build an interactive churn prediction application.
- Deploy the application using Streamlit Community Cloud.

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── ScreenShots/
│   ├── Screenshot (46).png
│   └── Screenshot (49).png
│
├── app.py
│
├── model.pkl
│
├── scaler.pkl
│
├── customer_churn_data.csv
│
├── requirements.txt
│
└── README.md
```

---

## Project Workflow

### 1. Data Collection

- Imported the customer churn dataset.
- Reviewed the dataset structure and available features.
- Identified relevant customer attributes for analysis.

### 2. Exploratory Data Analysis

Performed exploratory data analysis to understand customer behavior and data patterns.

Analysis included:

- Churn distribution.
- Correlation analysis.
- Monthly charges distribution.
- Customer tenure distribution.
- Relationships between customer features.

### 3. Data Preprocessing

- Prepared the dataset for machine learning.
- Applied feature scaling using a trained scaler.
- Prepared input features for model training and prediction.

### 4. Model Development

Trained and evaluated multiple machine learning algorithms to compare their performance.

### 5. Model Selection

Selected the Support Vector Classifier (SVC) model based on the reported accuracy of **88.5%**.

### 6. Model Serialization

Saved the trained model and feature scaler using Joblib for reuse in the Streamlit application.

### 7. Streamlit Application Development

Developed an interactive web application with:

- Overview section.
- Data exploration section.
- Prediction interface.
- Interactive visualizations.
- Model performance comparison.

### 8. Deployment

Deployed the application using Streamlit Community Cloud to make the prediction dashboard accessible online.

---

## Key Features

### 1. Predictive Analytics

- Predict customer churn using the trained SVC model.
- Generate predictions through an interactive interface.
- Provide churn probability information.

### 2. Interactive Data Visualization

The dashboard provides multiple visualizations to explore customer data:

- Correlation heatmap.
- Churn distribution pie chart.
- Monthly charges histogram.
- Customer tenure histogram.

### 3. Model Comparison

Compare the performance of four machine learning algorithms using evaluation metrics.

### 4. User-Friendly Interface

Developed an interactive Streamlit application with simple navigation and an easy-to-use prediction interface.

---

## Machine Learning Models

The project compares four machine learning algorithms:

- Support Vector Classifier (SVC)
- Logistic Regression
- Decision Tree
- Fourth algorithm used in the model comparison

The SVC model was selected for the final application based on its reported accuracy of 88.5%.

---

## Model Performance

The following performance metrics were reported:

| Evaluation Metric | SVC (Selected) | Logistic Regression | Decision Tree |
|---|---:|---:|---:|
| Accuracy | 0.885 | 0.875 | 0.855 |
| F1-Score | 0.939 | 0.938 | 0.918 |

**Selected Model:** Support Vector Classifier (SVC)

**Accuracy:** 88.5%

---

## Streamlit Dashboard

The application provides three main sections.

### 1. Overview

Provides an introduction to the customer churn prediction application and its purpose.

### 2. Data Exploration

Allows users to explore customer data through interactive visualizations, including:

- Correlation heatmaps.
- Customer churn distribution.
- Monthly charges distribution.
- Customer tenure distribution.

### 3. Prediction Interface

Allows users to provide customer information and generate churn predictions using the trained SVC model.

---

## Dashboard Screenshots

### 1. Data Exploration

![Data Exploration Dashboard](ScreenShots/Screenshot%20(46).png)

Provides interactive customer data exploration through visualizations such as correlation heatmaps, churn distribution, monthly charges, and customer tenure.

---

### 2. Prediction Interface

![Prediction Interface](ScreenShots/Screenshot%20(49).png)

Provides an interactive interface for entering customer information and generating churn predictions using the trained machine learning model.

---

## Technical Implementation

### Machine Learning Model

The trained SVC model is loaded using Joblib.

```python
import joblib

# Load trained machine learning model
model = joblib.load('model.pkl')

# Load feature scaler
scaler = joblib.load('scaler.pkl')
```

### Streamlit Navigation

The application uses Streamlit components to provide interactive navigation.

```python
import streamlit as st

st.sidebar.radio(
    "Navigation",
    ["Overview", "Data", "Prediction"]
)
```

### Interactive Visualization

Plotly is used to create interactive visualizations.

```python
import plotly.express as px

fig = px.imshow(corr_matrix)

st.plotly_chart(fig)
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/VishalPawar3696/customer-churn-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd customer-churn-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

**Live Dashboard:**

[Customer Churn Prediction Dashboard](https://customer-churn-prediction-hackathon-ldij34jss6ya74h5g8lhwq.streamlit.app/)

To run the application locally with a custom port:

```bash
python -m streamlit run app.py --server.port 8502
```

---

## Tools and Technologies

- Python
- Machine Learning
- Support Vector Classifier (SVC)
- Logistic Regression
- Decision Tree
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- Git and GitHub

---

## Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Data Cleaning and Preprocessing
- Feature Scaling
- Machine Learning Model Development
- Classification Algorithms
- Support Vector Machines
- Model Evaluation
- Accuracy and F1-Score Analysis
- Model Comparison
- Model Serialization
- Interactive Data Visualization
- Streamlit Application Development
- Real-Time Prediction Interface
- Application Deployment
- Git and GitHub

---

## Author

**Pritam Mohanta**

Data Analytics | Python | Machine Learning | Power BI | Streamlit

**GitHub:** [pritam-mohanta](https://github.com/pritam-mohanta)

**Email:** samarthmohanta7001@gmail.com

---

**Customer Churn Prediction Dashboard**

Machine Learning | Predictive Analytics | Interactive Web Application
