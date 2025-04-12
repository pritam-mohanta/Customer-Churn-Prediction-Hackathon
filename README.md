
# 🚀 Customer Churn Prediction Dashboard


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-hackathon-ldij34jss6ya74h5g8lhwq.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/Machine_Learning-SVC_88.5%25-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive machine learning application that predicts customer churn risk with **88.5% accuracy**, featuring comprehensive data exploration tools and real-time predictions.

## 🌟 Key Features
- **🔮 Predictive Analytics**: Instant churn probability using optimized SVC model
- **📊 Interactive Visualizations**: 
  - Correlation heatmaps
  - Churn distribution pie charts
  - Monthly charges/tenure histograms
- **📝 Model Comparison**: Performance metrics for 4 ML alg
- orithms
- **🎛️ User-Friendly UI**: Streamlit-powered intuitive interface

## 🖥️ Screenshots
| Data Exploration | Prediction Interface |
|------------------|----------------------|
| <img src="Screenshot (46).png" width="400"> | <img src="Screenshot (49).png" width="400"> |

#🛠️ Technical Implementation

``python


#Core ML Components


model = joblib.load('model.pkl')  # Trained SVC


scaler = joblib.load('scaler.pkl')  # Feature normalization

# Streamlit UI Components


st.sidebar.radio("Navigation", ["Overview", "Data", "Prediction"])


plotly.express.imshow(corr_matrix)  # Interactive heatmap


⚙️ Installation

# Clone repository


git clone https://github.com/VishalPawar3696/customer-churn-prediction.git

# Install dependencies


pip install -r requirements.txt

# Launch application


streamlit run app.py

📂 Project Structure

.
├── app.py                  # Main application logic


├── model.pkl               # Serialized SVC model


├── scaler.pkl              # Feature scaler


├── customer_churn_data.csv  # Sample dataset


├── requirements.txt        # Dependencies


└── screenshots/            # Documentation assets


📊 Model Performance Metrics


Metric	  SVC (Selected)	Logistic Regression	Decision Tree


Accuracy	0.885	          0.875	               0.855


Accuracy	0.885	          0.875	               0.855


F1-Score	0.939         	0.938	                0.918

🚀 Deployment Options

python -m streamlit run app.py --server.port(http://localhost:8502/)


📬 Contact:


 📧 vishalpawar3696@gmail.com
 Github: VishalPawar3696  

📧 samarthmohanta7001@gmail.com  
Github: pritam-mohanta
 
 












