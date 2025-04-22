# 🪙 Cryptocurrency Liquidity Prediction for Market Stability

## 📌 Overview
This project focuses on predicting liquidity in the cryptocurrency market to enhance market stability and inform trading decisions.

## ✨ Features
- **Data Collection**: Utilized APIs and data scraping techniques to gather real-time and historical cryptocurrency market data.
- **Preprocessing**: Cleaned and transformed data, handled missing values, and normalized features.
- **Modeling**: Implemented machine learning models (e.g., LSTM, Random Forest) to forecast liquidity based on historical data.
- **Evaluation**: Assessed model performance using metrics such as MAE, RMSE, and R² score.
- **Deployment**: Explored deployment strategies for real-time predictions and system integration.

## 🛠️ Technologies Used
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn  
- **Tools & APIs**: Binance/CoinGecko API, Jupyter Notebook, Google Colab

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/asaher84/crypto-liquidity-prediction.git
   cd crypto-liquidity-prediction


Install dependencies:
        pip install -r requirements.txt

Fetch data:
        Run data_collection.py to fetch historical market data from the chosen API.

Preprocess data:
        Execute preprocessing.ipynb to clean, normalize, and structure the data.

Train the model:
        Use model_training.ipynb to train and validate prediction models.

View results:
        Analyze prediction accuracy and graphs generated during evaluation.

📈 Model Evaluation Metrics:
        Mean Absolute Error (MAE)
        Root Mean Square Error (RMSE)
        R² Score
        Visual plots: Liquidity trend lines, prediction vs actual graphs