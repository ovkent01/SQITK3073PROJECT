# 📈 Malaysia Stock Forecast & Comparison Dashboard

## 📌 Overview

This is a group project for the **SQIT 3073: Business Analytic Programming** course. This Streamlit-powered dashboard serves as a comprehensive tool for short-term investors to analyze Malaysian stock market trends.

The application features two distinct modes:
1.  **Comparison Graph**: Compare the closing price trends of two different stocks side-by-side.
2.  **Stock Forecast**: Predict future stock prices for up to 5 selected stocks using a **Random Forest** machine learning model.

The system retrieves historical data via the **Yahoo Finance API** and allows users to customize the machine learning parameters for dynamic analysis.

---

## 🔍 Key Features

### 1. 📊 Comparison Mode
- **Dual Stock Analysis**: Select and compare the performance of any two stocks from the supported list.
- **Visual Comparison**: Overlays closing prices on a single interactive chart to identify relative performance.
- **Real-time Data**: Displays the latest closing prices for quick reference.

### 2. 📈 Forecast Mode
- **Multi-Stock Forecasting**: Select up to **5 stocks** simultaneously to view individual forecasts.
- **Machine Learning**: Uses `RandomForestRegressor` to predict stock prices for the next **30 days**.
- **Adjustable Parameters**:
    - **Training Range**: Slide to choose between **6 to 24 months** of historical training data.
    - **Feature Engineering**: Toggle between simple features (`Lag1`) or complex features (`MA7 + MA14 + Lag1`) to see how moving averages affect accuracy.
- **Model Evaluation**: Automatically calculates and displays **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error) to assess prediction quality.

### 3. 🎨 User Experience
- **Interactive Sidebar**: Full control over modes, stock selection, and model settings.
- **Dynamic Visuals**: 
    - Displays past **6 months** of historical data.
    - Differentiates historical data (Blue) from predicted data (Green).
- **Wide Layout**: Optimized for desktop viewing with a responsive grid system.

---

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **Data Source**: yfinance (Yahoo Finance API)
- **Machine Learning**: Scikit-Learn (Random Forest Regressor)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/stock-forecast-dashboard.git](https://github.com/your-username/stock-forecast-dashboard.git)
cd stock-forecast-dashboard