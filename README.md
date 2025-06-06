# California House Price Predictor

A machine learning web application that predicts median house prices in California using XGBoost and Streamlit.

## 🏠 Project Overview

This project is an interactive web application that predicts median house prices in California based on various housing features. It uses the California Housing dataset from scikit-learn and implements an XGBoost Regressor model for predictions.

## 📁 Project Structure

```
california-house-price-predictor/
├── app.py                           # Streamlit web application
├── california_house_price_prediction.py  # Core prediction model
├── California_House_Price_Prediction.ipynb  # Jupyter notebook for model development
└── requirements.txt                 # Project dependencies
```

## 🛠️ Features

- Interactive web interface built with Streamlit
- Real-time house price predictions
- Input sliders for all housing features
- Model performance metrics display
- Sample data visualization
- Cached model training for better performance

## 📊 Dataset Features

The model uses the following features from the California Housing dataset:

- **MedInc**: Median income in block group (10k USD)
- **HouseAge**: Median house age in block group (years)
- **AveRooms**: Average rooms per household
- **AveBedrms**: Average bedrooms per household
- **Population**: Population of the block group
- **AveOccup**: Average household occupancy
- **Latitude**: Latitude coordinate
- **Longitude**: Longitude coordinate

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/HARIHARANS24/california-house-price-predictor.git
cd california-house-price-predictor
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar sliders to input housing features

4. Click the "Predict House Price" button to get the prediction

## 📦 Dependencies

- streamlit
- numpy
- pandas
- scikit-learn
- xgboost

## 🎯 Model Performance

The model's performance is evaluated using:
- R² Score (coefficient of determination)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Author

- HARIHARANS24

## 🙏 Acknowledgments

- California Housing dataset from scikit-learn
- Streamlit for the web interface
- XGBoost for the machine learning model 
