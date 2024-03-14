# Flight-Price-Prediction
# Flight Price Prediction README

## Overview:
This repository contains code for predicting flight prices based on various features such as airline, departure time, duration, class, and days left until departure. The prediction is performed using machine learning algorithms.

## Contents:
1. **flight_price_prediction.ipynb**: Jupyter notebook containing the code for data preprocessing, exploratory data analysis (EDA), model building, and evaluation.
2. **Clean_Dataset.csv**: Cleaned dataset containing flight data.
3. **business.csv**: Dataset containing business class flight details.
4. **economy.csv**: Dataset containing economy class flight details.
5. **README.md**: This file providing an overview of the repository.

## Requirements:
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

## Instructions:
1. **Clone Repository**: Clone this repository to your local machine using the following command:
   ```
   git clone https://github.com/your_username/flight_price_prediction.git
   ```
2. **Install Dependencies**: Install the required libraries using pip:
   ```
   pip install -r requirements.txt
   ```
3. **Run Jupyter Notebook**: Open the `flight_price_prediction.ipynb` notebook in Jupyter Notebook and execute each cell sequentially.

## Dataset:
- The dataset contains flight details such as airline, flight number, source city, departure time, stops, arrival time, destination city, class, duration, days left until departure, and price.

## Exploratory Data Analysis (EDA):
- Exploratory data analysis is performed to understand the distribution of flights across different airlines, classes, departure times, and durations.
- Visualizations are created to analyze the relationship between flight features and prices.

## Model Building:
- Various machine learning algorithms such as Linear Regression, Random Forest Regressor, XGBoost, and Support Vector Regressor are trained on the dataset to predict flight prices.
- The dataset is split into training and testing sets, and features are scaled using MinMaxScaler.
- Models are evaluated using metrics such as Mean Squared Error (MSE), R-squared, and Mean Absolute Error (MAE).

## Future Improvements:
- Experiment with feature engineering techniques to improve model performance.
- Fine-tune hyperparameters of machine learning algorithms for better prediction accuracy.
- Explore ensemble methods to further enhance model performance.

## Contributors:
- Surabhi Sunil
Feel free to contribute to this project by opening issues, suggesting improvements, or submitting pull requests.
