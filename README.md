# california-house-price-predictor


This is an interactive web application built using **Streamlit** that predicts median house prices in California based on various housing features. The underlying model is trained with the California Housing dataset from scikit-learn using **XGBoost Regressor**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [App Interface](#app-interface)
- [Performance Metrics](#performance-metrics)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project aims to provide an easy-to-use web app where users can input housing-related parameters and receive an immediate prediction of the median house price in California. It highlights the combination of machine learning and interactive web apps for practical real-world problems.

---

## Dataset

- **Source:** California Housing dataset from [scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- **Description:** Contains 20,640 samples from the 1990 California census with 8 feature variables and one target (median house value).
- **Features:**

  | Feature    | Description                                |
  |------------|--------------------------------------------|
  | MedInc     | Median income in block group (10k USD)     |
  | HouseAge   | Median house age in block group (years)    |
  | AveRooms   | Average rooms per household                 |
  | AveBedrms  | Average bedrooms per household              |
  | Population | Population of the block group               |
  | AveOccup   | Average household occupancy                  |
  | Latitude   | Latitude coordinate                         |
  | Longitude  | Longitude coordinate                        |

- **Target:**

  | Target | Description                          |
  |--------|------------------------------------|
  | price  | Median house value (in 100,000 USD) |

---

## Model

- **Algorithm:** XGBoost Regressor
- **Why XGBoost?**  
  Known for superior performance and efficiency on tabular data, XGBoost is a robust choice for regression tasks.
- **Training Details:**  
  Dataset split into 80% training and 20% testing with a fixed random seed.
- **Evaluation Metrics:**
  - R² Score (coefficient of determination)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

---

## Features

- User-friendly sliders to input all 8 features.
- Instant predictions of median house prices.
- View a sample of raw dataset records.
- Display model performance metrics on test data.
- Clean UI powered by Streamlit.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/california-house-price-predictor.git
cd california-house-price-predictor

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# 3. Install required Python packages
pip install -r requirements.txt
