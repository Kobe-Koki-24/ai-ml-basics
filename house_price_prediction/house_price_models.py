# ===============================
# 📦 Import Required Libraries
# ===============================

import sys           # To read command-line arguments (e.g., python script.py input.json)
import json          # To read JSON files (input house features)
import pandas as pd  # For data manipulation, loading CSV files into DataFrames
import numpy as np   # For numerical computations, arrays, math functions
import matplotlib.pyplot as plt  # For plotting charts
import seaborn as sns            # For high-level visualization (based on matplotlib)

# Machine learning utilities
from sklearn.model_selection import train_test_split   # To split dataset into training and validation sets
from sklearn.metrics import mean_squared_error, r2_score  # To evaluate regression models (RMSE, R²)

from sklearn.pipeline import Pipeline             # To combine preprocessing + model steps
from sklearn.compose import ColumnTransformer     # To apply different transformations to different columns
from sklearn.impute import SimpleImputer          # To fill missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Encode categorical data, scale numeric data

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import joblib  # To save and load trained models
import os      # To check if files exist

# ===============================
# 🤖 Define Models
# ===============================

# Dictionary of all regression models to train and predict
models = {
    "LinearRegression": LinearRegression(),  # Basic linear regression
    "Ridge": Ridge(alpha=1.0),               # Linear regression with L2 regularization
    "Lasso": Lasso(alpha=0.001, max_iter=10000),  # Linear regression with L1 regularization
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),  # Combination of L1 + L2
    "DecisionTree": DecisionTreeRegressor(random_state=42),  # Single decision tree
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),  # Ensemble of trees
    "GradientBoosting": GradientBoostingRegressor(random_state=42),  # Boosted trees
    "AdaBoost": AdaBoostRegressor(random_state=42),  # Adaptive boosting
    "KNeighbors": KNeighborsRegressor(),           # Predicts based on nearest neighbors
    "SVR": SVR(kernel='rbf'),                      # Support Vector Regression
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),  # Neural network
    "XGBRegressor": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0),  # XGBoost
    "LGBMRegressor": LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1),  # LightGBM
    "CatBoost": CatBoostRegressor(verbose=0, n_estimators=200, learning_rate=0.05, random_state=42)  # CatBoost
}

# ===============================
# 📂 Training Function
# ===============================

def train_and_save_models():
    print("📂 Loading dataset...")
    df = pd.read_csv("train.csv")  # Load training dataset
    df = df.drop(columns=['Id'])   # Drop 'Id' column, not needed for prediction

    y = df['SalePrice']            # Target variable
    X = df.drop(columns=['SalePrice'])  # Features for training

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()  # Numeric columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()           # Categorical columns

    # Pipeline for numeric columns: fill missing values with median, then scale
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing numeric values
        ('scaler', StandardScaler())                    # Scale numeric values
    ])

    # Pipeline for categorical columns: fill missing values with 'Missing', then one-hot encode
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),  # Fill missing categories
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding
    ])

    # Combine numeric and categorical preprocessing
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    # Split dataset into training (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []  # To store RMSE and R² for each model

    # Train and save each model
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),  # Preprocessing step
            ('model', model)                 # Regression model
        ])
        pipeline.fit(X_train, y_train)      # Train on training data

        preds = pipeline.predict(X_val)     # Predict on validation data
        mse = mean_squared_error(y_val, preds)  # Mean squared error
        rmse = np.sqrt(mse)                     # Root mean squared error
        r2 = r2_score(y_val, preds)            # R² score

        results.append((name, rmse, r2))       # Save results

        # Save trained model to file
        joblib.dump(pipeline, f"{name}_house_model.pkl")

    # Convert results to DataFrame for easier comparison
    results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"]).sort_values(by="R2", ascending=False)
    print("\n🏆 Model Comparison:")
    print(results_df)

    # Save the best model name for later predictions
    best_model_name = results_df.iloc[0]["Model"]  # Model with highest R²
    with open("best_model.txt", "w") as f:
        f.write(best_model_name)
    print(f"\n✅ Best model saved: {best_model_name}")

# ===============================
# 🔮 Prediction Function
# ===============================

def predict_from_json(json_file):
    print(f"📥 Loading input data from {json_file}...")
    with open(json_file, "r") as f:
        input_data = json.load(f)  # Load JSON data containing house features

    df_input = pd.DataFrame([input_data])  # Convert JSON into DataFrame for sklearn

    predictions = {}  # Store predictions from all models

    print("\n🔮 Predictions from all models:")
    for name in models.keys():
        model_file = f"{name}_house_model.pkl"  # Corresponding saved model file
        if os.path.exists(model_file):          # Check if model exists
            model = joblib.load(model_file)    # Load model
            prediction = model.predict(df_input)[0]  # Predict price
            predictions[name] = prediction
            print(f"{name:15s} → Predicted Price: ${prediction:,.2f}")  # Print prediction
        else:
            print(f"{name:15s} → ❌ Model not trained yet.")

    # Highlight best model prediction based on training
    if os.path.exists("best_model.txt"):
        with open("best_model.txt", "r") as f:
            best_model_name = f.read().strip()  # Read best model name
        if best_model_name in predictions:
            best_price = predictions[best_model_name]
            print("\n🏆 Best Model Prediction:")
            print(f"{best_model_name:15s} → Predicted Price: ${best_price:,.2f}")

# ===============================
# 🚀 Main Entry Point
# ===============================

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # If a JSON file is provided as argument, run prediction
        predict_from_json(sys.argv[1])
    else:
        # If no argument, train and save all models
        train_and_save_models()
