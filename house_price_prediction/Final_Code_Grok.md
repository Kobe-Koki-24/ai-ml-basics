# ===============================
# 📦 Import Required Libraries
# ===============================

# Data handling libraries
import pandas as pd             # For data manipulation and analysis using DataFrames
import numpy as np              # For numerical operations and array handling

# Visualization libraries
import matplotlib.pyplot as plt # For creating static and interactive plots
import seaborn as sns           # For advanced statistical visualizations

# Machine learning utilities from scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# train_test_split: Splits data into training and validation sets
# cross_val_score: Evaluates model performance using cross-validation
# GridSearchCV: Tunes hyperparameters via grid search
from sklearn.metrics import mean_squared_error, r2_score
# mean_squared_error: Measures average squared prediction error
# r2_score: Measures proportion of variance explained by the model
from sklearn.pipeline import Pipeline
# Pipeline: Chains preprocessing and modeling steps into a single workflow
from sklearn.compose import ColumnTransformer
# ColumnTransformer: Applies different transformations to specific columns
from sklearn.impute import SimpleImputer
# SimpleImputer: Fills missing values with specified strategy (e.g., median)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# OneHotEncoder: Converts categorical variables to binary dummy variables
# StandardScaler: Standardizes numerical features (mean=0, std=1)
from sklearn.ensemble import RandomForestRegressor
# RandomForestRegressor: Ensemble model using multiple decision trees for regression
from sklearn.linear_model import LinearRegression
# LinearRegression: Basic linear regression model (not used, but imported)
import joblib
# joblib: Serializes and saves trained models to disk

# ===============================
# 📂 Load and Explore Dataset
# ===============================

df = pd.read_csv("train.csv")  # Load dataset from 'train.csv' into a pandas DataFrame
print("Shape:", df.shape)      # Print the number of rows and columns in the dataset
df.head()                      # Display the first 5 rows to inspect data structure

# Display summary of DataFrame: column types, non-null counts, and memory usage
df.info()

# Calculate missing values per column and sort in descending order
missing = df.isnull().sum().sort_values(ascending=False)
# Display top 20 columns with missing values (if any)
missing[missing > 0].head(20)

# ===============================
# 📊 Data Visualization
# ===============================

# Plot histogram of SalePrice with kernel density estimate (KDE)
sns.histplot(df['SalePrice'], kde=True)
plt.title("SalePrice distribution")  # Set plot title
plt.show()                           # Display the plot

# Compute correlation of numerical columns with SalePrice
num_corr = df.select_dtypes(include=['number']).corr()['SalePrice'].sort_values(ascending=False)
# Display top 10 numerical features most correlated with SalePrice
num_corr.head(10)

# ===============================
# 🛠️ Preprocessing
# ===============================

# Drop 'Id' column as it is not useful for prediction
df = df.drop(columns=['Id'])

# Define target variable (y) as SalePrice
y = df['SalePrice']
# Define features (X) by removing SalePrice from DataFrame
X = df.drop(columns=['SalePrice'])

# Identify numerical columns (int64 and float64 types)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Identify categorical columns (object type)
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create pipeline for numerical columns
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with column median
    ('scaler', StandardScaler())                    # Scale features to mean=0, std=1
])

# Create pipeline for categorical columns
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),  # Fill missing values with 'Missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Convert categories to one-hot encoded columns
])

# Combine numerical and categorical transformers
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),  # Apply numeric_transformer to numerical columns
    ('cat', categorical_transformer, cat_cols)  # Apply categorical_transformer to categorical columns
])

# ===============================
# 🤖 Build ML Pipeline
# ===============================

# Create full pipeline: preprocessing + Random Forest model
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Apply preprocessing steps
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))  # Train Random Forest with 100 trees
])
# RandomForestRegressor parameters:
# - n_estimators=100: Use 100 decision trees
# - random_state=42: Ensure reproducibility
# - n_jobs=-1: Use all available CPU cores for parallel processing

# ===============================
# 🔀 Train-Test Split
# ===============================

# Split data into 80% training and 20% validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_val.shape)  # Print shapes of training and validation feature sets

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# ===============================
# 📈 Model Evaluation
# ===============================

# Generate predictions on the validation set
preds = pipeline.predict(X_val)

# Calculate mean squared error (MSE) between actual and predicted values
mse = mean_squared_error(y_val, preds)
# Calculate root mean squared error (RMSE) as square root of MSE
rmse = np.sqrt(mse)

# Calculate R² score to measure variance explained by the model
r2 = r2_score(y_val, preds)
# Print RMSE (2 decimal places) and R² (3 decimal places)
print(f"Validation RMSE: {rmse:.2f}, R2: {r2:.3f}")

# Create scatter plot of actual vs. predicted SalePrice
plt.scatter(y_val, preds, alpha=0.5)  # Plot points with partial transparency
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')  # Add red dashed line for perfect predictions
plt.xlabel("Actual")                   # Label x-axis
plt.ylabel("Predicted")               # Label y-axis
plt.title("Actual vs Predicted")      # Set plot title
plt.show()                            # Display the plot

# ===============================
# 🌟 Feature Importance
# ===============================

# Access OneHotEncoder from the categorical transformer
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
# Get names of one-hot encoded categorical features
cat_ohe_names = list(ohe.get_feature_names_out(cat_cols))
# Combine numerical and categorical feature names
feature_names = num_cols + cat_ohe_names

# Extract feature importances from the Random Forest model
importances = pipeline.named_steps['model'].feature_importances_
# Create a Series of feature importances, indexed by feature names, sorted descending
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# Display top 20 most important features
feat_imp.head(20)

# Plot top 15 feature importances as a horizontal bar chart
feat_imp.head(15).plot(kind='barh', figsize=(8,6))  # Create bar plot with 8x6 inch size
plt.gca().invert_yaxis()                           # Invert y-axis to show highest importance at top
plt.title("Top Feature Importances")                # Set plot title
plt.show()                                         # Display the plot

# ===============================
# 💾 Save Trained Model
# ===============================

# Save the trained pipeline to a file for future use
joblib.dump(pipeline, "house_price_pipeline.joblib")