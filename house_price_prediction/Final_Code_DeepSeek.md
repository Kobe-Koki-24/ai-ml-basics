# ===============================
# 📦 Import Required Libraries
# ===============================

# Data handling libraries
import pandas as pd             # Used for data manipulation and analysis (DataFrames, CSV, etc.)
import numpy as np              # Provides mathematical functions, arrays, and numerical operations

# Visualization libraries
import matplotlib.pyplot as plt # For creating plots and charts
import seaborn as sns           # High-level visualization library (built on top of matplotlib)

# Machine learning utilities from scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# train_test_split → splits data into training and validation sets
# cross_val_score → performs cross-validation scoring
# GridSearchCV → hyperparameter tuning

from sklearn.metrics import mean_squared_error, r2_score
# mean_squared_error → evaluates prediction error
# r2_score → evaluates prediction accuracy (R² metric)

from sklearn.pipeline import Pipeline
# Pipeline → chains together preprocessing + model steps in one workflow

from sklearn.compose import ColumnTransformer
# ColumnTransformer → applies different transformations to different columns

from sklearn.impute import SimpleImputer
# SimpleImputer → fills in missing values (e.g., with median or constant)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
# OneHotEncoder → converts categorical values to numeric dummy variables
# StandardScaler → normalizes numeric data (mean=0, std=1)

from sklearn.ensemble import RandomForestRegressor
# RandomForestRegressor → machine learning algorithm based on ensemble decision trees

from sklearn.linear_model import LinearRegression
# LinearRegression → classical regression algorithm (not used here, but imported for testing)

import joblib
# joblib → saves and loads trained models (serialization)


# ===============================
# 📂 Load and Explore Dataset
# ===============================

df = pd.read_csv("train.csv")  # Loads the dataset from 'train.csv' into a pandas DataFrame
print("Shape:", df.shape)      # Prints number of rows and columns in dataset
df.head()                      # Displays the first 5 rows of the dataset

# Show summary info: column types, non-null counts, and memory usage
df.info()

# Check missing values per column, sorted by count (descending order)
missing = df.isnull().sum().sort_values(ascending=False)
# Show top 20 columns with most missing values
missing[missing > 0].head(20)

# ===============================
# 📊 Data Visualization
# ===============================

# Plot distribution of SalePrice column (target variable)
sns.histplot(df['SalePrice'], kde=True)
plt.title("SalePrice distribution")
plt.show()

# Correlation of numeric features with SalePrice (sorted highest → lowest)
num_corr = df.select_dtypes(include=['number']).corr()['SalePrice'].sort_values(ascending=False)
num_corr.head(10)


# ===============================
# 🛠️ Preprocessing
# ===============================

# Drop the 'Id' column (not useful for modeling)
df = df.drop(columns=['Id'])

# Define target variable (y) and features (X)
y = df['SalePrice']
X = df.drop(columns=['SalePrice'])

# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Transformer for numeric columns:
# - Fill missing values with median
# - Standardize values (mean=0, std=1)
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Transformer for categorical columns:
# - Fill missing values with "Missing"
# - Convert categories into one-hot encoded dummy variables
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine numeric and categorical transformers into one preprocessor
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])


# ===============================
# 🤖 Build ML Pipeline
# ===============================

# Full pipeline = preprocessing + model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
# RandomForestRegressor:
# - n_estimators=100 → number of trees
# - random_state=42 → ensures reproducibility
# - n_jobs=-1 → use all CPU cores for faster training


# ===============================
# 🔀 Train-Test Split
# ===============================

# Split into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_val.shape)

# Train pipeline on training data
pipeline.fit(X_train, y_train)

# ===============================
# 📈 Model Evaluation
# ===============================

# Make predictions on validation set
preds = pipeline.predict(X_val)

# Calculate RMSE (root mean squared error)
mse = mean_squared_error(y_val, preds)
rmse = np.sqrt(mse)

# Calculate R² score
r2 = r2_score(y_val, preds)
print(f"Validation RMSE: {rmse:.2f}, R2: {r2:.3f}")

# Scatter plot: Actual vs Predicted SalePrice
plt.scatter(y_val, preds, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--') # reference line (perfect predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()


# ===============================
# 🌟 Feature Importance
# ===============================

# Get names of one-hot encoded categorical features
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_ohe_names = list(ohe.get_feature_names_out(cat_cols))

# Combine numeric + categorical feature names
feature_names = num_cols + cat_ohe_names

# Extract feature importances from RandomForest
importances = pipeline.named_steps['model'].feature_importances_

# Create series of feature importance scores
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.head(20)

# Plot top 15 most important features
feat_imp.head(15).plot(kind='barh', figsize=(8,6))
plt.gca().invert_yaxis()  # highest importance at top
plt.title("Top Feature Importances")
plt.show()


# ===============================
# 💾 Save Trained Model
# ===============================

# Save the trained pipeline (preprocessor + model) to a file
joblib.dump(pipeline, "house_price_pipeline.joblib")
