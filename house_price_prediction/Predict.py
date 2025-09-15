# ===============================
# 📦 Import Required Libraries
# ===============================

import joblib      # For loading the previously saved ML model (house_price_pipeline.joblib)
import pandas as pd # For handling tabular data (creating a DataFrame for input data)
import json        # For reading input data in JSON format
import sys         # For accessing command-line arguments (input file path)


# ===============================
# 📂 Load Trained Model
# ===============================

# Load the trained pipeline model that was saved earlier
# This pipeline includes preprocessing steps + RandomForestRegressor
model = joblib.load("house_price_pipeline.joblib")


# ===============================
# 📥 Load Input Data
# ===============================

# Open the JSON file provided as the first command-line argument (sys.argv[1])
# Example: python predict.py input.json
with open(sys.argv[1]) as f:
    data = json.load(f)  # Parse JSON file into a Python dictionary


# ===============================
# 📝 Prepare Data for Prediction
# ===============================

# Convert dictionary into a DataFrame (single row)
# The model expects input in tabular format, same structure as training data
df = pd.DataFrame([data])


# ===============================
# 🤖 Make Prediction
# ===============================

# Predict SalePrice using the loaded model pipeline
# model.predict() returns an array, so take the first element [0]
pred = model.predict(df)[0]


# ===============================
# 📊 Display Prediction
# ===============================

# Print the predicted SalePrice with 2 decimal places
print(f"Predicted SalePrice: {pred:.2f}")



# Create a sample input.json file with the following content:
# Save this file as Predict.py and run the following command in terminal:
# python predict.py input.json
# python predict.py sample_input.json

