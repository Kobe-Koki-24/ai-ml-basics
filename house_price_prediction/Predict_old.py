import joblib
import pandas as pd
import json
import sys

model = joblib.load("house_price_pipeline.joblib")

with open(sys.argv[1]) as f:
    data = json.load(f)

df = pd.DataFrame([data])
pred = model.predict(df)[0]
print(f"Predicted SalePrice: {pred:.2f}")
