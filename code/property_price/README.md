Property Price Estimator (toy project)

This small project generates synthetic property data, trains a model to estimate property prices, and provides a CLI to predict prices for new samples.

Files:
- generate_data.py  - generate a synthetic CSV dataset (data/sample_properties.csv)
- train_model.py    - train a model and save it to models/property_price_model.joblib
- predict.py        - CLI to predict price for a single sample
- visualize.py      - basic scatter plot (price vs area colored by locality)
- requirements.txt  - Python dependencies

Quick start (Windows cmd):

1) Create a virtual environment (optional but recommended):
   python -m venv .venv
   .venv\Scripts\activate

2) Install dependencies:
   pip install -r code\property_price\requirements.txt

3) Generate data:
   python code\property_price\generate_data.py

4) Train model:
   python code\property_price\train_model.py

5) Predict (example):
   python code\property_price\predict.py --area 1100 --bedrooms 3 --age 8 --distance_km 5 --locality Downtown

6) Visualize:
   python code\property_price\visualize.py

Notes:
- This is a toy example using synthetic data and is intended for learning/demonstration only.
import numpy as np
import pandas as pd
import os

# Generate synthetic property data with locality categories and noise
def generate_property_data(n=1000, out_csv='data/sample_properties.csv', seed=42):
    np.random.seed(seed)
    localities = ['Downtown', 'Suburb', 'Riverside', 'Hillview']

    rows = []
    for _ in range(n):
        loc = np.random.choice(localities)
        # locality-dependent base price and area distribution
        if loc == 'Downtown':
            area = np.random.normal(900, 200)
            base = 200000
            distance = np.random.normal(2, 1)
        elif loc == 'Suburb':
            area = np.random.normal(1200, 250)
            base = 150000
            distance = np.random.normal(15, 5)
        elif loc == 'Riverside':
            area = np.random.normal(1000, 220)
            base = 170000
            distance = np.random.normal(8, 3)
        else:  # Hillview
            area = np.random.normal(1100, 230)
            base = 160000
            distance = np.random.normal(10, 4)

        area = max(200, area)
        bedrooms = int(np.clip(np.round(area / 400 + np.random.choice([-1, 0, 1])), 1, 6))
        age = int(np.clip(np.random.exponential(15), 0, 80))
        distance = max(0.1, distance)

        # Price is a function of these with some noise
        price = (
            base
            + area * 150
            + bedrooms * 12000
            - age * 800
            - distance * 2000
            + np.random.normal(0, 15000)
        )

        rows.append({
            'area': round(area, 1),
            'bedrooms': bedrooms,
            'age': age,
            'distance_km': round(distance, 2),
            'locality': loc,
            'price': round(max(10000, price), 2),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Generated {len(df)} rows and saved to {out_csv}")
    return df


if __name__ == '__main__':
    generate_property_data(n=1000)

