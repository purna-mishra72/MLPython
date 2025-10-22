"""
property_price_prediction.py

Train and persist a polynomial regression model to predict house prices.

This module expects a CSV file at `data/sample_properties.csv` (relative to this file)
containing at least the following columns:
- area: numeric square footage or area
- bedrooms: integer number of bedrooms
- age: numeric or integer age of the property (years)
- distance_km: numeric distance from city center in kilometers
- price: numeric target variable (house price)

Primary functions:
- load_data: load and validate the CSV file into a pandas DataFrame
- build_and_train: build polynomial features and train a LinearRegression model
- save_model: persist the fitted preprocessor and model using pickle
- main: high-level orchestration (load, split, train, evaluate, save)

Notes:
- Uses scikit-learn for preprocessing and modeling.
- Uses pickle for model persistence (per user request).
"""

import os
import pickle
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Paths: data CSV and where to store trained model
DATA_CSV = os.path.join(os.path.dirname(__file__), 'data', 'sample_properties.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'property_price_model.pkl')


def load_data(path: str = DATA_CSV) -> pd.DataFrame:
    """
    Load property data from a CSV file and perform basic validation.

    Parameters
    - path: path to the CSV file (defaults to DATA_CSV)

    Returns
    - pandas.DataFrame containing the loaded dataset

    Raises
    - FileNotFoundError: when the CSV file does not exist
    - ValueError: when required columns are missing from the file
    """
    # Ensure the file exists before attempting to read it
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}. Run generate_data.py first.")

    # Read CSV into a DataFrame
    df = pd.read_csv(path)

    # Validate that required columns are present
    required_cols = {"area", "bedrooms", "age", "distance_km", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    # Return the raw DataFrame (caller may perform further cleaning)
    return df


def build_and_train(df: pd.DataFrame) -> Tuple[PolynomialFeatures, LinearRegression]:
    """
    Build polynomial features from numeric inputs and train a linear regression model.

    Parameters
    - df: DataFrame containing the dataset with required columns

    Returns
    - tuple (fitted PolynomialFeatures transformer, trained LinearRegression model)

    Behavior / logical steps (documented here and commented in-line):
    1. Select predictor columns and the target column.
    2. Create degree-2 polynomial features (no bias term).
    3. Fit the transformer and transform the predictors.
    4. Train a LinearRegression model on the transformed features.
    """
    # Select input features (predictors) and the target column
    x = df[["area", "bedrooms", "age", "distance_km"]]
    y = df["price"]

    # Create polynomial feature transformer (degree 2) and fit it on X
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(x)  # fit + transform

    # Initialize and fit the linear regression model on polynomial features
    model = LinearRegression()
    model.fit(x_poly, y)

    # Return both the fitted transformer and the trained model
    return poly, model


def visualize_correlations(df: pd.DataFrame, out_dir: str = MODEL_DIR) -> None:
    """
    Compute and print the correlation matrix for numeric features and save
    scatter plots of each feature against price.

    The function saves a PNG (`feature_price_correlations.png`) into out_dir.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Select numeric columns used by the model
    cols = ["area", "bedrooms", "age", "distance_km", "price"]
    numeric = df[cols].copy()

    # Compute and print correlation matrix to stdout
    corr = numeric.corr()
    print("\nCorrelation matrix (features vs price):")
    print(corr.to_string())

    # Create scatter subplots for each feature vs price
    features = ["area", "bedrooms", "age", "distance_km"]
    n = len(features)
    cols_plot = 2
    rows_plot = (n + 1) // cols_plot
    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(10, 4 * rows_plot))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.scatter(numeric[feat], numeric["price"], alpha=0.7)
        ax.set_xlabel(feat)
        ax.set_ylabel("price")
        ax.set_title(f"{feat} vs price")

    # Hide any unused subplot axes
    for j in range(i + 1, len(axes)):
        try:
            axes[j].set_visible(False)
        except Exception:
            pass

    fig.tight_layout()
    out_path = os.path.join(out_dir, "feature_price_correlations.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved feature-vs-price plots to {out_path}")


def visualize_predicted_vs_actual(y_true, y_pred, out_dir: str = MODEL_DIR) -> None:
    """
    Create a scatter plot of predicted vs actual prices and save it.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7)
    # 45-degree reference line (perfect prediction)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    ax.plot([mn, mx], [mn, mx], color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Predicted vs Actual Prices")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "predicted_vs_actual.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved predicted vs actual plot to {out_path}")


def save_model(poly: PolynomialFeatures, model: LinearRegression, path: str = MODEL_PATH) -> None:
    """
    Persist the preprocessing transformer and the trained model to disk using pickle.

    Parameters
    - poly: fitted PolynomialFeatures transformer
    - model: trained LinearRegression model
    - path: destination file path for the pickle file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write a single pickle containing both the preprocessor and the model
    with open(path, 'wb') as f:
        pickle.dump({'preprocessor': poly, 'model': model}, f)

    # Informational print for the user
    print(f"Saved model to {path}")


def main() -> None:
    """
    High-level training flow.

    Steps performed:
    1. Load dataset from CSV
    2. Split into training and test sets
    3. Train polynomial regression on training data
    4. Evaluate on test set (RMSE and R^2)
    5. Save the trained pipeline (preprocessor + model)
    """
    # Load the dataset (may raise FileNotFoundError / ValueError)
    df = load_data()

    # Split the data into training and test sets for evaluation
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Train the pipeline on the training split
    poly, model = build_and_train(train)

    # Prepare test inputs and target for evaluation
    x_test = test[["area", "bedrooms", "age", "distance_km"]]
    y_test = test["price"]

    # Transform the test inputs using the fitted polynomial transformer
    x_test_poly = poly.transform(x_test)

    # Predict on the transformed test features
    preds = model.predict(x_test_poly)

    # Compute evaluation metrics: RMSE and R^2
    # rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    # Print evaluation results for quick feedback
    # print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R^2: {r2:.2f}")

    # Visualization: correlations and predicted vs actual
    try:
        visualize_correlations(df)
    except Exception as e:
        print(f"Warning: failed to create correlation plots: {e}")

    try:
        visualize_predicted_vs_actual(y_test.values, preds)
    except Exception as e:
        print(f"Warning: failed to create predicted-vs-actual plot: {e}")

    # Persist the trained pipeline to disk
    save_model(poly, model)


if __name__ == '__main__':
    # Entry point when run as a script
    main()
