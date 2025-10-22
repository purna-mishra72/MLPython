# Linear Regression Visualization
# This script demonstrates a simple linear regression using scikit-learn
# and visualizes the data points and fitted regression line using matplotlib.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def make_and_plot(seed=0):
    # Create dummy data
    np.random.seed(seed)
    X = 2 * np.random.rand(100, 1)  # 100 random points as feature
    y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with some noise

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions for the test set
    y_pred = model.predict(X_test)

    # Prepare a smooth line for the regression (for plotting)
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)

    # Plot the results
    plt.scatter(X, y, color='blue', label='Data points', alpha=0.6)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression line')
    plt.xlabel('Feature (X)')
    plt.ylabel('Target (y)')
    plt.title('Linear Regression Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print a small summary
    coef = model.coef_[0][0] if model.coef_.ndim > 1 else model.coef_[0]
    intercept = model.intercept_[0] if getattr(model.intercept_, 'shape', ()) else model.intercept_
    print(f'Model intercept: {intercept:.3f}, coef: {coef:.3f}')


if __name__ == '__main__':
    make_and_plot()

