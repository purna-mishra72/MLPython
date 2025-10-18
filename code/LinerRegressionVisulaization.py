# below code will demonstrate how simple linier regression work using python's sklearn library and visualize the results using matplotlib.
#please provide a code which cretae dummy data with some random noise and plot the regression line along with the data points.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Create dummy data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random points as feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with some noise
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Plot the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Linear Regression Visualization')
plt.legend()
plt.show()