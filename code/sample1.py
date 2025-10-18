import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'size_sqft': [1000, 1500, 1800, 2400, 3000],
    'bedrooms': [2, 3, 3, 4, 5],
    'age_years': [10, 5, 8, 2, 1],
    'price_lakhs': [50, 70, 80, 120, 150]
}

df = pd.DataFrame(data)
print(df)
X = df[['size_sqft', 'bedrooms', 'age_years']]   # Features
y = df['price_lakhs']                             # Target (label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predicted:", predictions)
print("Actual:", y_test.values)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
new_house = np.array([[2000, 3, 4]])
predicted_price = model.predict(new_house)

print("Predicted price (in lakhs):", predicted_price[0])