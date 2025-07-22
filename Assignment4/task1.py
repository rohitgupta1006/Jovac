from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


housing = fetch_california_housing()
X = housing.data[:, [3]]  
y = housing.target

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual Price')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Average Number of Rooms')
plt.ylabel('House Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
