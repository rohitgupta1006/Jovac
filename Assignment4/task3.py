from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
X_multi = housing.data  
y_multi = housing.target

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)

y_pred_scaled = scaled_model.predict(X_test_scaled)
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
y_pred_multi = multi_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_multi)
rmse = np.sqrt(mse)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
rmse_scaled = np.sqrt(mse_scaled)

r2 = r2_score(y_test, y_pred_multi)
r2_scaled = r2_score(y_test, y_pred_scaled)

print("\nBefore Scaling:")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

print("\nAfter Scaling:")
print(f"MSE: {mse_scaled:.4f}, RMSE: {rmse_scaled:.4f}, R2: {r2_scaled:.4f}")
