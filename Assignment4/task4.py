from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Target'] = housing.target

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

correlation_with_target = df.corr()['Target'].drop('Target')
strongest_features = correlation_with_target.abs().sort_values(ascending=False)
print("\nStrongest Relationships with Target:\n", strongest_features)

print("\nNote: Features with high correlation with each other (off-diagonal values close to 1 or -1) indicate multicollinearity,")
print("which can reduce model interpretability and inflate coefficient estimates.")
