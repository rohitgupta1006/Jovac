import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

print("Titanic Head:\n", titanic.head())
print("Info:\n")
titanic.info()
print("Summary Stats:\n", titanic.describe())

print("Missing values:\n", titanic.isnull().sum())

sns.countplot(x='survived', data=titanic)
plt.title("Survival Count")
plt.show()
