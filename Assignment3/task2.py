import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
    'Age': [20, 22, 19, 21, 20],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Marks': [85, 78, 92, 74, 88]
})

print("First two rows:\n", df.head(2))

print("Column Names:", df.columns.tolist())
print("Data Types:\n", df.dtypes)
print("Summary Stats:\n", df.describe())

df['Passed'] = df['Marks'] >= 80
print("With Passed Column:\n", df)
