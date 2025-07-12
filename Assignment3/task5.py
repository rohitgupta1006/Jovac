import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
    'Age': [20, 22, 19, 21, 20],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Marks': [85, 78, 92, 74, 88]
})
grouped = df.groupby('Gender')

print("Mean age and marks:\n", grouped[['Age', 'Marks']].mean())

print("Count per gender:\n", grouped.size())
