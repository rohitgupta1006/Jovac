import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
    'Age': [20, 22, 19, 21, 20],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Marks': [85, 78, 92, 74, 88]
})
df.to_csv("students_data.csv", index=False)

new_df = pd.read_csv("students_data.csv")

print("Loaded DataFrame:\n", new_df.head())
