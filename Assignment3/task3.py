import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
    'Age': [20, 22, 19, 21, 20],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Marks': [85, 78, 92, 74, 88]
})
df['Passed'] = df['Marks'] >= 80
print("With Passed Column:\n", df)

print("Name and Marks:\n", df[['Name', 'Marks']])

print("Students with Marks > 80:\n", df[df['Marks'] > 80])

topper = df[df['Marks'] == df['Marks'].max()]
print("Topper:\n", topper)
