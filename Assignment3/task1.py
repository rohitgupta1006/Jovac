import pandas as pd

data = [25, 30, 35, 40, 45]
s = pd.Series(data, index=['A', 'B', 'C', 'D', 'E'])

print("First 3 elements:\n", s.head(3))

print("Mean:", s.mean())
print("Median:", s.median())
print("Standard Deviation:", s.std())
