import numpy as np
arr = np.random.randint(10, 51, 15)
print("Original Array:", arr)

print("Elements > 25:", arr[arr > 25])

arr[arr < 30] = 0
print("Replaced <30 with 0:", arr)

count_div5 = np.sum(arr % 5 == 0)
print("Count divisible by 5:", count_div5)
