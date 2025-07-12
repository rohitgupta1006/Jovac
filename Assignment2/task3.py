import numpy as np
A = np.random.randint(1, 21, 5)
B = np.random.randint(1, 21, 5)

print("Array A:", A)
print("Array B:", B)

print("Addition:", A + B)
print("Subtraction:", A - B)
print("Multiplication:", A * B)
print("Division:", A / B)

print("Dot Product:", np.dot(A, B))

print("Mean of A:", np.mean(A))
print("Median of A:", np.median(A))
print("Std Dev of A:", np.std(A))
print("Variance of A:", np.var(A))

print("Max in B:", np.max(B), "at index", np.argmax(B))
print("Min in B:", np.min(B), "at index", np.argmin(B))
