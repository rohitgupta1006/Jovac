import time
import numpy as np
A = np.random.rand(100, 100)
B = np.random.rand(100, 100)

start_time = time.time()

product = np.dot(A, B)
try:
    det = np.linalg.det(product)
    inv = np.linalg.inv(product)
    print("Determinant:", det)
except np.linalg.LinAlgError:
    print("Matrix not invertible")

end_time = time.time()

print("Time taken:", end_time - start_time, "seconds")
