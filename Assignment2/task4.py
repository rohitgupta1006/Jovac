import numpy as np
arr = np.arange(1, 13)

reshaped_2d = arr.reshape(4, 3)
print("Reshaped to 2D (4x3):\n", reshaped_2d)

reshaped_3d = arr.reshape(2, 2, 3)
print("\nReshaped to 3D (2x2x3):\n", reshaped_3d)

transposed = reshaped_2d.T
print("\nTransposed 2D array:\n", transposed)
print("Transposed shape:", transposed.shape)
