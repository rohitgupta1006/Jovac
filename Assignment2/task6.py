import numpy as np

linspace_arr = np.linspace(0, 1, 10)
print("Equally spaced (0 to 1):", linspace_arr)

identity = np.eye(4)
print("Identity Matrix (4x4):\n", identity)

rand_arr = np.random.randint(1, 101, 20)
sorted_arr = np.sort(rand_arr)
print("Sorted Array:", sorted_arr)
print("Top 5 elements:", sorted_arr[-5:])
