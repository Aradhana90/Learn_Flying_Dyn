import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
good_init = np.array([1, 0, 1]).astype(bool)
print(a[~good_init, :])
