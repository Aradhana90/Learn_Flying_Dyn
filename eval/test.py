import numpy as np

from metrics import get_rmse

y = np.array([[1, 3],
              [2, 4]])
y_hat = np.array([[0, 2],
                  [2, 5]])

print(get_rmse(y, y_hat))
