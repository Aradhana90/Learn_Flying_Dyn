import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

a = np.array([[1, 2],
              [0, 2],
              [0, 1],
              [0, 1, 2]], dtype=list)
print(a[3])
