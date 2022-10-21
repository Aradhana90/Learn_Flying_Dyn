import numpy as np
from metrics import eul_norm

eul1 = np.array([[-100, 0],
                 [50, 150],
                 [179, 1]])

eul2 = np.array([[-179, 180],
                 [-170, -150],
                 [-160, -10]])

d1 = eul_norm(eul1[:, 0], eul2[:, 1])
d2 = eul_norm(eul1[:, 1], eul2[:, 1])

bla = np.array([4, 6, 3, 8])
print(np.arange(1, 15))
