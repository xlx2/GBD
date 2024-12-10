import numpy as np

from utils import create_block_diag_matrix

H = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(np.hstack([np.zeros([2, 3]), np.array([[np.sqrt(2)],[1]])]))