import numpy as np
import matplotlib.pyplot as plt


arr1 = np.array([3, 5, 1, 2, 7, 9])
arr2 = np.array([2, 4, 7, 6, 8, 1])
# arr2 = arr1 * 10
# sorted_idxs = np.argsort(arr1)
# print(sorted_idxs)

# print(arr2[sorted_idx])
# # >>> array([2, 1, 0, 4, 3])

# print(arr1[sorted_idxs])
# # >>> array([ 1,  3,  4, 21, 32])

# print(arr2[sorted_idxs])
# # >>> array([ 10,  30,  40, 210, 320])


iters = range(6)

plt.plot(arr1,iters)
plt.show()

