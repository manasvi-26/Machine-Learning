import numpy as np

arr1 = np.array([4,3,1,32,21])
arr2 = arr1 * 10
sorted_idxs = np.argsort(-arr1)

print(sorted_idxs)
# >>> array([2, 1, 0, 4, 3])

print(arr1[sorted_idxs])
# >>> array([ 1,  3,  4, 21, 32])

print(arr2[sorted_idxs])
# >>> array([ 10,  30,  40, 210, 320])