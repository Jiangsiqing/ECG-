import numpy as np
import skimage.measure as measure
import skimage

x = np.zeros((100,))
x[10:40] = 1
x[60:80] = 3
# print(np.where(x == 1))

y = measure.label(x, 4)
max_label = np.max(y)
print(y)
for label in range(1, max_label + 1):
    bool_array = np.where(y == label, 1, 0)
    print(bool_array)
    start, end = np.where(bool_array)[0][[0, -1]]
    print(start, end)
