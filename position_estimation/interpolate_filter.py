import numpy as np
from scipy.interpolate import interp1d

array_cluster = np.load("temporal_array.npy")
print(array_cluster[array_cluster[:, -1] == 0])

f_x = interp1d(lst_time, lst_x, kind="linear")
f_y = interp1d(lst_time, lst_y, kind="linear")

new_x = f_x(new_time)
new_y = f_y(new_time)