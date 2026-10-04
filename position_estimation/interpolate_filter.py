import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter

array_cluster = np.load("temporal_array.npy")
window_size = 15

for i in np.unique(array_cluster[:, -1]):
    substr = array_cluster[array_cluster[:, -1] == i]

    time = substr[:, 0]
    data = substr[:, 1:3]

    f = interp1d(time, data, axis=0, kind="linear")

    new_time = np.arange(time[0], time[-1] + window_size, window_size)

    interp_data = f(new_time)

    interp_x = interp_data[:, 0]
    interp_y = interp_data[:, 1]

    x_med = medfilt(interp_x, kernel_size=5)
    y_med = medfilt(interp_y, kernel_size=5)
    x_filt = savgol_filter(x_med, window_length=7, polyorder=2)
    y_filt = savgol_filter(y_med, window_length=7, polyorder=2)

    plt.scater(x_filt, y_filt)

plt.scatter(0, 0, c="pink")
plt.scatter(10, 0, c="teal")
plt.scatter(0, 20, c="teal")
plt.scatter(10, 20, c="teal")
plt.hlines(0, 0, 10, colors="k")
plt.hlines(20, 0, 10, colors="k")
plt.hlines(10, 0, 10, colors="k")
plt.hlines(3.05, 0, 10, colors="k")
plt.hlines(16.95, 0, 10, colors="k")
plt.vlines(0, 0, 20, colors="k")
plt.vlines(10, 0, 20, colors="k")
plt.vlines(5, 3.05, 16.95, colors="k")
plt.axis("equal")
plt.gca().invert_yaxis()
plt.savefig("graphique.png")