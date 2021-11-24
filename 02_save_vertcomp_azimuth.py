# Save all vertical data by azimuth
# Created by Indra Rivaldi Siregar, Pertamina University
# 1. Import package
import numpy as np
import matplotlib.pyplot as plt
from rf import read_rf, RFStream
from matplotlib.widgets import RadioButtons
from scipy.fftpack import fft
from func import norm

# 2. Read data
station = 'JAGI'        # station name=folder name            # change this

path = r"D:\training" + "/" + station
stream_file = path + "/" + "stream_sorted_" + station + ".h5"

stream = read_rf(stream_file)
stream = stream.sort(keys=['onset'])
# stream = stream[0:9]

# Don't modify the lines below ###################################
data = stream.select(component='Z')
data.sort(keys=['distance'])
print(data.__str__(extended=True))


distance = []
for i in range(len(data)):
    dist = data[i].stats.distance
    distance.append(dist)


time_bef_onset = np.round(data[0].stats.onset - data[0].stats.starttime)*-1
time_aft_onset = np.round(data[0].stats.endtime - data[0].stats.onset)
time_axis = np.linspace(time_bef_onset, time_aft_onset, data[0].stats.npts)

zeroline = data[0].stats.npts * [0]
station = data[0].stats.station

fig, ax1 = plt.subplots(figsize=(14, 7))

for label in (ax1.get_xticklabels()):
    label.set_fontsize(13)
ax1.set_title('Station ' + station, fontstyle='normal', fontweight='bold', fontsize=16)

for i, Distance in zip(data, distance):
    amp = norm(i.data)
    x = Distance + amp * 1      # 1.5 to make amplitude higher

    x_zero = np.linspace(min(distance)-5, max(distance)+5, data[0].stats.npts)

    ax1.plot(x_zero, zeroline, '--', color='black', linewidth=0.75)
    ax1.plot(x, time_axis, '-', color='black')
    # ax1.fill_betweenx(time_axis, Distance, x, where=(x > Distance), color='red')
    # ax1.fill_betweenx(time_axis, Distance, x, where=(x < Distance), color='grey')

ax1.set_ylim(time_bef_onset, time_aft_onset)
ax1.set_xlim(min(distance)-5, max(distance)+5)       # same with x_zero
ax1.invert_yaxis()
ax1.set_xlabel('Distance (°)', fontsize=14)
ax1.set_ylabel('Time (s)', fontsize=14)
plt.xticks(fontsize=13, color='black')
plt.yticks(fontsize=13, color='black')
plt.show()
