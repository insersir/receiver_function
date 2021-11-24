# Modification of RF Plot
# Created by Indra Rivaldi Siregar, Pertamina University

# Import package
import matplotlib.pyplot as plt
from rf import read_rf
import numpy as np
from func import norm

# 1. Read Data
data = read_rf(r"E:\Data_Gempa_Baru\Data-Gempa-REAL-MMRI\Good_RF_iterdecon_MMRI.h5")
# data = read_rf(r"E:\Data_Gempa_Baru\Data-Gempa-REAL-SOEI\Good_RF_iterdecon_SOEI.h5")

# Don't modify the lines below ###################################
data = data.sort(['distance'])
distance = []
for i in range(len(data)):
    dist = data[i].stats.distance
    distance.append(dist)

stack = []
for i in range(len(data)):
    stack.append(data[i].data)
stack = np.array(stack).sum(axis=0)
stack = norm(stack)

time_bef_onset = np.round(data[0].stats.onset - data[0].stats.starttime)*-1
time_aft_onset = np.round(data[0].stats.endtime - data[0].stats.onset)
time_axis = np.linspace(time_bef_onset, time_aft_onset, data[0].stats.npts)

zeroline = data[0].stats.npts * [0]
station = data[0].stats.station

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [1, 4]})
tittle = 'Stacking RF with ' + '%.0f' % len(data) + ' traces'

for label in (ax1.get_xticklabels()):
    label.set_fontsize(13)
ax1.set_title('Station ' + station, fontstyle='normal', fontweight='bold', fontsize=16)
ax1.set_title(tittle, fontstyle='italic', fontsize=11, loc='right')
ax1.plot(time_axis, stack, '-k')
ax1.fill_between(time_axis, stack, where=(stack > zeroline), color='red')
ax1.fill_between(time_axis, stack, where=(stack < zeroline), color='grey')
ax1.set_xlim(time_bef_onset, time_aft_onset)
# ax1.axes.yaxis.set_ticks([])
ax1.set_xlabel('Time (s)', fontsize=13)
ax1.set_ylabel('Normalized', fontsize=12)

for RF, Distance in zip(data, distance):
    amp = RF.data
    x = Distance + amp * 1.5      # 1.5 to make amplitude higher

    x_zero = np.linspace(min(distance)-5, max(distance)+5, data[0].stats.npts)

    ax2.plot(x_zero, zeroline, '--', color='black', linewidth=0.75)
    ax2.plot(x, time_axis, '-', color='black')
    ax2.fill_betweenx(time_axis, Distance, x, where=(x > Distance), color='red')
    ax2.fill_betweenx(time_axis, Distance, x, where=(x < Distance), color='grey')

ax2.set_ylim(time_bef_onset, time_aft_onset)
ax2.set_xlim(min(distance)-5, max(distance)+5)       # same with x_zero
ax2.invert_yaxis()
ax2.set_xlabel('Distance (°)', fontsize=14)
ax2.set_ylabel('Time (s)', fontsize=14)
plt.xticks(fontsize=13, color='black')
plt.yticks(fontsize=13, color='black')
plt.show()
