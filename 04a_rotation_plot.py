# Rotation Plot
# Created by Indra Rivaldi Siregar, Pertamina University

from rf import read_rf
import matplotlib.pyplot as plt
import numpy as np

# 1. Read data
station = 'UGM'        # station name=folder name            # change this

path = r"D:\training" + "/" + station
stream_file = path + "/" + "stream_sorted_" + station + ".h5"

stream = read_rf(stream_file)
stream = stream.sort(keys=['onset'])
# stream = stream[0:9]
# print(stream)

# 2. Rotation
stream_rotate = stream.copy().rotate(method='ZNE->LQT')

# time axis
time_bef_onset = np.round(stream[0].stats.onset - stream[0].stats.starttime)*-1
time_aft_onset = np.round(stream[0].stats.endtime - stream[0].stats.onset)
time_axis = np.linspace(time_bef_onset, time_aft_onset, stream[0].stats.npts)
# print(time_bef_onset, time_aft_onset)

# channel name
chan = [stream[0].stats.channel, stream[1].stats.channel, stream[2].stats.channel]
chan_rot = [stream_rotate[0].stats.channel, stream_rotate[1].stats.channel,
            stream_rotate[2].stats.channel]
channel = chan + chan_rot                       # chan: E, N , Z & chan_rot:T, Q, L
# print(channel)

# 3. Merge stream and stream rotate
st_list = stream + stream_rotate

E = st_list.select(channel=channel[0])
N = st_list.select(channel=channel[1])
Z = st_list.select(channel=channel[2])

T = st_list.select(channel=channel[3])
Q = st_list.select(channel=channel[4])
L = st_list.select(channel=channel[5])

st_merge = [E, N, Z, T, Q, L]
data_length = st_merge[0]                   # length data for each component

colors = ['r'] * 3 + ['k'] * 3

for j in range(len(data_length)):
    fig, ax = plt.subplots(nrows=len(channel), ncols=1, figsize=(16, 8), facecolor="#F5F5F5")
    for i in range(len(st_merge)):
        st = st_merge[i][j]
        # print(st)
        ax[0].set_title('Before vs After 3D Rotation\n' + station + '.onset time: ' +
                        np.str(T[j].stats.onset), fontweight='bold', fontsize=14, color='black')
        ax[i].plot(time_axis, st.data, '-', color=colors[i], label=np.str(st.stats.channel))
        ax[i].legend(loc='upper left')
        ax[i].set_xlim(time_bef_onset, time_aft_onset)
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    plt.subplots_adjust(hspace=0.3)
    plt.show()
