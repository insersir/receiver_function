# Save waveform to .mat
# Created by Indra Rivaldi Siregar, Pertamina University

from rf import read_rf
from scipy.io import savemat
import numpy as np
import os

station = 'JAGI'                         # station name as folder name                       # change this
path = r"D:\training" + "/" + station                                                  # change this

stream_file = path + "/" + "stream_" + station + ".h5"
stream = read_rf(stream_file)
stream = stream.sort(keys=['onset'])
stream = stream[0:12]
# print(stream)
# print('=' * 20)

data, onset, rayp = [], [], []
for i in range(len(stream)):
    dt = stream[i].data
    rp = stream[i].stats.slowness
    ons = stream[i].stats.onset
    ons1 = np.str(ons.date) + "T" + np.str(ons.hour) + "." + np.str(ons.minute) + "." + np.str(ons.second)

    data.append(dt)
    rayp.append(rp)
    onset.append(ons1)

# We split data according to each event
total = np.int(len(stream) / 3)                 # total events
data = np.array_split(data, total)
onset = np.array_split(onset, total)
rayp = np.array_split(rayp, total)

new_folder = path + "/" + 'Events (.mat)'
if not os.path.exists(new_folder):
    os.mkdir(new_folder)

for j in range(len(data)):
    comp_e = data[j][0]
    comp_n = data[j][1]
    comp_z = data[j][2]

    mdis = {"E": comp_e, "N": comp_n, "Z": comp_z, "rayp": rayp[j][0]}
    filename = 'Signal_' + np.str(stream[0].stats.station) + "_" + onset[j][0] + ".mat"

    savemat(new_folder + "/" + filename, mdis)
    # print("Success " + filename)
