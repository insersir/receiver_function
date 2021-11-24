# Plot Stack Align RF
# Created by Indra Rivaldi Siregar, Pertamina University

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from rf import read_rf, RFStream
from func import norm
from obspy.geodetics import degrees2kilometers
from obspy.geodetics import locations2degrees

# location of initial station (initial point) must be correct (left to right side);
# other stations can be put randomly

# 1. Input Station Name
# station = ["CISI", "AE2", "AG3", "AG4", "AI1", "AK2", "AK4"]
# station = ["DK1", "DL1", "NL2", "NL3", "NL5"]
station = ["AE2", "AG3", "AG4", "AI1", "AK2", "AK4"]

new_stream = RFStream()
for sta in station:
    path = r"D:\plot_stack_align" + "/" + sta               # change path
    file = path + "/" + "good_rf_iter_" + sta + ".h5"
    stream = read_rf(file)
    data_stack = stream.stack()
    new_stream.extend(data_stack)
# print(stream.__str__(extended=True))

# Don't modify the lines below ###################################
long, lat, data = [], [], []
for i in range(len(new_stream)):
    long.append(new_stream[i].stats.station_longitude)
    lat.append(new_stream[i].stats.station_latitude)
    data.append(new_stream[i].data)
long = np.array(long)
lat = np.array(lat)

data_norm = []
for j in range(len(data)):
    normalize = norm(data[j])
    data_norm.append(normalize)

# time axis
time_bef_onset = -5
time_aft_onset = 25
time_axis = []
for k in range(len(new_stream)):
    t_axis = np.linspace(time_bef_onset, time_aft_onset, new_stream[k].stats.npts)
    time_axis.append(t_axis)

# 2. Predict the latitude
x = np.array(long).reshape(-1, 1)
y = np.array(lat).reshape(-1, 1)

linear_regressor = LinearRegression()                       # create object for the class
linear_regressor.fit(x, y)                                  # perform linear regression
lat_pred = linear_regressor.predict(x)                      # make predictions
lat_pred = lat_pred.reshape(-1).tolist()

print("Longitude:", long)
print("Latitude:", lat)
print("Latitude from regression:", lat_pred)

# 3. Calculate stations distance
long0, lat0 = long[0], lat_pred[0]      # initial points
dist_station = [0, ]
for k in range(1, len(long)):
    dist = locations2degrees(lat_pred[k], long[k], lat0, long0)
    dist_km = degrees2kilometers(dist)
    dist_station.append(dist_km)
dist_station = np.array(dist_station)
# print(dist_station)

plt.title('Stack Align', fontstyle='normal', fontweight='bold', fontsize=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.plot(long, lat_pred, 'r.', markersize=20, label='prediction location')
plt.plot(long, lat, 'k.', markersize=20, label='original location')
plt.plot(long, lat_pred, 'b')
for w, txt in enumerate(station):
    plt.annotate(txt, (long[w], lat[w]), color='black', fontsize=7, ha='center', fontfamily="sans",
                 fontstyle="normal", fontweight='bold', bbox={"facecolor": "white", "alpha": 0.8})
plt.legend()
plt.show()

fig, ax = plt.subplots()
zeroline = new_stream[0].stats.npts * [0]

yloc_text = dist_station*0 + -4.1      # y-axis for station name

for RF, Distance, time in zip(data_norm, dist_station, time_axis):
    x = Distance + RF * 10       # 10 to make amplitude higher

    x_zero = np.linspace(min(dist_station) - 20, max(dist_station) + 20, new_stream[0].stats.npts)

    ax.plot(x_zero, zeroline, '--', color='grey', linewidth=0.9)
    ax.plot(x, time, '-', color='grey')
    ax.fill_betweenx(time, Distance, x, where=(x > Distance), color='grey')
    ax.fill_betweenx(time, Distance, x, where=(x < Distance), color='white')

for w, txt in enumerate(station):
    plt.annotate(txt, (dist_station[w], yloc_text[w]), color='black', fontsize=12, ha='center', fontfamily="sans",
                 fontstyle="normal", fontweight='bold', bbox={"facecolor": "white", "alpha": 0.8})
ax.set_ylim(time_bef_onset, time_aft_onset)
ax.set_xlim(min(dist_station)-20, max(dist_station)+20)       # same with x_zero
ax.invert_yaxis()
plt.xlabel('Distance (km)', fontsize=13)
plt.ylabel('Time (s)', fontsize=13)
plt.xticks(fontsize=13, color='black')
plt.yticks(fontsize=13, color='black')
plt.show()
