# Zhu & Kanamori Method
# Created by Indra Rivaldi Siregar, Pertamina University

# 1. Import package
from rf import read_rf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from sklearn import preprocessing
import time
from tqdm import tqdm
import colorama as color
from func import norm
from sklearn.linear_model import LinearRegression
start_time = time.time()

# 2. Read Data
dataRF = read_rf(r"E:\Data_Gempa_Baru\Data-Gempa-REAL-MMRI\Good_RF_iterdecon_MMRI.h5")

# 3. Parameter Input
H1 = 25             # H minimum
H2 = 40             # H maximum
vpvs1 = 1.5         # vpvs minimum
vpvs2 = 2.0         # vpvs maximum
vp = 6.0            # P-wave velocity (km/s)

# Weighting (w1, w2, w3) for Ps-wave, PpPs-wave, PpSs+PsPs-wave
weight1 = 0.7
weight2 = 0.2
weight3 = 0.1

dataRF = dataRF.sort(['distance'])
data = dataRF.copy().trim2(0, 25, 'onset')
fs = data[0].stats.sampling_rate
time_length_RF = (data[0].stats.npts - 1) / fs
# print(data)
# print(dataRF)

# Bandpass Filter 1 for Ps-wave (Hz)
Ps_min = 0.05
Ps_max = 2

# Bandpass Filter 2 for Multiples Wave (Hz)
Multiples_min = 0.025
Multiples_max = 0.5

# Don't modify the lines below ###################################
total_grid = 100
xline = np.linspace(H1, H2, total_grid)
yline = np.linspace(vpvs1, vpvs2, total_grid)

ray_param = []
model = []
list_total_ampw = []
with tqdm(total=len(data), desc='Calculating', disable=False) as pbar:
    for hh in range(len(data)):
        ray_p = data[hh].stats.slowness / 111.19492664455873        # Ray Parameter (s/km)
        ray_param.append(ray_p)
        if Ps_min and Ps_max and Multiples_min and Multiples_max is not None:
            data_high = data[hh].copy().filter("bandpass", freqmin=Ps_min, freqmax=Ps_max, corners=2, zerophase=True)
            data_low = data[hh].copy().filter("bandpass", freqmin=Multiples_min, freqmax=Multiples_max, corners=2,
                                              zerophase=True)
        else:
            data_high = data[hh].copy()
            data_low = data[hh].copy()

        Amp_t1 = []
        Amp_t2 = []
        Amp_t3 = []
        mod = []
        for m in range(len(yline)):
            for k in range(len(xline)):
                M = [xline[k], yline[m]]

                # t1 or Ps-wave arrival time
                t1 = ((1/(vp/M[1])**2 - ray_p**2)**0.5 - (1/vp**2 - ray_p**2)**0.5) * M[0]

                # t2 or PpPs-wave arrival time
                t2 = ((1/(vp/M[1])**2 - ray_p**2)**0.5 + (1/vp**2 - ray_p**2)**0.5) * M[0]

                # t3 or PpSs+PsPs-wave Arrival
                t3 = (2*((1/(vp/M[1])**2 - ray_p**2)**0.5)) * M[0]

                if t3 > time_length_RF:
                    t3 = time_length_RF

                # The RF Values on the waveform are taken based on the estimated values of t1, t2, t2 for all models
                pred_t1 = np.round(t1 * fs)
                pred_t1 = np.int(pred_t1)
                amp_t1 = data_high[pred_t1]
                Amp_t1.append(amp_t1)

                pred_t2 = np.round(t2 * fs)
                pred_t2 = np.int(pred_t2)
                amp_t2 = data_low[pred_t2]
                Amp_t2.append(amp_t2)

                pred_t3 = np.round(t3 * fs)
                pred_t3 = np.int(pred_t3)
                amp_t3 = data_low[pred_t3]
                Amp_t3.append(amp_t3)

                mod.append(M)
        model.append(mod)

        w1 = weight1
        w2 = weight2
        w3 = weight3
        total_ampw = []
        for bb in range(len(Amp_t1)):
            ampw_t1 = Amp_t1[bb] * w1
            ampw_t2 = Amp_t2[bb] * w2
            ampw_t3 = Amp_t3[bb] * w3
            total = ampw_t1 + ampw_t2 - ampw_t3
            total_ampw.append(total)
        list_total_ampw.append(total_ampw)
        pbar.update(1)
pbar.write("Wait a minute....")

# Stacking the Amplitude of All RF & Find The H,k estimation
stack = np.array(list_total_ampw).sum(axis=0)
maxi = np.where(stack == np.max(stack))
index_max = maxi[0][0]

model = model[0]
crustal_est = model[index_max][0]
vpvs_est = model[index_max][1]
print("Crustal Estimation:", crustal_est)
print("Vp/Vs Estimation:", vpvs_est)

# Calculate Error of H & k Estimation (Margin of Error with 95% confidence interval)
H = []
k = []
for i in range(len(list_total_ampw)):
    maxi_er = np.where(list_total_ampw[i] == np.max(list_total_ampw[i]))
    index_maxi_er = maxi_er[0][0]
    H_er = model[index_maxi_er][0]
    k_er = model[index_maxi_er][1]
    H.append(H_er)
    k.append(k_er)
std_H = np.std(H)
std_k = np.std(k)

# # 1.96 is a z-score with 95% confidence interval (normal distribution)
total_data = len(H)     # We can also use k
crustal_err = 1.96 * std_H / total_data ** 0.5
vpvs_err = 1.96 * std_k / total_data ** 0.5
print("Error of Crustal Thickness Estimation:", crustal_err)
print("Error of Vp/Vs Ratio Estimation:", vpvs_err)

info = [crustal_est, crustal_err, vpvs_est, vpvs_err]
# print(info)

# Normalization of the stacking amplitude
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
stack_norm = min_max_scaler.fit_transform(stack.reshape(-1, 1))
stack_norm = stack_norm.reshape(-1)

# Plotting the results of zhu & kanamori method
model_plot = np.array(model).T.tolist()
xaxis = model_plot[0]
yaxis = model_plot[1]

stack_grid = griddata((xaxis, yaxis), stack_norm, (xline[None, :], yline[:, None]))  # griddata must use xaxis and yaxis
plt.contourf(xline, yline, stack_grid, 100, cmap="RdBu_r")  # 100: total of contour lines
plt.colorbar()
# plt.plot(xaxis, yaxis, '.k')

station = data[0].stats.station
Title = "Station " + np.str(station)
plt.title(Title, fontsize=20, fontstyle="normal", fontfamily="sans", fontweight='bold')
plt.plot(crustal_est, vpvs_est, "P", markersize=18, color="yellow")
p1 = 'H (km): ' + np.str(np.round(crustal_est, 2)) + ' ± ' + np.str(np.round(crustal_err, 2))
p2 = 'Vp/Vs : ' + np.str(np.round(vpvs_est, 2)) + ' ± ' + np.str(np.round(vpvs_err, 2))
plt.title(p1 + '\n' + p2, fontstyle='normal', fontfamily='tahoma', fontsize=12, loc='right',
          color='blue', fontweight='bold')
plt.xlabel("Crustal Thickness (H)", fontsize=16)
plt.ylabel("Vp / Vs ratio (κ)", fontsize=16)
plt.xticks(fontsize=16, color='black')
plt.yticks(fontsize=16, color='black')
end_time = time.time()
runningtime = end_time - start_time
print(color.Fore.RESET)
print('Running Time:', np.round(end_time-start_time, 2), "seconds")
plt.show()


# Calculate Ps-wave and multiples-wave arrival time from Zhu Kanamori Method
ray_param = np.array(ray_param)
k = np.array(k)
ps = ((1/(vp/k)**2 - ray_param**2)**0.5 - (1/vp**2 - ray_param**2)**0.5) * H
M1 = ((1/(vp/k)**2 - ray_param**2)**0.5 + (1/vp**2 - ray_param**2)**0.5) * H
M2 = (2*((1/(vp/k)**2 - ray_param**2)**0.5)) * H

# Plot Ps-Wave and Multiples Arrival Time from The Results of Zhu Kanamori Method
distance = []
for i in range(len(dataRF)):
    dist = dataRF[i].stats.distance
    distance.append(dist)

time_bef_onset = np.round(dataRF[0].stats.onset - dataRF[0].stats.starttime)*-1
time_aft_onset = np.round(dataRF[0].stats.endtime - dataRF[0].stats.onset)
time_axis = np.linspace(time_bef_onset, time_aft_onset, dataRF[0].stats.npts)

zeroline = dataRF[0].stats.npts * [0]

stack_plot = []
for j in range(len(dataRF)):
    stack_plot.append(dataRF[j].data)
stack_plot = np.array(stack_plot).sum(axis=0)
stack_plot = norm(stack_plot)

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})

tittle = 'Stacking RF with ' + '%.0f' % len(data) + ' traces'
for label in (ax1.get_xticklabels()):
    label.set_fontsize(16)
ax1.set_title('Station ' + station, fontstyle='normal', fontweight='bold', fontsize=16)
ax1.set_title(tittle, fontstyle='italic', fontsize=13, loc='right')
ax1.plot(time_axis, stack_plot, '-k')
ax1.fill_between(time_axis, stack_plot, where=(stack_plot > zeroline), color='red')
ax1.fill_between(time_axis, stack_plot, where=(stack_plot < zeroline), color='grey')
ax1.set_xlim(time_bef_onset, time_aft_onset)
ax1.set_xlabel('Time (s)', fontsize=13)
ax1.axes.yaxis.set_ticks([])
# ax1.axes.yaxis.set_ticklabels([], ticks=None)
ax1.set_ylabel('Normalized', fontsize=12)

# Regression of waves arrival time (Ps, M1, M2)
x1 = np.array(distance).reshape(-1, 1)
y1 = ps.reshape(-1, 1)
y2 = M1.reshape(-1, 1)
y3 = M2.reshape(-1, 1)
waves_arr = [y1, y2, y3]

waves_arr_reg = []
for i in range(len(waves_arr)):
    regressor = LinearRegression()
    regressor.fit(x1, waves_arr[i])
    reg = regressor.predict(x1)
    waves_arr_reg.append(reg)

Ps_pred = waves_arr_reg[0].reshape(-1)
M1_pred = waves_arr_reg[1].reshape(-1)
M2_pred = waves_arr_reg[2].reshape(-1)

for RF, Distance in zip(dataRF, distance):
    amp = RF.data
    x = Distance + amp * 1

    x_zero = np.linspace(min(distance)-1, max(distance)+5, dataRF[0].stats.npts)

    ax2.plot(x, time_axis, '-', color='black')
    ax2.plot(x_zero, zeroline, '--', color='black', linewidth=0.75)
    ax2.fill_betweenx(time_axis, Distance, x, where=(x > Distance), color='red')
    ax2.fill_betweenx(time_axis, Distance, x, where=(x < Distance), color='grey')

# Regression Plot of 3 phases
plt.plot(distance, Ps_pred, '-', color='blue', label='Ps-wave', linewidth=2)
plt.plot(distance, M1_pred, '-', color='green', label='PpPs-wave', linewidth=2)
plt.plot(distance, M2_pred, '-', color='black', label='PpSs+PsPs-wave', linewidth=2)
plt.legend(loc='upper right')

ax2.set_ylim(time_bef_onset, time_aft_onset)
ax2.set_xlim(min(distance)-1, max(distance)+5)       # same with x_zero
ax2.invert_yaxis()
plt.xlabel('Distance (°)', fontsize=16)
plt.ylabel('Time (s)', fontsize=16)
plt.xticks(fontsize=16, color='black')
plt.yticks(fontsize=16, color='black')
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
plt.show()
