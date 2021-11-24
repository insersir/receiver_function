# Preprocessing Data
# Created by Indra Rivaldi Siregar, Pertamina University

# 1. Import package
import numpy as np
import matplotlib.pyplot as plt
from rf import read_rf, RFStream
from scipy.fftpack import fft
import os
import time
from tqdm import tqdm
start_time = time.time()

# 2. Read data
station = 'UGM'                     # station name=folder name                  # change this

path = r"D:\training" + "/" + station
stream_file = path + "/" + "stream_" + station + ".h5"

stream = read_rf(stream_file)
stream = stream.sort(keys=['onset'])
# stream = stream[0:1080]
# stream.plot()
# print(stream.__str__(extended=True))
# print('=' * 100)

# 3. Frequency contents
tr = 2                          # trace                                        # change this
npts = stream[tr].stats.npts
dt = stream[tr].stats.delta     # interval sampling
fNy = 1. / (2. * dt)            # FNyquist
freq = np.linspace(0, fNy, npts // 2 + 1)   # frequency axis

# FFT for y-axis (magnitude)
X = fft(stream[tr].data)
X_mag = abs(X[0:np.size(freq)])

# Plotting Spectrum amplitude
tr_cha = stream[tr].stats.channel
plt.figure(figsize=(12, 6), facecolor="#F5F5F5")
plt.title('Frequency-Domain Data\nAmplitude Spectrum', fontsize=13, fontweight='bold', color='black')
plt.plot(freq, X_mag, 'k', label='%s' % tr_cha)
plt.legend()
plt.xlabel('Frequency (Hz)', fontsize=13)
plt.show()

# Time axis
time_bef_onset = np.round(stream[0].stats.onset - stream[0].stats.starttime)*-1
time_aft_onset = np.round(stream[0].stats.endtime - stream[0].stats.onset)
time_axis = np.linspace(time_bef_onset, time_aft_onset, stream[0].stats.npts)
# print(len(time_axis))
# print(time_bef_onset, time_aft_onset)

# 4. Filtering data
# Detrend filter
stream_detrend = stream.copy().detrend('linear')

# Bandpass filter
fmin = 0.1                                                              # change this
fmax = 1.0                                                              # change this
order = 2                                                               # change this
stream_bandpass = stream_detrend.copy().filter('bandpass', freqmin=fmin, freqmax=fmax, corners=order)

# 5. Plotting data & remove bad events
chan_E = stream[0].stats.channel
chan_N = stream[1].stats.channel
chan_Z = stream[2].stats.channel

# stream before bandpass
stream_E = stream.select(channel=chan_E)
stream_N = stream.select(channel=chan_N)
stream_Z = stream.select(channel=chan_Z)

# stream after bandpass
stream_filt_E = stream_bandpass.select(channel=chan_E)
stream_filt_N = stream_bandpass.select(channel=chan_N)
stream_filt_Z = stream_bandpass.select(channel=chan_Z)

st_merge = [stream_filt_E, stream_E, stream_filt_N, stream_N, stream_filt_Z, stream_Z]
data_length = st_merge[0]                       # length data for each component

# 6. Signal to Noise Ratio (SNR)
from func import snr
snr_bef_filt = snr(stream, 20)
snr_aft_filt = snr(stream_bandpass, 20)

snr_merge = []                                  # snr aft_band E, snr_bef_band E, snr aft_band N, snr_bef_band N, ...
for z in range(len(snr_bef_filt)):
    SNR = snr_aft_filt[z], snr_bef_filt[z]
    snr_merge.extend(SNR)

snr_merge = np.array_split(snr_merge, len(data_length))             # split data according to the event
# print(snr_merge)
snr_merge = np.transpose(snr_merge)                                 # split to every component
# print(snr_merge)

# 7. Make new folder to save figure
new_folder = path + "/" + 'Pictures of Events'
if not os.path.exists(new_folder):
    os.mkdir(new_folder)

# 8. Plotting and Save figure
colors = (['r'] + ['k']) * 3
labels = ([' after'] + [' before']) * 3

good_stream = RFStream([])
with tqdm(total=len(data_length), desc='Saving Figures', disable=False) as pbar:
    for j in range(len(data_length)):
        fig, ax = plt.subplots(nrows=len(st_merge), ncols=1, figsize=(14, 8), facecolor="#F5F5F5")
        for i in range(len(st_merge)):
            st = st_merge[i][j]
            # print(st)
            ax[0].set_title('Filtering Data\n' + station + '.onset time: ' + np.str(stream_filt_E[j].stats.onset),
                            fontweight='bold', fontsize=14, color='black')
            ax[0].set_title('Event: %s' % j + ' of %s' % (len(stream_filt_E) - 1), fontstyle='italic',
                            fontsize=11, loc='right')
            ax[i].plot(time_axis, st.data, '-', color=colors[i], label=np.str(st.stats.channel)+labels[i] +
                       ' with SNR (dB): ' + np.str(snr_merge[i][j]))
            ax[i].legend(loc='upper left')
            ax[i].set_xlim(time_bef_onset, time_aft_onset)
            ax[5].set_xlabel('Time (s)')

        ons = stream_filt_E[j].stats.onset
        filename = np.str(ons.date) + "T" + np.str(ons.hour) + "." + np.str(ons.minute) + "." + np.str(ons.second)
        save = new_folder + "/" + filename + ".png"             # formats: .jpg, .png, .pdf, and others
        plt.rcParams.update({'figure.max_open_warning': 0})     # to avoid warning due to limit opening figure
        plt.savefig(save)
        pbar.update(1)

# 8. Save files
new_stream = path + "/" + 'stream_sorted_' + station + '.h5'
stream_bandpass.write(new_stream, 'H5')
print('\nTotal Events:', np.int(len(stream_bandpass)/3))

# 9. Calculate Running Time
end_time = time.time()
time_min = np.round((end_time-start_time)/60, 2)
print('Running Time:', time_min, "minutes")
