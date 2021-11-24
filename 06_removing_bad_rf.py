# Removing Bad Receiver Function
# Created by Indra Rivaldi Siregar, Pertamina University

import time
import matplotlib.pyplot as plt
from rf import read_rf, RFStream
import numpy as np
from matplotlib.widgets import RadioButtons
start_time = time.time()

# 1.Read RF File (after deconvolution)
station = 'UGM'                        # station name=folder name            # change this

path = r"D:\training" + "/" + station

iterdecon = read_rf(path + "/" + "rf_iter_" + station + ".h5")
freqdecon = read_rf(path + "/" + "rf_freq_" + station + ".h5")

# Don't modify the lines below ###################################
# 2. Save good RF to these
save1 = path + "/" + "good_rf_iter_" + station + ".h5"
save2 = path + "/" + "good_rf_freq_" + station + ".h5"

print("Station:", iterdecon[0].stats.station)
print("Longitude:", iterdecon[0].stats.station_longitude)
print("Latitude:", iterdecon[0].stats.station_latitude)
print("Elevation:", iterdecon[0].stats.station_elevation)
print("="*150)

# 3. Q-component
dataQ_iterdecon = iterdecon.select(component='Q')
dataQ_freqdecon = freqdecon.select(component='Q')

Q_merge = [dataQ_iterdecon, dataQ_freqdecon]
data_length = Q_merge[0]

print("Total data of RF iterative time:", len(dataQ_iterdecon))
print("Total data of RF frequency     :", len(dataQ_freqdecon))
print("="*150)

# 3.Removing Bad RF
good_RF_iterdecon = RFStream([])
good_RF_freqdecon = RFStream([])

# time axis
time_bef_onset = np.round(iterdecon[0].stats.onset - iterdecon[0].stats.starttime)*-1
time_aft_onset = np.round(iterdecon[0].stats.endtime - iterdecon[0].stats.onset)
time_axis = np.linspace(time_bef_onset, time_aft_onset, iterdecon[0].stats.npts)

zeroline = dataQ_iterdecon[0].stats.npts * [0]
tittle = ['Iterative\n', 'Frequency\n']

for j in range(len(data_length)):
    fig, ax = plt.subplots(nrows=len(Q_merge), ncols=1, figsize=(14, 7), facecolor="#F5F5F5")
    for i in range(len(Q_merge)):
        Q = Q_merge[i][j]
        ax[i].set_title(tittle[i] + station + '.onset time: ' + np.str(dataQ_iterdecon[j].stats.onset),
                        fontweight='normal', fontsize=14, color='black')
        ax[0].set_title('Mag: '+np.str(dataQ_iterdecon[j].stats.event_magnitude) +
                        ' | Event: %s' % j + ' of %s' % (len(dataQ_iterdecon) - 1), fontstyle='italic',
                        fontsize=11, loc='right')
        ax[i].plot(time_axis, Q.data, '-k')
        ax[i].fill_between(time_axis, Q.data, where=(Q.data > zeroline), color='black')
        ax[i].fill_between(time_axis, Q.data, where=(Q.data < zeroline), color='grey')
        ax[i].plot(time_axis, zeroline, '--k', lw=0.9)
        ax[i].set_xlim(time_bef_onset, time_aft_onset)
        ax[1].set_xlabel('Time (s)', fontsize=14)

    ax_color = plt.axes([0.88, 0.82, 0.25, 0.25], facecolor='#F5F5F5')
    ax_color.spines['left'].set_color('#F5F5F5')
    ax_color.spines['bottom'].set_color('#F5F5F5')
    color_button = RadioButtons(ax_color, ['Select\n  RF'],
                                [False], activecolor='r')  # use [] in false for deactivate the button color

    def click(event):
        print(dataQ_iterdecon[j])
        good_RF_iterdecon.append(dataQ_iterdecon[j])
        good_RF_freqdecon.append(dataQ_freqdecon[j])
    color_button.on_clicked(click)
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    plt.subplots_adjust(hspace=0.4, right=0.87)
    plt.show()
print("="*150)

# 5. Save File Good RF
good_RF_iterdecon.write(save1, "H5")
good_RF_freqdecon.write(save2, "H5")
print("RF iterative time after removing:", len(good_RF_iterdecon))
print("RF frequency after removing     :", len(good_RF_freqdecon))
print("="*150)

end_time = time.time()
print('Running Time:', np.round((end_time-start_time)/60, 2), "minutes")

# 6.Plot good RF
kw = {'fillcolors': ('black', 'grey'), 'show_vlines': False,
      'info': (('back_azimuth', u'baz (°)', 'C0'), ('distance', u'dist (°)', 'C3'))
      }
good_RF_iterdecon.sort(['distance']).select(component='Q').plot_rf(**kw)
good_RF_freqdecon.sort(['distance']).select(component='Q').plot_rf(**kw)
plt.show()
