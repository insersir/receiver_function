# Testing Deconvolution Parameters (Alpha and Water-level)
# 1. Import package
# Created by Indra Rivaldi Siregar, Pertamina University

from rf import read_rf
import numpy as np
import matplotlib.pyplot as plt
from func import norm
import time
start_time = time.time()

# 2. Read data
station = 'UGM'                             # station name = folder name                     # change this

path = r"D:\training" + "/" + station
stream_file = path + "/" + "stream_sorted_" + station + ".h5"

stream = read_rf(stream_file)
stream = stream.sort(keys=['onset'])

# 3. Rotation and Deconvolution
alpha1 = 0.5                                                            # change this
alpha2 = 2.5                                                            # change this
alpha3 = 7.5                                                            # change this
waterlevel1 = 0.005                                                     # change this
waterlevel2 = 0.005                                                     # change this
waterlevel3 = 0.005                                                     # change this


# Don't modify the lines below ###################################

# gauss = (np.sqrt(2)*alpha) / (2*np.pi)     # gauss = 0.225 * alpha // gauss = 0.5 -> alpha = 2.222
freq_decon1 = stream.copy().rf(deconvolve='freq', rotate='ZNE->LQT', gauss=(np.sqrt(2)*alpha1) / (2*np.pi),
                               waterlevel=waterlevel1, normalize=1)

freq_decon2 = stream.copy().rf(deconvolve='freq', rotate='ZNE->LQT', gauss=(np.sqrt(2)*alpha2) / (2*np.pi),
                               waterlevel=waterlevel2, normalize=1)

freq_decon3 = stream.copy().rf(deconvolve='freq', rotate='ZNE->LQT', gauss=(np.sqrt(2)*alpha3) / (2*np.pi),
                               waterlevel=waterlevel3, normalize=1)

# 4. Save files to these
rf_freq1 = path + "/" + 'rf_freq_test1_' + station + '.h5'         # water level deconvolution
rf_freq2 = path + "/" + 'rf_freq_test2_' + station + '.h5'         # water level deconvolution
rf_freq3 = path + "/" + 'rf_freq_test3_' + station + '.h5'         # water level deconvolution

freq_decon_trim1 = freq_decon1.trim2(-5, 25, 'onset')
freq_decon_trim1.write(rf_freq1, "H5")

freq_decon_trim2 = freq_decon2.trim2(-5, 25, 'onset')
freq_decon_trim2.write(rf_freq2, "H5")

freq_decon_trim3 = freq_decon3.trim2(-5, 25, 'onset')
freq_decon_trim3.write(rf_freq3, "H5")

# 5. Plotting
rf1 = freq_decon_trim1.select(component='Q')
rf2 = freq_decon_trim2.select(component='Q')
rf3 = freq_decon_trim3.select(component='Q')

rf1 = rf1.stack()
rf2 = rf2.stack()
rf3 = rf3.stack()

rf1 = norm(rf1[0].data)
rf2 = norm(rf2[0].data)
rf3 = norm(rf3[0].data)

xaxis = np.linspace(-5, 25, len(rf1))
zeroline = xaxis*[0]

total = "Total Events: " + np.str(len(freq_decon_trim1.select(component='Q')))
plt.subplot(3, 1, 1)
plt.title('Comparison of Time Deconvolution Parameters', fontstyle='normal', fontweight='bold', fontsize=16)
plt.title(total, fontstyle='italic', fontsize=11, loc='right')

plt.plot(xaxis, rf1, 'k-', label='alpha: ' + np.str(alpha1) + ' & w_level: ' + np.str(waterlevel1))
plt.plot(xaxis, zeroline, 'k--', lw=0.8)
plt.fill_between(xaxis, rf1, where=(rf1 > zeroline), color='black')
plt.xlim(-5, 25)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(xaxis, rf2, 'k-', label='alpha: ' + np.str(alpha2) + ' & w_level: ' + np.str(waterlevel2))
plt.plot(xaxis, zeroline, 'k--', lw=0.8)
plt.fill_between(xaxis, rf2, where=(rf2 > zeroline), color='black')
plt.xlim(-5, 25)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(xaxis, rf3, 'k-', label='alpha: ' + np.str(alpha3) + ' & w_level: ' + np.str(waterlevel3))
plt.plot(xaxis, zeroline, 'k--', lw=0.8)
plt.fill_between(xaxis, rf3, where=(rf3 > zeroline), color='black')
plt.xlim(-5, 25)
plt.legend()
plt.show()


# 6. Calculate Running Time and Plot RF
end_time = time.time()
time_min = np.round((end_time-start_time)/60, 2)
time_sec = np.round((end_time-start_time), 2)

if time_min >= 1:
    print('Running Time:', time_min, "minutes")
else:
    print('Running Time:', time_sec, "seconds")


# # 7. Plot RF
# kw = {'fillcolors': ('black', 'grey'), 'show_vlines': False, 'stack_height': 1.5,
#       'info': (('back_azimuth', u'baz (°)', 'C0'), ('distance', u'dist (°)', 'C3'))
#       }
# freq_decon_trim1[0:40].sort(['distance']).select(component='Q').plot_rf(**kw)
# freq_decon_trim2[0:40].sort(['distance']).select(component='Q').plot_rf(**kw)
# freq_decon_trim3[0:40].sort(['distance']).select(component='Q').plot_rf(**kw)
# plt.show()
