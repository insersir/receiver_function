# Rotation and Deconvolution
# Created by Indra Rivaldi Siregar, Pertamina University

# 1. Import package
from rf import read_rf
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

# 2. Read data
station = 'UGM'        # station name = folder name                             # change this

path = r"D:\training" + "/" + station
stream_file = path + "/" + "stream_sorted_" + station + ".h5"

stream = read_rf(stream_file)
stream = stream.sort(keys=['onset'])
# print(stream[0].stats)

# 3. Rotation and Deconvolution
alpha = 2.5                                                                    # change this

gauss = (np.sqrt(2)*alpha) / (2*np.pi)     # gauss = 0.225 * alpha // gauss = 0.5 -> alpha = 2.222
iter_decon = stream.copy().rf(deconvolve='iterative', rotate='ZNE->LQT', gauss=gauss,
                              itmax=750, minderr=0.0005, normalize=1)

freq_decon = stream.copy().rf(deconvolve='freq', rotate='ZNE->LQT', gauss=gauss,
                              waterlevel=0.005, normalize=1)

# 4. Save files to these
rf_iter = path + "/" + 'rf_iter_' + station + '.h5'         # iterative time deconvolution
rf_freq = path + "/" + 'rf_freq_' + station + '.h5'         # water level deconvolution

iter_decon_trim = iter_decon.trim2(-5, 25, 'onset')
iter_decon_trim.write(rf_iter, "H5")

freq_decon_trim = freq_decon.trim2(-5, 25, 'onset')
freq_decon_trim.write(rf_freq, "H5")

# 5. Calculate Running Time and Plot RF
end_time = time.time()
time_min = np.round((end_time-start_time)/60, 2)
time_sec = np.round((end_time-start_time), 2)

if time_min >= 1:
    print('Running Time:', time_min, "minutes")
else:
    print('Running Time:', time_sec, "seconds")

# 6. Plot Receiver Function
kw = {'fillcolors': ('black', 'grey'), 'show_vlines': False,
      'info': (('back_azimuth', u'baz (°)', 'C0'), ('distance', u'dist (°)', 'C3'))
      }

iter_decon_trim[0:39].sort(['distance']).select(component='Q').plot_rf(**kw)
freq_decon_trim[0:39].sort(['distance']).select(component='Q').plot_rf(**kw)
# freq_decon_trim.sort(['distance']).select(component='L').plot_rf(**kw)
# iter_decon_trim.sort(['distance']).select(component='L').plot_rf(**kw)
plt.show()
