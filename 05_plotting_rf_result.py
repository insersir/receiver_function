from rf import read_rf
import matplotlib.pyplot as plt

data = read_rf(r"D:\training\UGM\good_rf_iter_UGM.h5")
# print(data.__str__(extended=True))

kw = {'fillcolors': ('red', 'white'), 'show_vlines': False,
      'info': (('back_azimuth', u'baz (°)', 'C0'), ('distance', u'dist (°)', 'C3'))
      }
data.plot_rf(**kw)
plt.show()
