# Mapping
# Created by Indra Rivaldi Siregar, Pertamina University

import pygmt
import pandas as pd
import numpy as np

data = pd.read_excel(r"D:\test\crustal_thickness.xlsx")
sta = data.Station
long = data.Longitude
lat = data.Latitude
H = np.round(data.H.tolist(), 2)
k = data.k

volcanic = pd.read_excel(r"D:\test\volcanic.xlsx")
long1 = volcanic.long
lat1 = volcanic.lat

fig = pygmt.Figure()
fig.coast(region=[106, 113, -9.5, -4.5],    # region= long min, long max, lat min, lat max
          projection="M12c", land="lightgray", water="blue", borders="1/0.5p",
          shorelines="1/0.5p", frame="ag")

pygmt.makecpt(cmap="inferno", series=[H.min(), H.max()])        # cmap = inferno, batlow, topo and others
fig.plot(x=long, y=lat, style="c0.17c", color=H, cmap=True, pen="black")
fig.colorbar(frame='af+l"crustal thickness (km)"')
fig.plot(x=long1, y=lat1, style="t0.25c", color='red', pen="black")

for i in range(len(volcanic)):
    fig.text(text=volcanic.code[i], x=long1[i], y=lat1[i], font="5p,Helvetica-Bold,black")
fig.show()

# Save
filename = "test_plot_pygmt"
save = r"D:\test" + "/" + filename + ".pdf"
fig.savefig(save)
