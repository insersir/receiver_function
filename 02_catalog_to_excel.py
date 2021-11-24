# Save catalog to excel
# Created by Indra Rivaldi Siregar, Pertamina University

from obspy import read_events
import pandas as pd

station = 'JAGI'                         # station name=folder name            # change this

path = r"D:\training" + "/" + station

cat_file = path + "/" + "catalog_events_" + station + ".xml"
catalog = read_events(cat_file)
# catalog = catalog[0:1]
# print(catalog)

origin_time = []
magnitude = []
lon, lat, depth = [], [], []
for i in range(len(catalog)):
    event = catalog[i]
    origins = event.origins[0]
    lon.append(origins.longitude)
    lat.append(origins.latitude)
    depth.append(origins.depth/1000)
    origin_time.append(origins.time)

    magnitudes = event.magnitudes[0].mag
    magnitude.append(magnitudes)

save_catalog = pd.DataFrame({"Origin Time": origin_time, "Longitude": lon, "Latitude": lat, "Magnitude": magnitude,
                            "Depth (km)": depth})
save_catalog.to_excel(path + "/" + "catalog_excel_" + station + ".xlsx")
