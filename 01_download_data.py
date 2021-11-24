# Download Data
# Created by Indra Rivaldi Siregar, Pertamina University

# 1. Import packages
import os
from obspy import read_inventory, read_events, UTCDateTime
from obspy.clients.fdsn import Client
from rf import read_rf, RFStream
from rf import iter_event_data
from tqdm import tqdm
import numpy as np
from shapely import speedups
speedups.disable()

# 2. Make folder files
folder_name = 'UGM'                    # station name as folder name                            # change this
path = r"D:\training"                                                                           # change this

folder = os.path.join(path, folder_name)

invfile = folder + "/" + 'station_' + folder_name + '.xml'
catfile = folder + "/" + 'catalog_events_' + folder_name + '.xml'
datafile = folder + "/" + 'stream_' + folder_name + '.h5'

# 3. Download station inventory
station = 'UGM'                                                         # change this
network = 'GE'                                                          # change this
channel = 'BH*'                                                         # change this
client = Client('GFZ')                                                  # change this

if not os.path.exists(folder):
    os.mkdir(folder)

loc_sta = []                                        # [sta_long, sta_lat]
if not os.path.exists(invfile) or os.path.exists(invfile):
    inventory = client.get_stations(network=network, channel=channel, level='response', station=station)
    inventory.write(invfile, 'STATIONXML')
    network = inventory[0]
    station = network[0]

    sta_long = station.longitude
    sta_lat = station.latitude
    sta_elev = station.elevation
    loc_sta.append(sta_long)
    loc_sta.append(sta_lat)
    # print('station_longitude:', sta_long)
    # print('station_latitude:', sta_lat)
inventory = read_inventory(invfile)
inventory.plot('ortho', resolution='i', color='red')


# 4. Download Catalog Events
st_time = UTCDateTime("2011-01-01T00:00:00")                        # change this
end_time = UTCDateTime("2016-10-01T00:00:00")                       # change this
min_mag = 6.7
max_mag = 9

if not os.path.exists(catfile) or os.path.exists(catfile):
    cli = Client()
    kwargs = {'starttime': st_time, 'endtime': end_time, 'longitude': loc_sta[0], 'latitude': loc_sta[1],
              'minradius': 30, 'maxradius': 90, 'minmagnitude': min_mag, 'maxmagnitude': max_mag}
    catalog = cli.get_events(**kwargs)
    catalog.write(catfile, 'QUAKEML')
catalog = read_events(catfile)
print(catalog.__str__(print_all=True))
fig = inventory.plot(show=False)
catalog.plot('global', resolution='i', fig=fig)

filename = 'catalog_' + np.str(inventory[0][0].code)                # station name
save = folder + "/" + filename + ".png"
fig.savefig(save)

# 5. Download Waveform of events
stream = RFStream()
if not os.path.exists(datafile) or os.path.exists(datafile):
    with tqdm(desc='Downloading data') as pbar:
        for s in iter_event_data(catalog, inventory, client.get_waveforms,
                                 request_window=(-25, 75), phase='P',
                                 tt_model="ak135", pbar=pbar):
            s.remove_response(inventory)
            stream.extend(s)
    stream.write(datafile, 'H5')
stream = read_rf(datafile, 'H5')
print(stream.__str__(extended=True))
print('Total events downloaded:', np.int(len(stream)/3))

# NOTE:
# there are 8 parameters that can change
