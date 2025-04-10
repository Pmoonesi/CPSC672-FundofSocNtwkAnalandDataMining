import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations
import pickle
from sklearn.cluster import AgglomerativeClustering

stations_df = pd.read_csv('stations.csv', index_col=0)

D = np.loadtxt('distance_matrix.txt')

AC = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=0.1)

l = AC.fit_predict(D)

groups = {}

for i, label in enumerate(l):
    if label not in groups:
        groups[label] = []

    groups[label].append(i)

options = dict(filter(lambda x: len(x[1]) > 1, groups.items()))

mapping = {}

for grp, ids in options.items():

    print(stations_df[stations_df.index.isin(ids)])

    keep_id = int(input('which one to keep? '))

    while keep_id not in ids:
        keep_id = int(input('which one to keep? '))

    for id in ids:
        if id == keep_id:
            continue
        
        if id in mapping:
            print('pain')
            print(grp, id)

        mapping[id] = keep_id


# with open('proximity_mapping.pkl', 'wb') as f:
#     pickle.dump(mapping, f)

# with open('proximity_mapping.pkl', 'rb') as f:
#     mapping = pickle.load(f)

# remove 741 -> 373
# add 741 -> 42, 373 -> 42
# add 762 -> 136
# add 297 -> 296

del mapping[741]
mapping[741] = 42
mapping[373] = 42
mapping[762] = 136
mapping[297] = 296


def get_station_id(index):

    return stations_df[stations_df.index == index]['Id'].iloc[0]

def get_station_name(index):

    return stations_df[stations_df.index == index]['Name'].iloc[0]


new_mapping = {get_station_id(k): (get_station_id(v), get_station_name(v)) for k, v in mapping.items()}

with open('proximity_mapping_edited.pkl', 'wb') as f:
    pickle.dump(new_mapping, f)

# test_ids = [] # some test filtered ids
# illegal_ids = list(new_mapping.keys())
# for id in test_ids:

#     if id in illegal_ids:
#         print('illegal id seen in test ids.')


# illegal_ids = list(new_mapping.keys())

# grouped_stations_df = stations_df[~stations_df['Id'].isin(illegal_ids)]

# grouped_stations_df = grouped_stations_df.drop('count', axis=1)

# grouped_stations_df.to_csv('grouped_stations.csv')


## add the longitude and the latitude to the network

# df = pd.read_csv('grouped_stations.csv', index_col=0).sort_values('Id').reset_index(drop=True)

# data = dict([(str(x[0]), {'Latitude': x[1], 'Longitude': x[2]}) for x in list(zip(df['Id'], df['lat'], df['lng']))])

# import networkx as nx

# G = nx.read_gml('../processed/grouped_network.gml')

# nx.set_node_attributes(G, data)

# nx.write_gml(G, '../processed/grouped_network_geo.gml')
