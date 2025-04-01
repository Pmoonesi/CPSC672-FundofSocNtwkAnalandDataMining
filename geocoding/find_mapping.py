import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations

# stations_df = pd.read_csv('stations.csv', index_col=0)

# stations_df['count'] = stations_df.groupby(['lat', 'lng'])['Id'].transform('count')
# duplicates_df = stations_df[stations_df['count'] > 1]
# ids = duplicates_df.groupby(['lat', 'lng'])['Id'].apply(list).to_list()

# for i, id_grp in enumerate(ids):
#     # print(f"group #{i}:", id_grp)
#     print(stations_df[stations_df['Id'].isin(id_grp)].index)

# np.savetxt('temp.txt', stations_df['Name'].sort_values(), fmt='%s')

# def sort_address(address):
#     if '/' not in address:
#         return address
    
#     part1, _, part2 = address.partition('/')

#     part1, part2 = part1.strip(), part2.strip()

#     if part2 > part1:
#         return part1 + ' / ' + part2
#     else:
#         return part2 + ' / ' + part1
    

# sorted_addresses = sorted(list(map(sort_address, stations_df['Name'])))

# np.savetxt('temp.txt', sorted_addresses, fmt='%s')

# print(stations_df[stations_df['Name'].str.contains('Sunnybrook Health Centre')])

D = np.loadtxt('distance_matrix.txt')

# from sklearn.cluster import AgglomerativeClustering

# AC = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=0.1)

# l = AC.fit_predict(D)

# groups = {}

# for i, label in enumerate(l):
#     if label not in groups:
#         groups[label] = []

#     groups[label].append(i)

# options = dict(filter(lambda x: len(x[1]) > 1, groups.items()))

# mapping = {}

# for grp, ids in options.items():

#     print(stations_df[stations_df.index.isin(ids)])

#     keep_id = int(input('which one to keep? '))

#     while keep_id not in ids:
#         keep_id = int(input('which one to keep? '))

#     for id in ids:
#         if id == keep_id:
#             continue
        
#         if id in mapping:
#             print('pain')
#             print(grp, id)

#         mapping[id] = keep_id


# import pickle

# with open('proximity_mapping.pkl', 'wb') as f:
#     pickle.dump(mapping, f)


# from get_distances import get_distance

# temp_df = stations_df[stations_df['Name'].str.contains("King St W / York St")][['Name', 'lat','lng']]
# print(temp_df)
# print(get_distance(temp_df.iloc[0, 1:3].to_list(), temp_df.iloc[1, 1:3].to_list()))


# import pickle

# with open('proximity_mapping.pkl', 'rb') as f:
#     mapping = pickle.load(f)

### remove 741 -> 373
### add 741 -> 42, 373 -> 42
### add 762 -> 136
### add 297 -> 296

# del mapping[741]
# mapping[741] = 42
# mapping[373] = 42
# mapping[762] = 136
# mapping[297] = 296


# def get_station_id(index):

#     return stations_df[stations_df.index == index]['Id'].iloc[0]

# def get_station_name(index):

#     return stations_df[stations_df.index == index]['Name'].iloc[0]


# new_mapping = {get_station_id(k): (get_station_id(v), get_station_name(v)) for k, v in mapping.items()}

# print(new_mapping)
# with open('proximity_mapping_edited.pkl', 'wb') as f:
#     pickle.dump(new_mapping, f)

# test_ids = [] # some test filtered ids
# illegal_ids = list(new_mapping.keys())
# for id in test_ids:

#     if id in illegal_ids:
#         print('illegal id seen in test ids.')


# illegal_ids = list(new_mapping.keys())

# grouped_stations_df = stations_df[~stations_df['Id'].isin(illegal_ids)]

# grouped_stations_df = grouped_stations_df.drop('count', axis=1)

# grouped_stations_df.to_csv('grouped_stations.csv')

grp_stations_df = pd.read_csv('grouped_stations.csv', index_col=0).reset_index(drop=True)

DD = np.loadtxt('distance_matrix_grp.txt')

proximity = set(zip(*np.where(DD < 0.1)))

diagonal = set([(i,i) for i in range(DD.shape[0])])

from get_distances import get_distance

for pair in list(proximity - diagonal):

    temp_df = grp_stations_df[grp_stations_df.index.isin(pair)][['Name', 'lat','lng']]

    print(temp_df)
    print(get_distance(temp_df.iloc[0, 1:3].to_list(), temp_df.iloc[1, 1:3].to_list()))

from sklearn.cluster import AgglomerativeClustering

AC = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=0.1)

l = AC.fit_predict(DD)

groups = {}

for i, label in enumerate(l):
    if label not in groups:
        groups[label] = []

    groups[label].append(i)

print(max(l))
print(dict(filter(lambda x: len(x[1]) > 1, groups.items())))