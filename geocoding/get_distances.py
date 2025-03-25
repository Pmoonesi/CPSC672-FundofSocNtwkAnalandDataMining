import numpy as np

def get_distance(p1, p2):
    # https://en.wikipedia.org/wiki/Great-circle_distance

    def geocentric_radius(phi):
        # https://en.wikipedia.org/wiki/Earth_radius

        a = 6378.1370 # equatorial radius
        b = 6356.7523 # polar radius

        numerator = (a**2 * np.cos(phi))**2 + (b**2 * np.sin(phi))**2
        denominator = (a * np.cos(phi))**2 + (b * np.sin(phi))**2
        return np.sqrt(numerator / denominator)

    # phi -> latitude, lambda -> longitude
    phi1,lambda1 = np.deg2rad(p1) 
    phi2,lambda2 = np.deg2rad(p2)

    sin_p1 = np.sin(phi1)
    sin_p2 = np.sin(phi2)
    cos_p1 = np.cos(phi1)
    cos_p2 = np.cos(phi2)
    Dlambda = np.abs(lambda2 - lambda1)
    cos_Dl = np.cos(Dlambda)

    temp = sin_p1*sin_p2 + cos_p1*cos_p2*cos_Dl

    temp = np.clip(temp, -1.0, 1.0) # due to floating point errors

    Ds = np.arccos(temp)
    R = geocentric_radius((phi1 + phi2) / 2)

    return R * Ds

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    # https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    # https://en.wikipedia.org/wiki/Great-circle_distance
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


from sklearn.metrics import pairwise_distances
import pandas as pd

stations_df = pd.read_csv('stations.csv')

D = pairwise_distances(stations_df[['lat', 'lng']], metric=get_distance)
np.savetxt('distance_matrix.txt',D, fmt="%.4f")