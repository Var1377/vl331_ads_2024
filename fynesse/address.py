# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

from typing import Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import osmnx as ox
import matplotlib.pyplot as plt
import warnings

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """

    pois = ox.features_from_point((latitude, longitude), tags)

    pois_df = pd.DataFrame(pois)
    pois_df['latitude'] = pois_df.apply(lambda row: row.geometry.centroid.y, axis=1)
    pois_df['longitude'] = pois_df.apply(lambda row: row.geometry.centroid.x, axis=1)
    poi_counts = {}

    poi_types = ["amenity", "historic", "leisure", "shop", "tourism", "religion", "memorial"]

    for tag in poi_types:
        if tag in pois_df.columns:
            poi_counts[tag] = pois_df[tag].notnull().sum()
        else:
            poi_counts[tag] = 0

    return poi_counts

def cluster_locations(locations, tags, n_clusters=3):
    """
    Cluster locations based on latitude and longitude.
    Args:
        locations (list): List of dictionaries with 'latitude' and 'longitude' keys.
        n_clusters (int): Number of clusters to form. Default is 3.
    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of locations in each cluster.
    """
    location_poi_counts = []
    for location_name, (latitude, longitude) in locations.items():
        poi_counts = count_pois_near_coordinates(latitude, longitude, tags)
        poi_counts['Location'] = location_name
        location_poi_counts.append(poi_counts)

    poi_counts_df = pd.DataFrame(location_poi_counts)
    poi_counts_df = poi_counts_df.set_index('Location')

    # Select features for clustering (exclude any non-numerical columns)
    features = poi_counts_df.select_dtypes(include=['number'])

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(scaled_features)

    # Add cluster labels to the dataframe
    poi_counts_df['Cluster'] = kmeans.labels_
    return poi_counts_df

def osm_buildings_data(lat, lon, distance=1000):
    # Define the tags to retrieve building information
    building_tags = {
        "building": True,
    }

    # Retrieve building data from OpenStreetMap
    buildings = ox.geometries_from_bbox(latitude + distance/222, latitude - distance/222, longitude + distance/222, longitude - distance/222, building_tags)
    graph = ox.graph_from_bbox(latitude + distance/222, latitude - distance/222, longitude + distance/222, longitude - distance/222)

    nodes, edges = ox.graph_to_gdfs(graph)

    buildings["area_sqm"] = buildings.geometry.to_crs({'init': 'epsg:3395'}).area

    # Filter buildings with full address information
    buildings_with_address = buildings.dropna(subset=["addr:housenumber", "addr:street"])

    buildings_without_address = buildings[~buildings.index.isin(buildings_with_address.index)]

    # Plot the buildings
    fig, ax = plt.subplots(figsize=(10, 10))

    edges.plot(ax=ax, linewidth=0.5, color='grey')

    ax.set_xlim(longitude - distance/222, longitude + distance/222)
    ax.set_ylim(latitude - distance/222, latitude + distance/222)

    # Plot buildings with address
    buildings_with_address.plot(ax=ax, color='blue', alpha=0.7, label='With Address')

    # Plot buildings without address
    buildings_without_address.plot(ax=ax, color='red', alpha=0.7, label='Without Address')

    plt.legend()
    plt.show()

    return buildings

def select_pp_data(conn, lat, lon):
    cursor = conn.cursor()
    cursor.execute("SELECT pp_data.postcode, latitude, longitude, primary_addressable_object_name, secondary_addressable_object_name, street FROM pp_data inner join postcode_data on pp_data.postcode = postcode_data.postcode WHERE latitude > %s and latitude < %s and longitude > %s AND longitude < %s", (latitude - 0.0045, latitude + 0.0045, longitude - 0.0045, longitude + 0.0045))

    # put into a dataframe
    pp_cambridge = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    return pp_cambridge

def join_pp_osm(pp_data, osm_data):
    merged1 = pd.merge(pp_cambridge, buildings, how='inner', 
         left_on=[pp_cambridge['secondary_addressable_object_name'], pp_cambridge['street'].str.lower()], 
         right_on=[buildings['addr:housenumber'], buildings['addr:street'].str.lower()])

    merged2 = pd.merge(pp_cambridge, buildings, how='inner', 
            left_on=[pp_cambridge['primary_addressable_object_name'], pp_cambridge['street'].str.lower()], 
            right_on=[buildings['addr:housenumber'], buildings['addr:street'].str.lower()])

    merged = pd.concat([merged1, merged2])

    # get rid of NaN in the addr:housenumber and addr:street columns
    merged = merged.dropna(subset=['addr:housenumber', 'addr:street'])[["addr:housenumber", "addr:street","postcode"]]
    # remove duplicates
    # merged = merged.drop_duplicates()
    return merged