from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


# def data():
#     """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
#     df = access.data()
#     raise NotImplementedError

# def query(data):
#     """Request user input for some aspect of the data."""
#     raise NotImplementedError

# def view(data):
#     """Provide a view of the data that allows the user to verify some aspect of its quality."""
#     raise NotImplementedError

# def labelled(data):
#     """Provide a labelled set of data ready for supervised learning."""
#     raise NotImplementedError


from typing import Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import osmnx as ox
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import geopandas as gpd

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

def osm_buildings_data(latitude, longitude, distance=1):
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

def select_pp_data(conn, latitude, longitude):
    cursor = conn.cursor()
    cursor.execute("SELECT pp_data.postcode, latitude, longitude, primary_addressable_object_name, secondary_addressable_object_name, street, price, date_of_transfer FROM pp_data inner join postcode_data on pp_data.postcode = postcode_data.postcode WHERE latitude > %s and latitude < %s and longitude > %s AND longitude < %s", (latitude - 0.0045, latitude + 0.0045, longitude - 0.0045, longitude + 0.0045))

    # put into a dataframe
    pp_cambridge = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    return pp_cambridge

def join_pp_osm(pp_data, osm_data):
    merged1 = pd.merge(pp_data, osm_data, how='inner', 
         left_on=[pp_data['secondary_addressable_object_name'], pp_data['street'].str.lower()], 
         right_on=[osm_data['addr:housenumber'], osm_data['addr:street'].str.lower()])

    merged2 = pd.merge(pp_data, osm_data, how='inner', 
            left_on=[pp_data['primary_addressable_object_name'], pp_data['street'].str.lower()], 
            right_on=[osm_data['addr:housenumber'], osm_data['addr:street'].str.lower()])

    merged = pd.concat([merged1, merged2])

    # get rid of NaN in the addr:housenumber and addr:street columns
    merged = merged.dropna(subset=['addr:housenumber', 'addr:street'])
    return merged