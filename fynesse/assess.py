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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import multiprocessing as mp
import dask_geopandas as ddg
import re
import hashlib
import os
import dask.dataframe as dd
import seaborn as sns

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

def select_pp_data(latitude, longitude):
    cursor = access.conn.cursor()
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

def parse_height_to_meters(height_series):
    def convert_to_meters(height):
        if pd.isnull(height):
            return None
        
        height_str = str(height).strip().lower()
        pure_number_pattern = r'^([\d,.]+)\s*(m|meter|meters|metre|metres|ft|foot|feet)?$'
        feet_inches_pattern = r"^(\d+)'\s*(\d+)?\"?$"
        
        # Try pure number with optional unit
        match = re.match(pure_number_pattern, height_str)
        
        if match:
            value, unit = match.groups()
            # Replace comma with dot for decimal conversion if necessary
            value = value.replace(',', '.')
            
            try:
                value = float(value)
            except ValueError:
                return None
            
            # Define conversion factors
            unit = unit.lower() if unit else 'm'  # Assume meters if no unit provided
            
            if unit in ['m', 'meter', 'meters', 'metre', 'metres']:
                return value
            elif unit in ['ft', 'foot', 'feet']:
                return value * 0.3048  # 1 foot = 0.3048 meters
            else:
                return None
        
        # Try feet and inches pattern
        match = re.match(feet_inches_pattern, height_str)
        if match:
            feet, inches = match.groups()
            try:
                feet = int(feet)
                inches = int(inches) if inches else 0
            except ValueError:
                return np.nan  # Unable to convert to integers
            
            total_meters = feet * 0.3048 + inches * 0.0254
            return round(total_meters, 4)  # Rounded to 4 decimal places
        
        # If no pattern matches, return None
        return None
    
    # Apply the conversion to each element in the Series
    return height_series.apply(convert_to_meters)

def setup_cutoffs(cutoffs, df):
    for radius, _ in cutoffs:
        df[f"{radius}m"] = df["geometry"].to_crs(epsg=6933).buffer(radius).to_crs(epsg=4326).simplify(radius/10)


def collect_features_for_condition(oas, cutoffs, cache_dir, condition, name):
    keys_to_keep = ["code", "osmid", "distance", "area"]

    condition_hash = hashlib.md5(condition.encode()).hexdigest()
    filepath = os.path.join(cache_dir, f"{name}_{condition_hash}.csv")

    if os.path.exists(filepath):
        print(f"Already processed {condition}, loading from file...")
        df = dd.read_csv(f"{filepath}/*.part", sep="|", index_col=False)
        if df.columns.equals(keys_to_keep):
            return df.set_index("code")
        else:
            df = df[keys_to_keep]
            df.to_csv(filepath, sep="|", index=False)
            return df.set_index("code")
    else:
        print(f"Processing {condition}...")
        # Load data from database
        # Make sure to pass 'conn' and 'available_parallelism' if not defined globally
        gdf_sql = pd.read_sql(
            f"SELECT *, ST_AsText(geometry) as wkt FROM osm_features WHERE {condition}",
            access.conn
        )
        gdf = ddg.from_geopandas(access.load_geometry_from_wkt(gdf_sql), npartitions=mp.cpu_count())
        size = len(gdf)

        print(f"{condition}")
        print(f"Found {size} features")

        for r, cutoff in cutoffs:
            if size < cutoff:
                radius = r
                break

        print(f"Looking within {radius}m")

        oas_reset = oas.set_geometry(oas.geometry.buffer(radius)).reset_index(drop=True)
        joined = ddg.sjoin(gdf, oas_reset, predicate="intersects")
        joined = joined.compute()

        print(f"Found {len(joined)} relationships, calculating distances...")
        joined["distance"] = gpd.GeoSeries(
            joined["geometry_left"], crs="EPSG:4326"
        ).to_crs(epsg=6933).distance(
            gpd.GeoSeries(joined["geometry_right"], crs="EPSG:4326").to_crs(epsg=6933)
        )

        print("Saving...")
        df = joined[keys_to_keep]
        df = dd.from_pandas(df, npartitions=mp.cpu_count())
        df.to_csv(filepath, sep="|", index=False)
        return df.set_index("code")
    

def calculate_scores(df, name, oas_with_diameter, cache_dir='oa_scores'):
    """
    Calculates scores for each Output Area (OA) based on provided DataFrame and caches the results.

    Parameters:
        df (dask.dataframe.DataFrame): DataFrame containing the features for calculation.
        name (str): Name for caching the results.
        oas_with_diameter (pandas.DataFrame): DataFrame containing 'diameter' for each OA.
        cache_dir (str, optional): Directory to store cached scores. Defaults to 'oa_scores'.

    Returns:
        pandas.Series: Series containing scores indexed by 'code'.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{name}.csv")
    
    if os.path.exists(cache_file):
        print(f"Loading cached scores for {name} from {cache_file}")
        return pd.read_csv(cache_file).set_index("code")["score"]
    
    # Ensure 'area' column is numeric
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    
    # Compute mean area excluding zeros and NaNs
    median_area = df[df["area"] > 0.0]["area"].compute().median()
    if np.isnan(median_area):
        median_area = 1.0
    else:
        median_area = min(median_area, 1.0)
    df["area"] = df["area"].fillna(median_area).replace(0.0, median_area)
    print(f"Median area: {median_area}")
    
    # Join the DataFrame with oas_with_diameter to include the diameter column
    df_with_diameter = df.merge(
        oas_with_diameter[["diameter"]],
        how="left",
        left_index=True,
        right_index=True
    )
    
    # Calculate the scores
    result = df_with_diameter.groupby(df_with_diameter.index).apply(
        lambda group: ((group["area"]) / (group["distance"] + group["diameter"] / 4)).sum(),
        meta=('score', 'float64')
    ).compute()
    
    # Save the results to cache
    result = result.reset_index()
    result.columns = ['code', 'score']
    result.to_csv(cache_file, index=False)
    print(f"Scores saved to {cache_file}")
    
    return result.set_index('code')["score"]


def analyse_osm_feature(condition, name, oas, oas_with_student_proportion, Y, Y_logit):
    features = collect_features_for_condition(condition, name)
    scores = calculate_scores(features, name)
    scores = scores.rename("score").to_frame()
    plottable = oas.merge(scores, how="left", left_on="code", right_index=True).fillna(0).compute()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # plot a histogram of the scores
    sns.histplot(plottable["score"], bins=100, kde=True, color='blue', label='Score', alpha=0.5, ax=ax)
    ax.set_title(f"Distribution of Scores for {name}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    plt.show()

    # plot the scores on a map side by side with the student proportion
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    plottable.plot(column="score", ax=axes[0], legend=True, cmap="coolwarm", legend_kwds={"label": "Score"})
    axes[0].set_title(f"Score for {name}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    oas_with_student_proportion.plot(column="student proportion", ax=axes[1], legend=True, cmap="coolwarm", legend_kwds={"label": "Student Proportion"})
    axes[1].set_title("Student Proportion")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    plt.tight_layout()
    plt.show()

    # plot the two scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    sns.regplot(x=plottable["score"], y=Y, scatter_kws={'s': 10}, line_kws={'color': 'red'}, ax=axes[0])
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Student Proportion")
    axes[0].set_title("Score vs Student Proportion")
    axes[0].grid(True)

    sns.regplot(x=plottable["score"], y=Y_logit, scatter_kws={'s': 10}, line_kws={'color': 'red'}, ax=axes[1])
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Logit Transformed Student Proportion")
    axes[1].set_title("Score vs Logit Transformed Student Proportion")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_all_feature_correlations(scores_df, Y, Y_logit):
    """
    Plot correlation graphs for all features against student proportion and its logit transform.
    
    Parameters:
        scores_df (pd.DataFrame): DataFrame containing feature scores
        Y (pd.Series): Student proportion
        Y_logit (pd.Series): Logit transformed student proportion
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate optimal grid dimensions
    n_features = len(scores_df.columns)
    n_plots = n_features * 2  # Two plots per feature
    
    # Calculate grid dimensions aiming for square layout
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Create subplot grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    # Plot correlations
    for i, column in enumerate(scores_df.columns):
        # Calculate correlations
        corr_Y = scores_df[column].corr(Y)
        corr_Y_logit = scores_df[column].corr(Y_logit)
        
        # Raw proportion plot
        sns.regplot(x=scores_df[column], y=Y, 
                   ax=axes[i*2], 
                   scatter_kws={'s': 10}, 
                   line_kws={'color': 'red'})
        axes[i*2].set_title(f"{column} vs Student Proportion\n(r={corr_Y:.2f})")
        axes[i*2].set_xlabel(column)
        axes[i*2].set_ylabel("Student Proportion")
        
        # Logit transformed plot
        sns.regplot(x=scores_df[column], y=Y_logit, 
                   ax=axes[i*2 + 1], 
                   scatter_kws={'s': 10}, 
                   line_kws={'color': 'red'})
        axes[i*2 + 1].set_title(f"{column} vs Logit Student Proportion\n(r={corr_Y_logit:.2f})")
        axes[i*2 + 1].set_xlabel(column)
        axes[i*2 + 1].set_ylabel("Logit Student Proportion")
    
    # Hide unused subplots
    for j in range(i*2 + 2, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_transformation(fn, scores_df):
    scores_transformed = scores_df.apply(fn)

    # plot the first 5 transformed scores
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))

    for ax, column in zip(axes, scores_transformed.columns):
        sns.histplot(scores_transformed[column], bins=100, kde=True, color='blue', label=column, alpha=0.5, ax=ax)
        ax.set_title(f"Distribution of {column} scores")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    return scores_transformed