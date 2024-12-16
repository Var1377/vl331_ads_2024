import os
import pandas as pd
import geopandas as gpd
import numpy as np
import re
import multiprocessing

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

headers = "osmid|area|amenity|building|building:use|building:levels|height|shop|leisure|sport|landuse|office|railway|public_transport|highway|aeroway|waterway|man_made|geometry".split("|")

# for every geojson file in the osm_features directory, save it to a csv

def process_file(file):
    if not file.endswith(".geojson"):
        return

    name = file.split(".")[0]


    if os.path.exists(f"osm_features/{name}.csv"):
        return
        
    print(f"Processing {name}")

    pois = gpd.read_file(f"osm_features/{file}")

    pois["height"] = parse_height_to_meters(pois["height"])
    pois["building:levels"] = pois["building:levels"].apply(lambda x: int(x) if pd.notnull(x) and x.isdigit() else None)
    pois["area"] = pois["geometry"].set_crs(epsg=4326).to_crs(epsg=27700).area

    pois = pois.dropna(subset=["geometry"])

    to_save = pd.DataFrame(columns=headers)

    for col in headers:
        if col in pois.columns:
            to_save[col] = pois[col]

    to_save.to_csv(f"osm_features/{name}.csv", sep="|", index=False)

if __name__ == '__main__':
    files = os.listdir("osm_features")
    with multiprocessing.Pool() as pool:
        pool.map(process_file, files)