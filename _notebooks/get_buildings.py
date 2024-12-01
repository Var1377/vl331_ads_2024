
import pandas as pd
import osmnx as ox
import geopandas as gpd
import multiprocessing as mp
import sys

print("Loading output areas")

output_areas_gdf = gpd.read_file("output_areas.geojson")

print("Loaded output areas")

# set the default geometry column
output_areas_gdf.set_geometry("geometry", inplace=True)

output_areas_gdf.geometry.set_crs(epsg=27700, inplace=True)
output_areas_gdf.geometry = output_areas_gdf.geometry.to_crs(epsg=4326)

print("Converted to WGS84")

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

all_buildings = gpd.GeoDataFrame()

for idx, row in output_areas_gdf[start_idx:end_idx].iterrows():
    try:
        buildings = ox.features_from_polygon(row["geometry"], {"building": True})
        buildings["output_area"] = row["OA21CD"]
        all_buildings = pd.concat([all_buildings, buildings])
        print(f"Processed {idx} with {len(buildings)} total buildings")
    except Exception as e:
        print(f"Error processing {idx}: {e}")

all_buildings.to_file(f"output_buildings_{start_idx}_{end_idx}.geojson", driver="GeoJSON")
