from .config import *
import requests
import pymysql
import csv
from zipfile import ZipFile
import os
import json
import osmium
import geopandas as gpd
from assess import parse_height_to_meters
import shapely as shp


def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    filename_template = "/pp-<year>-part<part>.csv"


    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            #if file exists then skip it
            filename = "." + filename_template.replace("<year>", str(year)).replace("<part>", str(part))
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping download")
                continue

            url = base_url + filename_template.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as file:
                    file.write(response.content)

def download_urls(urls):
    for url in urls:
        if isinstance(url, tuple):
            filename, url = url
        else:
            filename = f"./{url.split('/')[-1]}"

        if not os.path.exists(filename):
            print(f"Downloading {url}")
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Already downloaded {filename}")


    if filename.endswith('.zip') and not os.path.exists(filename.replace('.zip', '')):
        with ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()

conn = None

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    global conn
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def housing_upload_join_data(year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()
  print('Data stored for year: ' + str(year))


def extract_osm_features(input_path: str, output_dir: str, filtered_tags: dict) -> None:
    """
    Extracts OSM features from a PBF file and saves them as GeoJSON feature collections.
    
    :param input_path: Path to the .osm.pbf input file.
    :param output_dir: Directory where GeoJSON chunks will be saved.
    :param filtered_tags: A dictionary defining which tags to filter on, in the format:
                          {
                              "tag_key": ["allowed_value1", "allowed_value2", "*"]
                          }
                          Use "*" as a wildcard to accept any value for that key.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine which tags to include as properties
    tags_to_include = sorted(filtered_tags.keys())

    # 1. Extract node coordinates
    # 2. Extract way geometries from node references
    # 3. Process nodes, ways, and relations for features

    class NodeHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            # Store node coordinates: node_id -> (lon, lat)
            self.node_coords = {}

        def node(self, n):
            if n.location:
                self.node_coords[n.id] = (n.location.lon, n.location.lat)

    # First pass: read all nodes
    node_handler = NodeHandler()
    node_handler.apply_file(input_path, locations=True)
    node_coords = node_handler.node_coords

    class WayHandler(osmium.SimpleHandler):
        def __init__(self, node_coords):
            super().__init__()
            self.node_coords = node_coords
            # way_id -> list of [lon, lat]
            self.way_geometries = {}

        def way(self, w):
            coords = []
            for nref in w.nodes:
                if nref.ref in self.node_coords:
                    coords.append(list(self.node_coords[nref.ref]))
            self.way_geometries[w.id] = coords

    # Second pass: read ways and store their geometries
    way_handler = WayHandler(node_coords)
    way_handler.apply_file(input_path, locations=True)
    way_geometries = way_handler.way_geometries

    # Now define a helper function to determine if a set of tags matches the filters
    def matches_filter(tags):
        for k, vals in filtered_tags.items():
            if k in tags:
                if "*" in vals or tags[k] in vals:
                    return True
        return False

    # Helper to build a Feature object
    def build_feature(geometry, osmid, tags):
        # Keep only tags we're interested in
        props = {k: v for k, v in tags.items() if k in tags_to_include}
        # Add osmid
        props["osmid"] = osmid
        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": props
        }

    # Prepare for final pass: process nodes, ways, and relations into features
    # We'll chunk every 200,000 features into a separate file
    features = []
    file_count = 0
    chunk_size = 200000

    def save_chunk():
        nonlocal file_count, features
        if features:
            output_path = os.path.join(output_dir, f"chunk_{file_count}.geojson")
            collection = {
                "type": "FeatureCollection",
                "features": features
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(collection, f, ensure_ascii=False)
            file_count += 1
            features = []

    class ElementHandler(osmium.SimpleHandler):
        def __init__(self, node_coords, way_geometries):
            super().__init__()
            self.node_coords = node_coords
            self.way_geometries = way_geometries

        def node(self, n):
            if n.location is None:
                return
            tags = {k: v for k, v in n.tags}
            if matches_filter(tags):
                geometry = {
                    "type": "Point",
                    "coordinates": [n.location.lon, n.location.lat]
                }
                feat = build_feature(geometry, n.id, tags)
                features.append(feat)
                if len(features) >= chunk_size:
                    save_chunk()

        def way(self, w):
            tags = {k: v for k, v in w.tags}
            if matches_filter(tags):
                coords = self.way_geometries.get(w.id, [])
                if len(coords) > 1:
                    # Check if it's an area or building to form a polygon
                    if "area" in tags or "building" in tags:
                        polygon_coords = coords[:]
                        if polygon_coords and polygon_coords[0] != polygon_coords[-1]:
                            polygon_coords.append(polygon_coords[0])
                        if len(polygon_coords) >= 4:
                            geometry = {
                                "type": "Polygon",
                                "coordinates": [polygon_coords]
                            }
                        else:
                            return
                    else:
                        # Treat as a line
                        geometry = {
                            "type": "LineString",
                            "coordinates": coords
                        }

                    feat = build_feature(geometry, w.id, tags)
                    features.append(feat)
                    if len(features) >= chunk_size:
                        save_chunk()

        def relation(self, r):
            tags = {k: v for k, v in r.tags}
            if matches_filter(tags):
                # Collect and process members (outer, inner rings, etc.)
                outer_rings = []
                inner_rings = []
                point_members = []
                nested_relations = []

                for member in r.members:
                    if member.type == "w":
                        way_geom = way_geometries.get(member.ref, [])
                        if member.role == "outer":
                            outer_rings.append(way_geom)
                        elif member.role == "inner":
                            inner_rings.append(way_geom)
                    elif member.type == "n":
                        if member.ref in self.node_coords:
                            point_members.append(list(self.node_coords[member.ref]))
                    elif member.type == "r":
                        nested_relations.append(member.ref)

                # Close and validate rings
                def process_ring(ring):
                    if ring and ring[0] != ring[-1]:
                        ring.append(ring[0])
                    return ring if len(ring) >= 4 else None

                processed_outer = [r for r in (process_ring(ring[:]) for ring in outer_rings) if r]
                processed_inner = [r for r in (process_ring(ring[:]) for ring in inner_rings) if r]

                if processed_outer:
                    if len(processed_outer) == 1:
                        # Single outer ring polygon
                        rings = [processed_outer[0]] + processed_inner
                        geometry = {
                            "type": "Polygon",
                            "coordinates": rings
                        }
                    else:
                        # Multiple outer rings -> multipolygon
                        # (inner rings are not paired with specific outers in this simple logic)
                        polygons = [[[outer]] for outer in processed_outer]
                        geometry = {
                            "type": "MultiPolygon",
                            "coordinates": polygons
                        }

                    feat = build_feature(geometry, r.id, tags)
                    features.append(feat)
                    if len(features) >= chunk_size:
                        save_chunk()

    handler = ElementHandler(node_coords, way_geometries)
    handler.apply_file(input_path, locations=True)

    # Save remaining features
    if features:
        save_chunk()

def process_osm_geojson_files(input_dir='osm_features', output_dir='osm_features', headers=[
        "osmid", "area", "amenity", "building", "building:use", "building:levels", "height",
        "shop", "leisure", "sport", "landuse", "office", "railway", "public_transport",
        "highway", "aeroway", "waterway", "man_made", "geometry"
    ]):
    """
    Processes GeoJSON files in the specified input directory and saves them as CSV files after applying transformations.
    """
    import os
    import pandas as pd
    

    files = os.listdir(input_dir)

    for file in files:
        if not file.endswith(".geojson"):
            continue

        name = os.path.splitext(file)[0]
        output_file = os.path.join(output_dir, f"{name}.csv")

        if os.path.exists(output_file):
            continue

        print(f"Processing {name}")
        pois = gpd.read_file(os.path.join(input_dir, file))

        pois["height"] = parse_height_to_meters(pois["height"])
        pois["building:levels"] = pois["building:levels"].apply(
            lambda x: int(x) if pd.notnull(x) and str(x).isdigit() else None
        )
        pois["area"] = pois["geometry"].to_crs(epsg=27700).area
        pois = pois.dropna(subset=["geometry"])

        to_save = pois[[col for col in headers if col in pois.columns]]
        to_save.to_csv(output_file, sep="|", index=False)

def load_osm_features_to_db(input_dir='osm_features'):
    """
    Loads OSM features from CSV files into the database using the provided pymysql connection.

    :param connection: A pymysql connection object to the database.
    :param input_dir: Directory where the CSV files are located.
    """
    import os

    cursor = conn.cursor()

    for file in os.listdir(input_dir):
        if not file.endswith(".csv"):
            continue

        name = os.path.splitext(file)[0]
        print(f"Processing {name}")

        filepath = os.path.join(input_dir, file).replace('\\', '/')

        command = f"""
        LOAD DATA LOCAL INFILE '{filepath}'
        INTO TABLE osm_features
        FIELDS TERMINATED BY '|'
        ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (osmid, @area, @amenity, @building, @building_use, building_levels, @height, @shop, @leisure, @sport, @landuse,
        @office, @railway, @public_transport, @highway, @aeroway, @waterway, @man_made, @geo)
        SET geometry = ST_GeomFromText(@geo, 4326),
            area = NULLIF(@area, '0.0'),
            amenity = NULLIF(@amenity, ''),
            building = NULLIF(@building, ''),
            building_use = NULLIF(@building_use, ''),
            shop = NULLIF(@shop, ''),
            leisure = NULLIF(@leisure, ''),
            sport = NULLIF(@sport, ''),
            landuse = NULLIF(@landuse, ''),
            office = NULLIF(@office, ''),
            railway = NULLIF(@railway, ''),
            public_transport = NULLIF(@public_transport, ''),
            highway = NULLIF(@highway, ''),
            aeroway = NULLIF(@aeroway, ''),
            waterway = NULLIF(@waterway, ''),
            man_made = NULLIF(@man_made, ''),
            height = NULLIF(@height, 'NaN');
        """

        try:
            cursor.execute(command)
            conn.commit()
            print(f"Loaded data from {file} into osm_features table.")
        except Exception as e:
            print(f"Failed to load data from {file}: {e}")
            conn.rollback()

    cursor.close()

def load_geometry_from_wkt(df, geo_col="geometry", wkt_col="wkt", crs="EPSG:4326"):
    df[geo_col] = df[wkt_col].apply(shp.wkt.loads)
    df.drop(columns=[wkt_col], inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=geo_col, crs=crs)
    return gdf