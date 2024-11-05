from .config import *
import requests

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

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


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    download_price_paid_data(2011, 2021)