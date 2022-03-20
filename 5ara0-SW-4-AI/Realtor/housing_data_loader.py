import os
import tarfile
import urllib
import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Housing data loader marks the root directory
DATA_PATH = os.path.join(ROOT_DIR, "datasets")

# Features used for prediction
FEATURES = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
            "households", "median_income", "ocean_proximity"]

# Specify which features are categorical (dictionary keys),
# together with valid categories for the corresponding feature (dictionary values).
CATEGORIES = {"ocean_proximity": ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]}

# Quantity that is predicted
TARGET = "median_house_value"


def query_input_data(features=FEATURES, categories=CATEGORIES):
    """
    Query user for input data.

    :param features: list of valid features;
    :param categories: dictionary of categorical features and valid categories;
    :return: a DataFrame with sanitized inputs.
    """

    # Initialize an empty dictionary of data entries
    input_data = {}
    for feature in features:
        input_data[feature] = []

    # Append sanitized values to dictionary
    for feature in features:
        raw_value = input(f"{feature}: ")  # Request input
        (_, sanitized_value) = sanitize_data_entry(feature, raw_value, features, categories)
        input_data[feature].append(sanitized_value)  # Update dictionary with entry

    return pd.DataFrame(input_data)  # Convert entries to DataFrame


# <ASSIGNMENT 2.2: Complete the function>
def sanitize_data_entry(feature, raw_value, features=FEATURES, categories=CATEGORIES):
    """
    Sanitize a user-provided data entry.

    :param feature: feature name (string)
    :param raw_value: user-provided input value (string)
    :param features: list of valid features;
    :param categories: dictionary of categorical features and valid categories;
    :return: a tuple of the original feature name and sanitized value.
    """

    sanitized_value = raw_value  # <YOUR CODE HERE INSTEAD>

    
    if feature not in categories.keys(): # deal with behavior of numeric feature
        if sanitized_value == "": #empty input for a numerical feature
            sanitized_value = np.nan
        else:
            try: 
                sanitized_value = np.float(sanitized_value)
            except ValueError: # raise error when input is not a number for a numerical feature
                raise 
    else: #deal with behavior of string feature
        if feature not in features:
            raise ValueError # raise error when entry for queried feature is not in the valid features
        elif sanitized_value not in categories[feature]:
            raise ValueError #raise error when input is not present with in the valid categories

    return feature, sanitized_value
# </ASSIGNMENT 2.2>

def fetch_housing_data(data_url=DATA_URL, data_path=DATA_PATH, overwrite=False):
    """
    Download and unpack the California housing dataset.

    :param data_url: dataset source;
    :param data_path: root path for storing the download;
    :param overwrite: flag for overwriting existing dataset.
    """

    if not (os.path.exists(data_path)) or overwrite:
        print("Downloading dataset...")
        os.makedirs(data_path, exist_ok=True)
        tgz_path = os.path.join(data_path, "housing.tgz")
        urllib.request.urlretrieve(data_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=data_path)
        housing_tgz.close()
        print("... done.")
    else:
        print("Using existing dataset.")


def load_housing_data(data_path=DATA_PATH):
    """
    Load downloaded California housing dataset in a DataFrame.

    :param data_path: root folder of housing data file;
    :return: DataFrame with raw dataset.
    """

    csv_path = os.path.join(data_path, "housing.csv")
    return pd.read_csv(csv_path)
