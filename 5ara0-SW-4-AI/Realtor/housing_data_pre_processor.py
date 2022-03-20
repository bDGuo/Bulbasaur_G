import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from housing_attributes_extender import HousingAttributesExtender
from housing_data_loader import FEATURES, CATEGORIES, TARGET, ROOT_DIR

PIPELINE_PATH = os.path.join(ROOT_DIR, "pipelines")


# <ASSIGNMENT 3.1: Complete the function>
def split_housing_train_test(data):
    """
    Split input DataFrame in separate train- and test sets.

    :param data: input dataset (DataFrame)
    :return: list of DataFrames
    """
    
    stratify = pd.cut(data["median_income"], labels=[1, 2, 3, 4, 5],bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf])  # <COMPLETE THE OPTIONS>

    return train_test_split(data, random_state=42,test_size=0.2,train_size=0.8,stratify=stratify)  # <COMPLETE THE OPTIONS>
# </ASSIGNMENT 3.1>

def pre_process_housing_data(train_set, test_set, write_to_file=True):
    """
    Define a pre-precessing pipeline, fit the pipeline on `train_set`, and apply it to `test_set`.

    :param train_set: DataFrame;
    :param test_set: DataFrame;
    :param write_to_file: Store the fitted pre-processing pipeline to a file (Boolean);
    :return: fitted Pipeline and pre-processed datasets (numpy arrays).
    """

    cat_features = list(CATEGORIES.keys())
    num_features = FEATURES.copy()
    for cat_feature in cat_features:
        num_features.remove(cat_feature)

    # <ASSIGNMENT 3.4: Complete the pre-processing pipeline definitions>
    # Define pipeline for numerical features
    num_pipeline = make_pipeline(SimpleImputer(),HousingAttributesExtender(),StandardScaler())  # <COMPLETE ARGUMENTS>

    # Define pipeline for categorical features
    cat_pipeline = make_pipeline(SimpleImputer(strategy="constant",fill_value="missing"),OneHotEncoder())  # <COMPLETE ARGUMENTS>
    # </ASSIGNMENT 3.4>

    pipeline = make_column_transformer((num_pipeline, num_features),
                                       (cat_pipeline, cat_features))
    # Fit and apply pipeline
    X_train = pipeline.fit_transform(train_set[FEATURES])  # Train an apply the pipeline on the train set
    y_train = np.array(train_set[TARGET])  # Extract the quantity of interest
    X_test = pipeline.transform(test_set[FEATURES])  # Apply the trained pipeline to the test set
    y_test = np.array(test_set[TARGET])

    # Store fitted pipeline to file (note, *.pkl files are in .gitignore)
    if write_to_file:
        os.makedirs(PIPELINE_PATH, exist_ok=True)
        pipeline_file = os.path.join(PIPELINE_PATH, "pipeline.pkl")
        joblib.dump(pipeline, pipeline_file)

    return pipeline, X_train, y_train, X_test, y_test
