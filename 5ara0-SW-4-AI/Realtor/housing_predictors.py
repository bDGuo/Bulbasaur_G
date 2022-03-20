import os
import joblib
from housing_models import MODEL_PATH
from housing_data_pre_processor import PIPELINE_PATH


# <ASSIGNMENT 3.8: Complete the function>
def predict_median_house_value(data_set, pipeline_path=PIPELINE_PATH, model_path=MODEL_PATH):
    """
    Using the previously trained pre-processing pipeline and model to predict housing values for sanitized user-input.

    :param data_set: sanitized user input (DataFrame)
    :param pipeline_path: path to trained pre-processing pipeline (string)
    :param model_path: path to trained model (string)
    :return: predicted median housing value (numpy array)
    """

    pipeline_file = os.path.join(pipeline_path, "pipeline.pkl")
    model_file = os.path.join(model_path, "model.pkl")

    pipeline = joblib.load(pipeline_file)
    model = joblib.load(model_file)

    X_pred = pipeline.transform(data_set)  # <APPLY 'pipeline' TO TRANSFORM 'data_set'>
    y_pred = model.predict(X_pred)  # <APPLY 'model' TO PRE-PROCESSED 'X_pred'>

    return y_pred
# </ASSIGNMENT 3.8>