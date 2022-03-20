import os
import pytest
import numpy as np
import pandas as pd

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test')  # Mark the test root directory


# Context to retain objects passed between steps
class Context:
    pass


@pytest.fixture(scope='session')
def context():
    return Context()


@pytest.fixture
def clean_data_set():
    n = 10
    X_num = np.random.randn(n, 9)
    X_cat = np.zeros((n, 5))
    for i in range(n):
        j = np.random.randint(0, 5)  # Select a random category index
        X_cat[i, j] = 1
    X = np.c_[X_num, X_cat]
    y = np.round(207000.0 + 115700.0*np.random.randn(n))

    return X, y


@pytest.fixture
def raw_data_set():
    test_csv_path = os.path.join(TEST_DIR, "housing.csv")
    return pd.read_csv(test_csv_path)


def __init__():
    pass