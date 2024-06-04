# The more features and functionality you want to test, your
# test suite is going to be longer. The goes for testing more
# situations or corner cases.

from joblib import load, dump
import numpy as np
from linear_model import train_linear_model
from os import path
import sklearn
from sklearn.model_selection import train_test_split
import pytest
import math

def random_data_constructor(noise_mag=1.0):
    """
    Random data constructor utility for tests
    """
    num_points = 100
    X = 10*np.random.random(size=num_points)
    y = 2*X+3+2*noise_mag*np.random.normal(size=num_points)
    return X,y

#----------------------------------------------------------

def fixed_data_constructor():
    """
    Fixed data constructor utility for tests
    """
    num_points = 100
    X = np.linspace(1,10,num_points)
    y = 2*X+3
    return X,y

#----------------------------------------------------------

def test_model_return_object():
    """
    Tests the returned object of the modeling function
    """
    X,y = random_data_constructor()
    scores = train_linear_model(X,y)

    assert isinstance(scores, dict)
    assert len(scores) == 2
    assert 'Train-score' in scores and 'Test-score' in scores

#----------------------------------------------------------

def test_model_return_value():
    """
    Tests for the returned values of the modeling function
    """

    X,y = random_data_constructor()
    scores = train_linear_model(X,y)

    assert isinstance(scores['Train-score'], float)
    assert isinstance(scores['Test-score'], float)

#----------------------------------------------------------

def test_loaded_model_works():
    """
    Tests if the loading of the model works correctly
    """
    X,y = fixed_data_constructor()
    if len(X.shape) == 1:
        X = X.reshape(-1,1)
    if len(y.shape) == 1:
        y = y.reshape(-1,1)
    filename = 'testing'
    scores = train_linear_model(X,y, filename=filename)
    loaded_model = load('testing.sav')

    assert scores['Train-score'] == 1.0
    assert scores['Test-score'] == 1.0

    np.testing.assert_allclose(y, loaded_model.predict(X))