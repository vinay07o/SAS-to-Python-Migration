"""Reusable fixture"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data_path():
    return "data/store1.csv"


@pytest.fixture
def sample_data():
    df = pd.read_csv("data/store1.csv")
    return df
