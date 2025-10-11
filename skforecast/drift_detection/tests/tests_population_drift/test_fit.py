# Unit test fit
# ==============================================================================
import re
import pytest
import pandas as pd
import numpy as np
import nannyml as nml
import warnings
from ....drift_detection import PopulationDriftDetector
import joblib

#fixtures
data = joblib.load('fixture_data_population_drift.joblib')


