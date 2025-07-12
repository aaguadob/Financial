import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LinearRegression
import xgboost as xgb

import openbb as obb

output = obb.obb.equity.price.historical("AAPL")
df = output.to_dataframe()
print(df)




