import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing_bunch = fetch_california_housing(as_frame= True)
type(housing_bunch)

#Below is the full data set contaiing features and target:
df = pd.DataFrame(housing_bunch.frame)
df.describe().T

# Histogram of every feature (data distribution)
df.hist(bins=50, figsize= (20,20))
# Shape of data (rows and columns)
df.shape
# counts, missing values, data types of all features/columns
df.info()

#check for missing values, no missing values
df.isna().T
df.isna().sum()

#correlation matrix
df.corr(method= 'kendall', numeric_only=True)
df.corr(numeric_only=True)

df.plot(
    kind= 'scatter',
    x='Longitude',
    y='Latitude',
    alpha =0.3,
    grid =True,
    s =df['Population']/100, # Size of the dot is decided by the population
    label = 'Population', 
    c = 'MedHouseVal', 
    colormap='jet', 
    figsize=(12,8), 
    legend= True
    )

df.plot(
    kind= 'scatter',
    x='Longitude',
    y='Latitude',
    alpha =0.4,
    grid =True,
    s =df['MedHouseVal']*10, # Size of the dot is decided by the population
    label = 'Houseval', 
    c = 'MedHouseVal', 
    colormap='jet', 
    figsize=(12,8), 
    legend= True
    )

# Scatter plot using matplotlib
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["Longitude"],
    df["Latitude"],
    c=df["MedHouseVal"],      # colour coding
    cmap="viridis",           # nice green→blue gradient
    alpha=0.6                 # transparency
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("House Prices by Location")
plt.colorbar(scatter, label="Median House Value")
plt.show()

import numpy as np

median_price = df["MedHouseVal"].median()
df["HighValue"] = (df["MedHouseVal"] > median_price).astype(int)

df["HighValue"].value_counts().get(1, 0)

