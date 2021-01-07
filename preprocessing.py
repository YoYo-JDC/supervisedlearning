import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df=pd.read_csv('Gapminder.csv')
df.boxplot('life','Region',rot=60)
plt.show()

#create dummy variable
df_region=pd.get_dummies(df)
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
#axis=1 is columns, azis=0 is rows
df_region = df_region.drop('Region_America',axis=1)

# Print the new columns of df_region
print(df_region.columns)


# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5,normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridgefit=ridge.fit(X,y)
ridge_cv = cross_val_score(ridgefit,X,y, cv=5)

# Print the cross-validated scores
print(ridge_cv)
