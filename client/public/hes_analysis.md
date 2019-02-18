# Welcome to Wrattler

## HES wrangling + analysis challenge
### 1. Load data
We read a subset of the data from agd-1a. This corresponds to "houses that were monitored for approximately one month at 2 minute intervals (most households)". There are two further chunks of this category of data (agd-1b and agd-1b)
```python

# Imports
import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.linear_model import ElasticNetCV


# load the data
df_appliance_type_codes = pd.read_csv("csv/small/appliance_type_codes.csv")
df_appliance_types = pd.read_csv("csv/small/appliance_types.csv")
df_appliance_codes = pd.read_csv("csv/small/appliance_codes.csv")
```
```python

# clean column headers and remove "other" and "unknown" categories
df_appliance_types.columns = df_appliance_types.columns.str.strip()
df_appliance_type_codes.columns = df_appliance_type_codes.columns.str.strip()
df_appliance_types = df_appliance_types.merge(
    df_appliance_type_codes,
    left_on="GroupCode",
    right_on="Code"
)
df_appliance_types = df_appliance_types.loc[
    ~df_appliance_types["Name"].isin(["Other", "Unknown"]),
    ["ApplianceCode", "GroupCode"]
]
```

```python
df_profiles =  pd.read_csv("csv/agd-1a/appliance_group_data-1a_0.001.csv",header=None,names=['IntervalID','Household','ApplianceCode','DateRecorded','Data','TimeRecorded'])

#get the appliance group codes by joining the tables
df_profiles = df_profiles.merge(df_appliance_types, how='left', on='ApplianceCode')
df_profiles.columns = df_profiles.columns.str.strip()
df_profiles["Household2"] = df_profiles["Household"]



### 2. Drop rows with temperature readings
# Some of the rows correspond to temperature readings, which are not appliances, and therefore must be dropped.


temp_rows = df_profiles[df_profiles['GroupCode'].isnull()].index
df_profiles.drop(temp_rows, inplace=True)

```


### 3. Demographic data
Try a simple model where we use some of the demographic information and the monitoring information to infer the fraction of energy usage that comes from different appliance groups.

```python

# ADD your data path here!
df_demo = pd.read_csv("csv/anonhes/ipsos-anonymised-corrected_310713.csv")
```

```python
df_feat = df_demo.copy()[
    ["Household",
    "HouseholdOccupancy",
    "SinglePensioner",
    "SingleNonPensioner",
    "MultiplePensioner",
    "HouseholdWithChildren",
    "MultiplePersonWithNoDependentChildren",
    "HouseholdType",
    "House.age",
    "Social.Grade"]
]
```


#### Wrangling
There's redundancy of information in the table, so we will engineer some new features.

1. In the HouseholdOccupancy column, the occupancy is recorded for up to 6 people, and for more than this it is recorded as "6+". As a simple fix for this, I'll replace "6+" with 6, and cast the column to int.
2. PensionerOnly feature: 0 if anyone who lives in the house is not a pensioner, 1 otherwise. If we create this feature, we can remove several others and reduce redundancy.
3. HouseholdType is a redundant feature, it is simply a combination of the other features. So is MultiplePersonWithNoDependentChildren, and the pensioner features (after creating the PensionerOnly feature).
4. Social grade is categorical but ordered, and has a single missing value. I'll fill this in with the most frequent value and then convert the social grade to a number.
5. The House.age column is a string, e.g. "1950-1966". I will convert this to a number by taking the midpoint of the range. There are some missing values set to -1, which I will replace with the most common value.

```python
# Fix household occupancy column
df_feat.loc[:, "HouseholdOccupancy"] = df_feat[
    "HouseholdOccupancy"
].str.replace("+", "").astype(int)

# Make pensioner only feature
df_feat.loc[:, "PensionerOnly"] = (
    df_feat["SinglePensioner"] | df_feat["MultiplePensioner"]
).astype(int)

# Drop redundant features
df_feat = df_feat.drop(labels=["HouseholdType",
                               "MultiplePersonWithNoDependentChildren",
                               "SinglePensioner",
                               "MultiplePensioner",
                               "SingleNonPensioner"]
                       , axis=1)

# Social grade feature
social_cats = np.sort(df_feat["Social.Grade"].dropna().unique())
df_feat = df_feat.replace(
    to_replace={
        "Social.Grade": {v: int(i) for i, v in enumerate(social_cats)}
    }
)
df_feat.loc[:, "Social.Grade"] = df_feat.loc[:, "Social.Grade"].fillna(
    df_feat["Social.Grade"].value_counts().index[0]
)

# Age of house
def get_age(yrs):
    if yrs == '-1':
        return get_age("1950-1966")
    else:
        try:
            start, end = [int(i) for i in yrs.split('-')]
            yr = start + (end - start) // 2
        except ValueError:
            yr = 2007
    return 2010 - yr

df_feat.loc[:, "House.age"] = df_feat["House.age"].apply(get_age)
df_feat["Household2"] = df_feat["Household"]
test = pd.DataFrame(list(df_feat.dtypes))
```

Now join this table to the electricity monitoring data and do some further preprocessing:


```python

# make table of usage in each appliance group per house on each day
df_y = df_profiles.groupby(
    ['Household', 'DateRecorded', 'GroupCode']
)["Data"].sum().unstack("GroupCode").stack(dropna=False)


# make features table
df_X = df_y.groupby(
    ["Household", "DateRecorded"]
).sum(skipna=True).reset_index().merge(
    df_feat,
    how="inner",
    left_on="Household",
    right_on="Household"
).rename(columns={0: "TotalUsage"})

df_X["dt"] = pd.to_datetime(df_X['DateRecorded'])
df_X["dow"] = df_X["dt"].dt.weekday_name
df_X["month"] = df_X["dt"].dt.month



df_X = df_X.drop(labels=["dt"], axis=1).set_index(
    ["Household", "DateRecorded"]
)
df_X = pd.concat(
    (df_X, pd.get_dummies(df_X["dow"])), axis=1
).drop(labels=["dow"],axis=1)

# make table of usage per appliance group
df_y = df_y.reset_index().merge(
    df_X.reset_index(),
    how="inner",
    left_on=["Household", "DateRecorded"],
    right_on=["Household", "DateRecorded"]
)[["Household", "DateRecorded", "GroupCode", 0, "TotalUsage"]]
df_y.loc[:, 0] = np.log1p(df_y[0]) # log transform usage
df_y = df_y.drop(labels=["TotalUsage"], axis=1)
df_y = df_y.set_index(
    ["Household", "DateRecorded"]
).pivot(columns="GroupCode")[0]

# fourier features
def make_fourier_features(t, order=2):
    # ripped off from fbprophet
    return np.column_stack([
        trig((2.0 * (i + 1) * np.pi * t / 12))
        for i in range(order)
        for trig in (np.sin, np.sin)
    ])

fourier_feats = np.array(
    [make_fourier_features(t) for t in df_X["month"].values]
)[:, 0, :]
for i in range(fourier_feats.shape[1]):
    df_X["fourier"+str(i)] = fourier_feats[:, i]
df_X = df_X.drop(labels=["month"], axis=1)

from sklearn.preprocessing import MinMaxScaler

# standardise columns where appropriate
for column in ["TotalUsage", "House.age", "HouseholdOccupancy", "Social.Grade"]:
    scaler = MinMaxScaler()
    val_scaled = scaler.fit_transform(df_X[column].values.reshape(-1, 1))
    df_X.loc[:, column] = val_scaled
    
for column in df_y.columns:
    scaler = MinMaxScaler()
    val_scaled = scaler.fit_transform(df_y[column].fillna(0.).values.reshape(-1, 1))
    df_y.loc[:, column] = val_scaled    
    

```

```python

# prepare data for modelling
df_all = df_X.merge(df_y, left_index=True, right_index=True)
X = df_all[df_all.columns[:-13]].values
y = df_all[df_all.columns[-1:-13:-1]].values

```

### Model energy usage of houses
We attempt to predict the quantity of energy that a given household


```python

from sklearn.model_selection import train_test_split

# split the houses into train/test sets
houses = df_all["Household2"].unique()
houses_train, houses_test = train_test_split(houses, random_state=2)

# use the above train/test houses to create train/test dataframes
df_train = df_all.loc[df_all['Household2'].isin(houses_train)]
df_test = df_all.loc[df_all['Household2'].isin(houses_test)]


def get_X_y(df):
    # create X and y arrays for scikit learn from dataframes
    X = df[df.columns[:-9]].values
    y = df[df.columns[-1:-10:-1]].values
    return X, y

X_train, y_train = get_X_y(df_train)
X_test, y_test = get_X_y(df_test)


```



```python


# drop columns with all zeros
ind = np.copy(~np.all(y_test == 0, axis=0))
y_train_final = y_train.loc[:, ind]
y_test_final = y_test.loc[:, ind]

apply_rows = [1013.0, 1008.0, 1007.0, 1006.0, 1004.0, 1002.0, 1001.0]

appliances = df_appliance_type_codes.set_index("Code").loc[apply_rows,"Name"].values

```

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV


def train_models(X_train, y_train):
    """A simple model that first classifies the measurement as being 0 or non zero, then 
    predicts the non-zero values."""
    
    X_train = X_train.values
    y_train = y_train.values
    # train classifier to detect zeros
    y_bin = np.copy(y_train)
    y_bin[y_bin != 0.0] = 1.0
    clf = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=42
    )
    clf.fit(X_train, y_bin)
    
    # train regression to predict non-zero values
    # have to do this separately for each of the outputs
    y_reg = np.copy(y_train)
    X_reg = np.copy(X_train)
    regressors = dict()
    for i in range(y_reg.shape[1]):
        regressors[i] = ElasticNetCV(
            random_state=42
        )
        y_i = y_reg[:, i]
        ind = y_i != 0
        X_i = X_reg[ind, :]
        y_i = y_i[ind]
        regressors[i].fit(X_i, y_i)
    
    return (clf, regressors)


def predict(X_test, clf, regressors):
    """Given a classifier and set of regressors, produce predictions."""
    X_test = X_test.values
    y_bin = clf.predict(X_test)
    
    y_pred = np.copy(y_bin)
    for i in range(y_bin.shape[1]):
        y_i = y_pred[:, i]
        ind = y_i != 0
        if ind.sum() != 0:
            y_pred[ind, i] = regressors[i].predict(X_test[ind, :])
    
    return y_bin, y_pred
    
# fit the models and predict on the test set
clf, regressors = train_models(X_train, y_train_final)
y_bin, y_pred = predict(X_test, clf, regressors)
y_baseline = np.repeat(y_train_final.mean(axis=0)[:, None], y_test_final.shape[0], axis=1).T

mean_absolute_error_model =  mean_absolute_error(y_train_final, y_pred)
mean_absolute_error_baseleine = mean_absolute_error(y_train_final, y_baseline)

```