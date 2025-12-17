from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# Data preparation functions
# Get the project ro
# ot directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')

data = pd.read_csv(DATA_PATH)

def impute_ames_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # 1. Drop weak columns
    drop_cols = ['Utilities', 'Condition2', 'RoofMatl']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # 2. Categorical: feature does not exist
    none_cols = [
        'Alley','MasVnrType',
        'BsmtQual','BsmtCond','BsmtExposure',
        'BsmtFinType1','BsmtFinType2',
        'FireplaceQu',
        'GarageType','GarageFinish','GarageQual','GarageCond',
        'PoolQC','Fence','MiscFeature'
    ]

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # 3. Numeric: feature does not exist
    zero_cols = ['MasVnrArea', 'GarageYrBlt']
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 4. LotFrontage: median by Neighborhood
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = (
            df.groupby('Neighborhood')['LotFrontage']
              .transform(lambda x: x.fillna(x.median()))
        )

    # 5. Electrical: mode
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    return df

data = impute_ames_data(data)



# Feature Engineering 
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['Age'] = data['YrSold'] - data['YearBuilt']
data['RemodAge'] = data['YrSold'] - data['YearRemodAdd']
data['HasBasement'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data["HasGarage"] = data["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
data["HasPool"] = data["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
data["HasFireplace"] = data["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)
data["TotalBath"] = (data["FullBath"] + (0.5 * data["HalfBath"]) +
                     data["BsmtFullBath"] + (0.5 * data["BsmtHalfBath"]))
data['TotalPorchSF'] = (data['OpenPorchSF'] + data['3SsnPorch'] +
                        data['EnclosedPorch'] + data['ScreenPorch'] +
                        data['WoodDeckSF'])
print(data[['TotalSF', 'Age', 'RemodAge', 'HasBasement', 'HasGarage',
            'HasPool', 'HasFireplace', 'TotalBath', 'TotalPorchSF']].head())



# remove unwanted columns from i derived features
data = data.drop(columns=['Id', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                          'YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageArea',
                          'PoolArea', 'Fireplaces', 'FullBath', 'HalfBath',
                          'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF',
                          '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF',
                          'Street', 'Alley','Heating', 'PoolQC', 'MiscFeature', "Fence",
                          'PavedDrive', 'GarageCond', 'LandSlope', 'BsmtFinSF2', 'BsmtFinType2', 
                          'BldgType', 'LowQualFinSF', 'MiscVal', 'HasGarage', 
                          'HasBasement', 'HasFireplace', 'HasPool'])


# Encoding categorical variables

# Ordinal Encoding
ordinal_cols = [
    'ExterQual','ExterCond','BsmtQual','BsmtCond',
    'HeatingQC','KitchenQual','FireplaceQu',
    'GarageQual'
]

ordinal_map = {
    'None': 0,
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

for col in ordinal_cols:
    data[col] = data[col].map(ordinal_map)

# Binary Encoding
data['CentralAir'] = data['CentralAir'].map({'N': 0, 'Y': 1})

# Target Encoding for high cardinality categorical variables

from category_encoders import TargetEncoder

nominal_cols = [
    'MSZoning','LotShape','LandContour','LotConfig',
    'Neighborhood','Condition1','HouseStyle','RoofStyle',
    'Exterior1st','Exterior2nd','MasVnrType','Foundation',
    'BsmtExposure','BsmtFinType1',
    'Electrical','Functional','GarageType',
    'GarageFinish','SaleType','SaleCondition'
]

te = TargetEncoder(cols=nominal_cols, smoothing=0.3)
data[nominal_cols] = te.fit_transform(data[nominal_cols], data['SalePrice'])

# Scaling continuous features
continuous_cols = ['LotFrontage', 'LotArea', 'Neighborhood', 'MasVnrArea',
                   'BsmtFinSF1', 'BsmtUnfSF', 'GrLivArea', 'GarageYrBlt',
                   'SalePrice', 'TotalSF', 'Age', 'RemodAge', 'TotalPorchSF']

def scale_continuous_features(df, continuous_cols):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    return df_scaled, scaler


# if not data.to_csv("../data/processed_ames_data.csv", index=False):
#     print("Processed data saved successfully.")