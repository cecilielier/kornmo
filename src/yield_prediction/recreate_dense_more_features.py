# Databricks notebook source
import pandas as pd
import sys

from kornmo import KornmoDataset
from frostdataset import FrostDataset
from visualize import plot
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import kornmo_utils as ku
from visualize import plot_history
from keras.models import load_model

%load_ext autoreload
%autoreload 2

# COMMAND ----------

def filter_by_years(years, data):
    return data[data['year'].isin(years)]

def get_interpolated_data(years, weather_feature):
    data = pd.DataFrame()

    print(f"Loading {weather_feature} data...")
    for year in years:
        tmp_df = pd.read_csv(f'../../kornmo-data-files/raw-data/weather-data/nn_interpolated/{weather_feature}/{weather_feature}_interpolated_{year}-03-01_to_{year}-10-01.csv')
        tmp_df.insert(0, 'year', year)
        data = pd.concat([data, tmp_df])

    # Drop columns containing 'Unnamed'
    data.drop(columns=[col for col in data.columns if 'Unnamed' in col], inplace=True)

    return_data = ku.normalize(data.filter(regex='day_.*'))
    columns_to_add = ['orgnr', 'year', 'longitude', 'latitude', 'elevation']
    for i, col in enumerate(columns_to_add):
        return_data.insert(i, col, data[col])

    print(f"Number of loaded entries: {return_data.shape[0]}")
    return return_data

def get_proximity_data(years, weather_feature):
    data = pd.DataFrame()
    print(f"Loading {weather_feature} data...")
    for year in years:
        tmp_df = pd.read_csv(f'../../kornmo-data-files/raw-data/weather-data/by_proximity/{weather_feature}/{weather_feature}_by_proximity_{year}-03-01_to_{year}-10-01.csv')
        tmp_df.drop(columns=['ws_id'], inplace=True)
        tmp_df.insert(0, 'year', year)
        data = pd.concat([data, tmp_df])

    return_data = ku.normalize(data.filter(regex='day_.*'))
    columns_to_add = ['orgnr', 'year']
    for i, col in enumerate(columns_to_add):
        return_data.insert(i, col, data[col])


    print(f"Number of loaded entries: {return_data.shape[0]}")
    return return_data

def get_soilquality_data():
    data = pd.read_csv(f'../../kornmo-data-files/raw-data/farm-information/farmers-with-coordinates-and-soil_quality.csv')
    data.drop(columns=['Unnamed: 0', 'latitude', 'longitude', 'elevation'], inplace=True)
    return_data = ku.normalize(data.drop(columns=['orgnr']))
    return_data.insert(0, 'orgnr', data['orgnr'])
    return return_data

# COMMAND ----------

frost = FrostDataset()
kornmo = KornmoDataset()

years = [2017, 2018, 2019]

# Grants and deliveries
data = kornmo.get_deliveries().pipe(ku.split_farmers_on_type)
data = filter_by_years(years, data)
data


# COMMAND ----------

# Temperature and Precipitation
temp_and_precip_data = frost.get_as_aggregated(1)

# COMMAND ----------

sunlight_data = get_interpolated_data(years, 'sunlight')
daydegree5_data = get_interpolated_data(years, 'daydegree5').drop(columns=['longitude', 'latitude', 'elevation'])
ground_data = get_proximity_data(years, 'ground')
weather_data = temp_and_precip_data.merge(sunlight_data, how='left', on=['orgnr', 'year'])
weather_data = weather_data.merge(daydegree5_data, how='left', on=['orgnr', 'year'])
weather_data = weather_data.merge(ground_data, how='left', on=['orgnr', 'year'])

print(f"Merged {temp_and_precip_data.shape[1]} features of temp and precip data, {sunlight_data.shape[1]} features of sunlight data, {daydegree5_data.shape[1]} features of daydegree data, {ground_data.shape[1]} features of ground data to a total of {weather_data.shape[1]} features")

data = data.merge(weather_data, how='left', on=['year', 'orgnr'])
data = filter_by_years(years, data)
data


# COMMAND ----------

soilquality_data = get_soilquality_data()
data = data.merge(soilquality_data, on=['orgnr'])
data


# COMMAND ----------

# Legacy grants
historical_data = ku.get_historical_production(kornmo, data.year.unique(), 4)
data = data.merge(historical_data, on=['orgnr', 'year'])
data = filter_by_years(years, data)
data


# COMMAND ----------

data.dropna(inplace=True)

data['y'] = data['levert'] / data['areal']
data.drop('levert', axis=1, inplace=True)

data['y'] = ku.normalize(data['y'], 0, 1000)
data['areal'] = ku.normalize(data['areal'])
data['fulldyrket'] = ku.normalize(data['fulldyrket'])
data['overflatedyrket'] = ku.normalize(data['overflatedyrket'])
data['tilskudd_dyr'] = ku.normalize(data['tilskudd_dyr'])
data['growth_start_day'] = ku.normalize(data['growth_start_day'])
data['elevation'] = ku.normalize(data['elevation'])
data['latitude'] = ku.normalize(data['latitude'])
data['longitude'] = ku.normalize(data['longitude'])

y_column = ['y']
remove_from_training = ['orgnr', 'kommunenr', 'gaardsnummer', 'bruksnummer', 'festenummer', 'year'] + y_column

data


# COMMAND ----------

train, val = train_test_split(shuffle(data), test_size=0.2)
val, test = train_test_split(val, test_size=0.2)

train_x = train.drop(remove_from_training, axis=1).to_numpy()
train_y = train[y_column].to_numpy()

val_x = val.drop(remove_from_training, axis=1).to_numpy()
val_y = val[y_column].to_numpy()

print(f'Training dataset x: {train_x.shape}')
print(f'Training dataset y: {train_y.shape}')
print(f'Validation dataset x: {val_x.shape}')
print(f'Validation dataset y : {val_y.shape}')


# COMMAND ----------

from dense_model import train_simple_dense
logs_name = 'more_features_with_soil_quality'

model, history = train_simple_dense(train_x, train_y, val_x, val_y)
plot(model, val_x, val_y)

