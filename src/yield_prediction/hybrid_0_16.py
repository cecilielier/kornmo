# Databricks notebook source
import pandas as pd
from kornmo.kornmo import KornmoDataset
from kornmo.geodata import get_farmer_elevation
import kornmo.kornmo_utils as ku
from kornmo.frostdataset import FrostDataset
import numpy as np

import tensorflow as tf

data_path =  'data'

%load_ext autoreload
%autoreload 2
TIMESTEPS = 17

# COMMAND ----------

def filter_by_years(years, data):
    return data[data['year'].isin(years)]


def get_interpolated_data(years, weather_feature):
    data = pd.DataFrame()

    print(f"Loading {weather_feature} data...")
    for year in years:
        tmp_df = pd.read_csv(f'{data_path}/frost/nn_interpolated/{weather_feature}/{weather_feature}_interpolated_{year}-03-01_to_{year}-10-01.csv')
        tmp_df.insert(0, 'year', year)
        data = pd.concat([data, tmp_df])

    # Drop columns containing 'Unnamed'
    data.drop(columns=[col for col in data.columns if 'Unnamed' in col], inplace=True)

    return_data = ku.normalize(data.filter(regex='day_.*'))
    return_data = return_data.rename(columns=lambda x: f"{weather_feature + x[4:]}")
    columns_to_add = ['orgnr', 'year', 'longitude', 'latitude', 'elevation']
    for i, col in enumerate(columns_to_add):
        return_data.insert(i, col, data[col])

    print(f"Number of loaded entries: {return_data.shape[0]}")
    return return_data

def get_proximity_data(years, weather_feature):
    data = pd.DataFrame()
    print(f"Loading {weather_feature} data...")
    for year in years:
        tmp_df = pd.read_csv(f'{data_path}/frost/by_proximity/{weather_feature}/{weather_feature}_by_proximity_{year}-03-01_to_{year}-10-01.csv')
        tmp_df.drop(columns=['ws_id'], inplace=True)
        tmp_df.insert(0, 'year', year)
        data = pd.concat([data, tmp_df])

    return_data = ku.normalize(data.filter(regex='day_.*'))
    return_data = return_data.rename(columns=lambda x: f"{weather_feature + x[4:]}")

    columns_to_add = ['orgnr', 'year']
    for i, col in enumerate(columns_to_add):
        return_data.insert(i, col, data[col])


    print(f"Number of loaded entries: {return_data.shape[0]}")
    return return_data

def get_soilquality_data():
    data = pd.read_csv(f'{data_path}/farm-information/farmers-with-coordinates-and-soil_quality.csv')
    data.drop(columns=['Unnamed: 0', 'latitude', 'longitude', 'elevation'], inplace=True)
    return_data = ku.normalize(data.drop(columns=['orgnr']))
    return_data.insert(0, 'orgnr', data['orgnr'])
    return return_data


def get_area_and_croptype():
    data = pd.read_csv(f'{data_path}/crop_classification_data/classification_week_1_to_16_results/field_areas.csv')
    data.drop(columns=['Unnamed: 0'], inplace=True)
    data.drop(data[data['area'] < 1500].index, inplace = True)
    data["area"] = ku.normalize(data["area"])
    return data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load deliveries, weather, and historical data

# COMMAND ----------

years = [2017, 2018, 2019]

frost = FrostDataset()
kornmo = KornmoDataset()
deliveries = kornmo.get_deliveries().pipe(ku.split_farmers_on_type)

elevation_data = get_farmer_elevation()
deliveries = deliveries.merge(elevation_data, on=['orgnr'], how='left').fillna(0)

deliveries["yield"] = ku.normalize(deliveries["levert"]/deliveries["areal"], 0, 1000)
deliveries["areal"] = ku.normalize(deliveries["areal"])
deliveries['fulldyrket'] = ku.normalize(deliveries['fulldyrket'])
deliveries['overflatedyrket'] = ku.normalize(deliveries['overflatedyrket'])
deliveries['tilskudd_dyr'] = ku.normalize(deliveries['tilskudd_dyr'])
deliveries['lat'] = ku.normalize(deliveries['lat'])
deliveries['elevation'] = ku.normalize(deliveries['elevation'])

deliveries["key"] = deliveries.orgnr.astype(str) + "/" + deliveries.year.astype(str)
deliveries = deliveries.set_index("key")
deliveries = filter_by_years(years, deliveries)

deliveries
# test = deliveries.loc["869093262/2017"]
# test.loc[test['bygg'] == 2.0]

# COMMAND ----------

historical = ku.get_historical_production(kornmo, [2017, 2018, 2019], 4)
historical = deliveries.merge(historical, how='left').fillna(0)
historical["key"] = historical.orgnr.astype(str) + "/" + historical.year.astype(str)
historical = historical.drop(columns=deliveries.columns)
historical = historical.drop_duplicates(subset='key')
historical = historical.set_index("key")
historical

# COMMAND ----------

sunlight_data = get_interpolated_data(years, 'sunlight')
daydegree5_data = get_interpolated_data(years, 'daydegree5').drop(columns=['longitude', 'latitude', 'elevation'])
ground_data = get_proximity_data(years, 'ground')

temp_and_precip_data = frost.get_as_aggregated(1).dropna().astype(float)
weather_data = temp_and_precip_data.merge(sunlight_data, how='left', on=['orgnr', 'year'])
weather_data = weather_data.merge(daydegree5_data, how='left', on=['orgnr', 'year'])
weather_data = weather_data.merge(ground_data, how='left', on=['orgnr', 'year'])

print(f"Merged {temp_and_precip_data.shape[1]} features of temp and precip data, {sunlight_data.shape[1]} features of sunlight data, {daydegree5_data.shape[1]} features of daydegree data, {ground_data.shape[1]} features of ground data to a total of {weather_data.shape[1]} features")

#weather_data = frost.get_as_aggregated(1).dropna().astype(float)

weather_data["key"] = weather_data.orgnr.astype(int).astype(str) + "/" + weather_data.year.astype(int).astype(str)
weather_data = filter_by_years(years, weather_data)
weather_data.drop(columns=["year", "orgnr"], inplace=True)
weather_data = weather_data.drop_duplicates(subset=["key"])
weather_data = weather_data.set_index("key")

weather_data


# COMMAND ----------

soilquality_data = get_soilquality_data()
soilquality_data["key"] = soilquality_data.orgnr.astype(int).astype(str)
soilquality_data.drop(columns=["orgnr"], inplace=True)
soilquality_data = soilquality_data.drop_duplicates(subset=["key"])
soilquality_data = soilquality_data.set_index("key")
soilquality_data.dropna(inplace=True)
soilquality_data

# COMMAND ----------

area_data = get_area_and_croptype()
area_data["key"] = area_data.orgnr.astype(int).astype(str) + "/" + area_data.year.astype(int).astype(str)
print(len(set(area_data['orgnr'])))
area_data.drop(columns=["year", "orgnr"], inplace=True)
#area_data = area_data.drop_duplicates(subset=["key"])
area_data = area_data.set_index("key")
area_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split training and validation data, and add data to the image iterators

# COMMAND ----------

import sys
from kornmo.sentinel.storage import SentinelDataset

sd = SentinelDataset(f"{data_path}/sentinel_images_combined_uncompressed_100x100.h5")
train, val = sd.to_iterator().split(rand_seed='abc')


def add_historical(orgnr, year, data):
    if f"{orgnr}/{year}" in historical.index.values:
        h_data = historical.loc[f"{orgnr}/{year}"]
        return {'historical': h_data.values }
    else:
        return []


def add_soilquality(orgnr, year, data):
    if str(orgnr) in soilquality_data.index.values:
        soil_data = soilquality_data.loc[orgnr]
        return {'soil_data': soil_data.values}
    else:
        return []


def add_weather(orgnr, year, data):
    if f"{orgnr}/{year}" not in weather_data.index:
        return []

    wd = weather_data.loc[f"{orgnr}/{year}"]

    min_temps = wd.filter(regex='min_temp(([0-9])|([1-9][0-9])|(10[0-9])|(11[0-8]))$').values
    mean_temps = wd.filter(regex='mean_temp(([0-9])|([1-9][0-9])|(10[0-9])|(11[0-8]))$').values
    max_temps = wd.filter(regex='max_temp(([0-9])|([1-9][0-9])|(10[0-9])|(11[0-8]))$').values
    total_rain = wd.filter(regex='total_rain(([0-9])|([1-9][0-9])|(10[0-9])|(11[0-8]))$').values

    sunlight = wd.filter(regex='sunlight(([0-9])|([1-9][0-9])|(10[0-9])|(11[0-8]))$').values
    daydegree5 = wd.filter(regex='daydegree5(([0-9])|([1-9][0-9])|(10[0-9])|(11[0-8]))$').values
    ground = wd.filter(regex='ground(([0-9])|([1-9][0-9])|(10[0-9])|(11[0-8]))$').values

    assert len(min_temps) == len(mean_temps) == len(max_temps) == len(total_rain) == len(sunlight) == len(daydegree5) == len(ground) == TIMESTEPS*7
    wd = np.concatenate((min_temps, mean_temps, max_temps, total_rain, sunlight, daydegree5, ground), axis=0)

    return { 'weather': wd }

def add_grain_types(orgnr, year, data):
    samples = deliveries.loc[[f"{orgnr}/{year}"]]
    if f"{orgnr}/{year}" in area_data.index.values:
        farm_area = area_data.loc[[f"{orgnr}/{year}"]]
        all_data = []

        for i, row in farm_area.iterrows():
            sample = {}
            if row['crop_type'] == 'bygg': sample["type"] = (1,0,0)
            elif row['crop_type'] == 'havre': sample["type"] = (0,1,0)
            elif row['crop_type'] == 'hvete': sample["type"] = (0,0,1)

            sample['area'] = row['area']

            if isinstance(samples, pd.DataFrame):

                del_sample = samples.loc[samples[row['crop_type']] == 1.0]
                if len(del_sample.index) > 0:
                    sample["lat"] = del_sample["lat"].values[0]
                    sample["elevation"] = del_sample["elevation"].values[0]
                    sample["yield"] = del_sample["yield"].values[0]
                    all_data.append(sample)

            else:
                del_sample = samples
                sample["lat"] = del_sample["lat"]
                sample["elevation"] = del_sample["elevation"]
                sample["yield"] = del_sample["yield"]
                all_data.append(sample)


        return all_data
    else:
        return []



train = train.with_data(add_historical, True)\
             .with_data(add_weather, True)\
             .with_data(add_grain_types, True)\
             .with_data(add_soilquality, True)

val = val.with_data(add_historical, True)\
         .with_data(add_weather, True)\
         .with_data(add_grain_types, True)\
         .with_data(add_soilquality, True)

# COMMAND ----------

from tqdm import tqdm
from src.grain_classification.mask_dataset_classification import MaskDatasetClassification
from kornmo.mask.utils import add_mask_as_channel, apply_mask_to_image_series

mask_dataset_path = f'E:/MasterThesisData/data/crop_classification_data/classification_week_1_to_16_results/week_1_16_masks.h5'


mask_dataset = MaskDatasetClassification(mask_dataset_path)
mask_iterator = mask_dataset.get_iterator()
mask_dict = {}

classes = ['bygg', 'havre', 'hvete']

for orgnr, year, crop_type, mask in tqdm(mask_iterator, total=mask_iterator.n):
    mask_dict[f'{orgnr}/{year}/{crop_type}'] = mask

def apply_mask(orgnr, year, imgs, data):
    crop_type = classes[data['type'].index(1)]

    mask = mask_dict[f"{orgnr}/{year}/{crop_type}"]
    return apply_mask_to_image_series(mask, imgs)

train = train.filter(lambda orgnr, year, _,data: f"{orgnr}/{year}/{classes[data['type'].index(1)]}" in mask_dict)
val = val.filter(lambda orgnr, year, _,data: f"{orgnr}/{year}/{classes[data['type'].index(1)]}" in mask_dict)

print(f"train samples: {len(train)}")
print(f"val samples: {len(val)}")

# COMMAND ----------

import tensorflow as tf
!pip install tensorflow_addons
import tensorflow_addons as tfa
import numpy as np
from kornmo.sentinel.transform import salt_n_pepper, rotate180, rotate90
import matplotlib.pyplot as plt


timesteps = 17

stride = 10
def top_left(imgs):
    return imgs[...,:-stride, :-stride,:]
def top_right(imgs):
    return imgs[...,:-stride, stride:,:]
def bot_left(imgs):
    return imgs[...,stride:, :-stride,:]
def bot_right(imgs):
    return imgs[...,stride:, stride:,:]
def center(imgs):
    s = stride//2
    return imgs[...,s:-s, s:-s,:]

def rotate_random(imgs):
    angle = tf.constant(np.pi/8)
    return tfa.image.rotate(imgs, angle)

augmented_dataset = train\
    .transform(apply_mask)\
    .augment([center, top_left, top_right, bot_left, bot_right], keep_original=False)\
    .transform(salt_n_pepper())\
    .transform(rotate_random)


def apply_output(orgnr, year, img_source, data):
    features = data["area"], data["type"][0], data["type"][1], data["type"][2], data["soil_data"][0], data["soil_data"][1], data["soil_data"][2], data["soil_data"][3]
    output = data["yield"]
    weather = data["weather"]
    return {"cnn_input": img_source[0:timesteps], "feature_input": features, "weather_input": weather}, output


train_dataset = tf.data.Dataset.from_generator(
    augmented_dataset.apply(apply_output).shuffled(),
    output_types=({"cnn_input": tf.dtypes.float64, "feature_input": tf.dtypes.float64, "weather_input": tf.dtypes.float64}, tf.dtypes.float64),
).apply(tf.data.experimental.assert_cardinality(len(augmented_dataset)))


val_dataset = tf.data.Dataset.from_generator(
    val.transform(apply_mask).transform(center).apply(apply_output),
    output_types=({"cnn_input": tf.dtypes.float64, "feature_input": tf.dtypes.float64, "weather_input": tf.dtypes.float64}, tf.dtypes.float64),
).apply(tf.data.experimental.assert_cardinality(len(val)))

print(f"Training samples: {len(train)}")
print(f"Augmented samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# COMMAND ----------

from keras import layers, models


def CNN(input_dim, output_dim):
    input_layer = layers.Input(shape=input_dim)
    y = layers.Conv2D(16, (3, 3), activation=tf.nn.relu, padding='same')(input_layer)
    y = layers.MaxPool2D((2, 2))(y)
    y = layers.Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same')(y)
    y = layers.MaxPool2D((2, 2))(y)
    y = layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same')(y)
    y = layers.MaxPool2D((2, 2))(y)
    y = layers.Flatten()(y)
    y = layers.Dense(output_dim, activation=tf.nn.relu)(y)

    return models.Model(inputs=[input_layer], outputs=[y], name="SingleImageCNN")

scnn = CNN((90, 90, 12), 64)
# scnn.summary(line_length=130)

input_weather = layers.Input(shape=timesteps*7*7, name="weather_input")
t_wm = layers.Reshape((7, timesteps*7))(input_weather)
t_wm = layers.Permute((2, 1))(t_wm)
t_wm = layers.Conv1D(64, 7, 7, activation=tf.nn.relu)(t_wm)

input_cnn = layers.Input(shape=(timesteps, 90, 90, 12), name="cnn_input")

feature_input = layers.Input(shape=(8,), name="feature_input")
feature_repeated = layers.RepeatVector(timesteps)(feature_input)

cnn = layers.TimeDistributed(scnn)(input_cnn)
cnn = layers.Concatenate(axis=2)([cnn, feature_repeated, t_wm])
cnn = layers.GRU(128)(cnn)
cnn = layers.Flatten()(cnn)
cnn = layers.Dense(128, activation=tf.nn.relu)(cnn)
cnn = layers.Dense(1)(cnn)

cnn = models.Model(inputs=[input_weather, input_cnn, feature_input], outputs=cnn, name="CNN")
# cnn.summary(line_length=130)

cnn.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_absolute_error')



stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    f'E:/MasterThesisData/1_16_yield/model_checkpoint',
    monitor="val_loss",
    verbose=0,
    mode="min",
    save_best_only=True,
    save_freq='epoch'
)

callbacks = [stopping, model_checkpoint]

# Use this if loading a model
# cnn = models.load_model(f'{data_path}/training/epoch_26.hdf5')

cnn_history = cnn.fit(
    train_dataset.take(10000).batch(32).prefetch(tf.data.AUTOTUNE),
    validation_data=val_dataset.batch(10).prefetch(tf.data.AUTOTUNE),
    epochs=100,
    verbose=1,
    callbacks=callbacks
)




# COMMAND ----------


import seaborn as sns
import matplotlib.pyplot as plt


def denormalize(df, lower: float, upper: float):
    return df * (upper - lower) + lower


def plot_history(history, save_file: str=None):
    sns.set_style('whitegrid')
    sns.set_context("paper")

    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend()
    plt.title("Mean absolute error loss")
    if save_file is not None:
        plt.savefig(save_file, dpi=600)
    plt.grid()


plot_history(cnn_history, save_file=data_path)


predictions = cnn.predict(val_dataset.batch(32).prefetch(2))
predictions = np.array(predictions).flatten()
facts = val.apply(lambda orgnr, year, img, data: data["yield"]).as_array()

absolute_error = np.abs(predictions - facts)
print(f"Denormalized MAE: {denormalize(absolute_error.mean(), 0, 1000)}")

# Denormalized MAE: 89.79933446094495


# COMMAND ----------

from keras.models import load_model

best_model = load_model(data_path)

predictions = best_model.predict(val_dataset.batch(32).prefetch(2))
predictions = np.array(predictions).flatten()
facts = val.apply(lambda orgnr, year, img, data: data["yield"]).as_array()

absolute_error = np.abs(predictions - facts)
print(f"Denormalized MAE: {denormalize(absolute_error.mean(), 0, 1000)}")


