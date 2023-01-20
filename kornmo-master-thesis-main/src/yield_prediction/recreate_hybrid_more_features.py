# Databricks notebook source
import pandas as pd
import numpy as np
import sys

from kornmo import KornmoDataset
from geodata import get_farmer_elevation
import kornmo_utils as ku
from frostdataset import FrostDataset

%load_ext autoreload
%autoreload 2
import tensorflow as tf



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

# COMMAND ----------

years = [2017, 2018, 2019, 2020]
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

# COMMAND ----------

historical = ku.get_historical_production(kornmo, deliveries.year.unique(), 4)
historical = deliveries.merge(historical, how='left').fillna(0)
historical["key"] = historical.orgnr.astype(str) + "/" + historical.year.astype(str)
historical = historical.drop(columns=deliveries.columns)
historical = historical.drop_duplicates(subset='key')
historical = historical.set_index("key")
historical = filter_by_years(years, historical)

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
weather_data.drop(columns=["year", "orgnr"], inplace=True)
weather_data = weather_data.drop_duplicates(subset=["key"])
weather_data = weather_data.set_index("key")
weather_data

# COMMAND ----------

#Combine dataset

# sat_img_path = 'C:/'
# from sentinel.storage import SentinelDataset
# print("Reading sentinel_100x100_0.h5")
# ds0 = SentinelDataset(f"{sat_img_path}/sentinel_100x100_0.h5")
# print("Reading sentinel_100x100_1.h5")
# ds1 = SentinelDataset(f"{sat_img_path}/sentinel_100x100_1.h5")
# print("Combining both")
# SentinelDataset.combine_datasets([ds0, ds1], "E:/combined_compressed.h5", compression=4)

# COMMAND ----------

from sentinel.storage import SentinelDataset
sat_img_path = 'E:/MasterThesisData/Satellite_Images'
#sat_img_path = 'C:/'
sd = SentinelDataset(f"{sat_img_path}/combined_uncompressed.h5")
train, val = sd.to_iterator().split(rand_seed='abc')

def add_historical(orgnr, year, data):
    if f"{orgnr}/{year}" in historical.index.values:
        h_data = historical.loc[f"{orgnr}/{year}"]
        return {'historical': h_data.values }
    else:
        return []
def add_weather(orgnr, year, data):
    if f"{orgnr}/{year}" not in weather_data.index:
        return []
    wd = weather_data.loc[f"{orgnr}/{year}"]


    return { 'weather': wd.values }

def add_grain_types(orgnr, year, data):
    samples = deliveries.loc[[f"{orgnr}/{year}"]]

    all_grains = []
    for _, row in samples.iterrows():
        sample = {}
        if row.bygg: sample["type"] = (1,0,0,0)
        elif row.havre: sample["type"] = (0,1,0,0)
        elif row.rug_og_rughvete: sample["type"] = (0,0,1,0)
        elif row.hvete: sample["type"] = (0,0,0,1)

        sample["areal"] = row["areal"]
        sample["lat"] = row["lat"]
        sample["elevation"] = row["elevation"]
        sample["yield"] = row["yield"]
        sample['fulldyrket'] = row['fulldyrket']
        sample['overflatedyrket'] = row['overflatedyrket']
        sample['tilskudd_dyr'] = row['tilskudd_dyr']
        all_grains.append(sample)

    return all_grains

train = train.with_data(add_historical, True)\
             .with_data(add_weather, True)\
             .with_data(add_grain_types, True)

val = val.with_data(add_historical, True)\
         .with_data(add_weather, True)\
         .with_data(add_grain_types, True)



# COMMAND ----------

from mask.mask_dataset import MaskDataset
from mask.utils import add_mask_as_channel, apply_mask_to_image_series

mask_dataset_path = "data/masks/nibio_disposed_properties_masks.h5"
mask_dataset = MaskDataset(mask_dataset_path)
#print(mask_dataset.labels)

mask_iterator = mask_dataset.get_iterator()
mask_dict = {}
for orgnr, year, mask in mask_iterator:
    mask_dict[f'{orgnr}/{year}'] = mask

def apply_mask(orgnr, year, imgs):
    mask = mask_dict[f'{orgnr}/{year}']
    return apply_mask_to_image_series(mask, imgs)

train = train.filter(lambda orgnr, year, _,__: f"{orgnr}/{year}" in mask_dict)
val = val.filter(lambda orgnr, year, _,__: f"{orgnr}/{year}" in mask_dict)

print(f"train samples: {len(train)}")
print(f"val samples: {len(val)}")


# COMMAND ----------

import tensorflow_addons as tfa
import numpy as np
from tensorflow.data.experimental import assert_cardinality
from sentinel.transform import salt_n_pepper, rotate180, rotate90

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
    angle = np.random.rand(30) * 6.28
    return tfa.image.rotate(imgs, angle)

augmented_dataset = train\
    .transform(apply_mask)\
    .transform(salt_n_pepper())\
    .augment([center, top_left, top_right, bot_left, bot_right], keep_original=False)\
    .transform(rotate_random)

def apply_output(orgnr, year, img_source, data):
    features = data["areal"], *data["type"]
    output = data["yield"]
    weather = data["weather"][1:]
    return {"cnn_input": img_source[0:30], "feature_input": features, "weather_input": weather}, output

train_dataset = tf.data.Dataset.from_generator(
    augmented_dataset.apply(apply_output).shuffled(),
    output_types=({"cnn_input": tf.dtypes.float64, "feature_input": tf.dtypes.float64, "weather_input": tf.dtypes.float64}, tf.dtypes.float64),
).apply(assert_cardinality(len(augmented_dataset)))

val_dataset = tf.data.Dataset.from_generator(
    val.transform(apply_mask).transform(center).apply(apply_output),
    output_types=({"cnn_input": tf.dtypes.float64, "feature_input": tf.dtypes.float64, "weather_input": tf.dtypes.float64}, tf.dtypes.float64),
).apply(assert_cardinality(len(val)))

print(f"Augmented samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# COMMAND ----------

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow import keras



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

file_path = "./training/yield_hybrid/hybrid_yield_model.h5"
model_checkpoint = keras.callbacks.ModelCheckpoint(
    file_path,
    monitor="val_loss",
    verbose=0,
    mode="min",
    save_best_only=True,
    save_weights_only=True,
)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

callbacks = [callback, model_checkpoint]

restart = False
if restart:

    scnn = CNN((90, 90, 12), 64)
    #scnn.summary(line_length=130)
    input_weather = layers.Input(shape=1501, name="weather_input") #shape = 856 / 1501
    t_wm = layers.Reshape((19, 79))(input_weather) # (4, 214) / (19, 79)
    t_wm = layers.Permute((2, 1))(t_wm)
    t_wm = layers.Conv1D(64, 50, activation=tf.nn.relu)(t_wm) # (64, 7, 7) / (64, 50)

    input_cnn = layers.Input(shape=(30, 90, 90, 12), name="cnn_input")

    feature_input = layers.Input(shape=(5,), name="feature_input")
    feature_repeated = layers.RepeatVector(30)(feature_input)

    cnn = layers.TimeDistributed(scnn)(input_cnn)
    cnn = layers.Concatenate(axis=2)([cnn, feature_repeated, t_wm])
    cnn = layers.GRU(128, return_sequences=False)(cnn)
    cnn = layers.Flatten()(cnn)
    cnn = layers.Dense(128, activation=tf.nn.relu)(cnn)
    cnn = layers.Dense(1)(cnn)

    cnn = models.Model(inputs=[input_weather, input_cnn, feature_input], outputs=cnn, name="CNN")
    #cnn.summary(line_length=130)

    cnn.compile(optimizer=optimizers.Adam(), loss='mean_absolute_error')

    cnn_history = cnn.fit(
        train_dataset.take(10000).batch(32).prefetch(2),
        validation_data=val_dataset.batch(32).prefetch(2),
        epochs=10,
        verbose=1,
        callbacks=callbacks
    )

else:
    cnn = load_model('./training/yield_hybrid/epoch_10.hdf5')
    # update the learning rate
    cnn_history = cnn.fit(
        train_dataset.take(10000).batch(32).prefetch(2),
        validation_data=val_dataset.batch(32).prefetch(2),
        epochs=10,
        verbose=1,
        callbacks=callbacks
    )

# COMMAND ----------

cnn = load_model('./training/yield_hybrid/epoch_10.hdf5')

# COMMAND ----------

cnn.evaluate(val_dataset.batch(32))

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_context("paper")

plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.plot(history['loss'].tolist(), label="loss")
plt.plot(history['val_loss'].tolist(), label="val_loss")
plt.legend()
plt.title("Mean absolute error loss")
plt.savefig('logs/hybrid_more_features.svg', dpi=600)
plt.grid()
