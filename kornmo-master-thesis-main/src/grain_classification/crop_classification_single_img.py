# Databricks notebook source
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import os

from src.utils import to_rgb
from sentinel.storage import SentinelDataset, SentinelDatasetIterator
from mask.mask_dataset import MaskDataset, MaskDatasetIterator
from mask.utils import apply_mask_to_image_series, apply_mask_to_image

%load_ext autoreload
%autoreload 2

data_path = '../../kornmo-data-files/raw-data/crop-classification-data/'

# COMMAND ----------

print("Reading data")
data = gpd.read_file(f"{data_path}/training_data.gpkg")
data['orgnr'] = data['orgnr'].astype(int)
data['year'] = data['year'].astype(int)
print("Reading masks")
masks = MaskDataset(f"{data_path}/train_data_masks.h5")
print("Reading satellite images")
satellite_imgs = SentinelDataset('E:/MasterThesisData/Satellite_Images/satellite_images_train.h5')
#


# COMMAND ----------

data = data.loc[data['planted'] != 'erter']
data = data.loc[data['planted'] != 'rughvete']
data = data.loc[data['planted'] != 'oljefro']
data = data.loc[data['planted'] != 'rug']
print(pd.Series(list(data['planted'])).value_counts())


# COMMAND ----------


labels = list(set(data['planted']))
print(labels)
print(pd.Series(list(data['planted'])).value_counts())
def add_labels(orgnr, year, data_arg):
    orgnr = int(orgnr)
    year = int(year)
    if orgnr in data['orgnr'].unique() and year in data.loc[data['orgnr'] == orgnr]['year'].unique():
        label = data.loc[data['orgnr'] == orgnr].loc[data['year'] == year]['planted'].iloc[0]
        index = labels.index(label)
        arr = [0 for _ in range(0, len(labels))]
        arr[index] = 1
        return {'class': arr}
    else:
        return []


# COMMAND ----------

train, val = satellite_imgs.to_iterator().split(rand_seed='corn')

train = train.with_data(add_labels, show_progress=True)
val = val.with_data(add_labels, show_progress=True)
masks_it = masks.get_iterator()
mask_dict = {}

for orgnr, year, mask in masks_it:
    mask_dict[f'{orgnr}/{year}'] = mask


train = train.filter(lambda orgnr, year, _,__: f"{orgnr}/{year}" in mask_dict)
val = val.filter(lambda orgnr, year, _,__: f"{orgnr}/{year}" in mask_dict)

print(f"train samples: {len(train)}")
print(f"val samples: {len(val)}")

#7737, 1937

# COMMAND ----------

import random

def train_generator():
    for orgnr, year, imgs, label in train:
        for img in imgs:
            img = apply_mask_to_image(mask_dict[f'{orgnr}/{year}'], img)
            yield img, label['class']


def val_generator():
    for orgnr, year, imgs, label in val:
        for img in imgs:
            img = apply_mask_to_image(mask_dict[f'{orgnr}/{year}'], img)
            yield img, label['class']



# COMMAND ----------

for item in train_generator():
    print(item)
    break

# COMMAND ----------


train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_types=(tf.dtypes.float64, tf.dtypes.int64),
    output_shapes=((100, 100, 12), 3)
)

val_dataset = tf.data.Dataset.from_generator(
    val_generator,
    output_types=(tf.dtypes.float64, tf.dtypes.int64),
    output_shapes=((100, 100, 12), 3)
)

# print(f"training samples: {len(train_dataset)}")
# print(f"Validation samples: {len(val_dataset)}")

# COMMAND ----------

from tensorflow.python.data import AUTOTUNE
from keras import models
from keras.applications.densenet import layers
from keras.models import load_model
from tensorflow import optimizers, keras

model_checkpoint = keras.callbacks.ModelCheckpoint(
    './training/single_img_model.h5',
    monitor="val_loss",
    verbose=0,
    mode="min",
    save_best_only=True,
    save_weights_only=True,
)

callbacks = [model_checkpoint]

restart = True
if restart:
    input_layer = layers.Input(shape=(100, 100, 12))
    cnn = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same')(cnn)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same')(cnn)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Flatten()(cnn)
    cnn = layers.Dense(128, activation="relu")(cnn)
    cnn = layers.Dropout(0.2)(cnn)
    cnn = layers.Dense(3, activation='softmax')(cnn)




    cnn = models.Model(inputs=[input_layer], outputs=cnn, name="cnn_pure")
    cnn.compile(
        optimizer=optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy']
    )
    history = cnn.fit(
        train_dataset.batch(32).prefetch(buffer_size=AUTOTUNE),
        validation_data=val_dataset.batch(32).prefetch(buffer_size=AUTOTUNE),
        epochs=100,
        callbacks=callbacks
    )
else:

    cnn_net = load_model('./training/epoch_4.hdf5')

    cnn_history = cnn_net.fit(
        train_dataset.take(10000).batch(32).prefetch(2),
        validation_data=val_dataset.batch(32).prefetch(2),
        epochs=100,
        verbose=1,
        callbacks=callbacks
    )


# COMMAND ----------

from tensorflow.python.data import AUTOTUNE
from keras import models
from keras.applications.densenet import layers
from keras.models import load_model
from tensorflow import optimizers, keras

model_checkpoint = keras.callbacks.ModelCheckpoint(
    './training',
    monitor="val_loss",
    verbose=0,
    mode="min",
    save_best_only=True,
    save_weights_only=True,
)

callbacks = [model_checkpoint]

restart = True
if restart:
    input_layer = layers.Input(shape=(100, 100, 12))
    cnn = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same')(cnn)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same')(cnn)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Flatten()(cnn)
    cnn = layers.Dense(128, activation="relu")(cnn)
    cnn = layers.Dropout(0.2)(cnn)
    cnn = layers.Dense(3, activation='softmax')(cnn)




    cnn = models.Model(inputs=[input_layer], outputs=cnn, name="cnn_pure")
    cnn.compile(
        optimizer=optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy']
    )
    history = cnn.fit(
        train_dataset.batch(32).prefetch(buffer_size=AUTOTUNE),
        validation_data=val_dataset.batch(32).prefetch(buffer_size=AUTOTUNE),
        epochs=100,
        callbacks=callbacks
    )
else:

    cnn_net = load_model('./training/epoch_4.hdf5')

    cnn_history = cnn_net.fit(
        train_dataset.take(10000).batch(32).prefetch(2),
        validation_data=val_dataset.batch(32).prefetch(2),
        epochs=100,
        verbose=1,
        callbacks=callbacks
    )


# COMMAND ----------


