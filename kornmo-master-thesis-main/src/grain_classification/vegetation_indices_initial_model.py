# Databricks notebook source
import geopandas as gpd
import numpy as np
import tensorflow as tf
import os

from kornmo.sentinel.storage import SentinelDataset
from kornmo.mask.mask_dataset import MaskDataset
from kornmo.mask.utils import apply_mask_to_image_series
from keras.models import load_model
from tensorflow import optimizers
from keras import layers, models

data_path = 'data'

# COMMAND ----------

print("Reading data")
data = gpd.read_file(f"{data_path}/training_data.gpkg")
data['orgnr'] = data['orgnr'].astype(int)
data['year'] = data['year'].astype(int)

print("Reading masks")
masks = MaskDataset(f"{data_path}/mask/field_based_vegetation_indices_masks_16x16.h5")

satellite_imgs = SentinelDataset(f'{data_path}/sentinelhub/field_based_images/field_based_vegetation_indices_16x16.h5')
print("Done reading satellite images")


# COMMAND ----------

# Fewer classes code:

data = data.loc[data['planted'] != 'erter']
data = data.loc[data['planted'] != 'rughvete']
data = data.loc[data['planted'] != 'oljefro']
data = data.loc[data['planted'] != 'rug']


labels = list(set(data['planted']))
print(labels)

def add_labels(orgnr, year, data_arg):
    orgnr = int(orgnr[:9])
    year = int(year)

    if orgnr in data['orgnr'].unique() and year in data.loc[data['orgnr'] == orgnr]['year'].unique():
        label = data.loc[data['orgnr'] == orgnr].loc[data['year'] == year]['planted'].iloc[0]
        index = labels.index(label)
        arr = [0 for _ in range(0, len(labels))]
        arr[index] = 1
        return {'class': arr}




# COMMAND ----------

# All classes code:
"""
labels = list(set(data['planted']))
print(labels)

def add_labels(orgnr, year, data_arg):
    label = data.loc[data['orgnr'] == int(orgnr)].loc[data['year'] == int(year)]['planted'].iloc[0]
    index = labels.index(label)
    arr = [0 for i in range(0, len(labels))]
    arr[index] = 1
    return {'class': arr}
"""


# COMMAND ----------

train, val = satellite_imgs.to_iterator().split(rand_seed='corn')
train = train.with_data(add_labels)
val = val.with_data(add_labels)
masks_it = masks.get_iterator()
mask_dict = {}

for orgnr, year, all_masks in masks_it:
    merged_mask = np.zeros((100, 100))

    for mask in all_masks:
        merged_mask = merged_mask + mask

    for i in range(100):
        for j in range(100):
            if merged_mask[i][j] > 1:
                merged_mask[i][j] = 1

    mask_dict[f'{orgnr}/{year}'] = merged_mask

train = train.filter(lambda orgnr, year, _,__: f"{orgnr}/{year}" in mask_dict)
val = val.filter(lambda orgnr, year, _,__: f"{orgnr}/{year}" in mask_dict)

print(f"train samples: {len(train)}")
print(f"val samples: {len(val)}")


# COMMAND ----------

#import random

def apply_mask(orgnr, year, imgs):
    mask = mask_dict[f'{orgnr}/{year}']
    return apply_mask_to_image_series(mask, imgs)

def apply_output(orgnr, year, img_source, data):
    label = data['class']
    #num = random.randint(0, 29)
    return {"cnn_input": img_source[4:20]}, label

# COMMAND ----------

from tensorflow.python.data.experimental import assert_cardinality

train_dataset = tf.data.Dataset.from_generator(
    train.transform(apply_mask).apply(apply_output).shuffled(),
    output_types=({'cnn_input': tf.dtypes.float64}, tf.dtypes.int64)
).apply(assert_cardinality(len(train)))

val_dataset = tf.data.Dataset.from_generator(
    val.transform(apply_mask).apply(apply_output).shuffled(),
    output_types=({'cnn_input': tf.dtypes.float64}, tf.dtypes.int64)
).apply(assert_cardinality(len(val)))

print(f"training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# COMMAND ----------


def cnn(input_dim, output_dim):
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

# COMMAND ----------

from tensorflow import keras

runtime_name = "per-field-big-indices-few-classes"
num_images = 16
num_crop_Types = 3

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
    cnn_net = cnn((100, 100, 10), 64)
    input_cnn = layers.Input(shape=(num_images, 100, 100, 10), name="cnn_input")

    cnn = layers.TimeDistributed(cnn_net)(input_cnn)
    cnn = layers.GRU(128, return_sequences=False)(cnn)
    cnn = layers.Flatten()(cnn)
    cnn = layers.Dense(128)(cnn)
    cnn = layers.Dense(num_crop_Types, activation='softmax')(cnn)

    cnn = models.Model(inputs=input_cnn, outputs=cnn, name="CNN")

    cnn.compile(
        optimizer=optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy']
    )


    cnn_history = cnn.fit(
            train_dataset.take(10000).batch(32).prefetch(2),
            validation_data=val_dataset.batch(32).prefetch(2),
            epochs=10,
            verbose=1,
            callbacks=callbacks
    )
else:
    cnn_net = load_model(f'./results/{runtime_name}/epoch_2.hdf5')

    cnn_history = cnn_net.fit(
        train_dataset.take(10000).batch(32).prefetch(2),
        validation_data=val_dataset.batch(32).prefetch(2),
        epochs=10,
        verbose=1,
        callbacks=callbacks
    )

# COMMAND ----------


