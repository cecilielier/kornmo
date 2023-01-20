# Databricks notebook source
from keras.layers import TimeDistributed, GRU, Dense, Dropout
from keras.layers import Conv2D, BatchNormalization, GlobalMaxPool2D
from keras import Sequential

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


from kornmo.sentinel.storage import SentinelDataset
from kornmo.mask.mask_dataset import MaskDataset
from kornmo.mask.utils import apply_mask_to_image_series


data_path = 'data'
IMG_SIZE = 16
NUM_IMGS = 30
NUM_INDICES = 10
runtime_name = "all-indices-0-30-five-types-no-mask-optimized"


# COMMAND ----------


print("Reading data")
data = gpd.read_file(f"{data_path}/training_data.gpkg")
data['orgnr'] = data['orgnr'].astype(int)
data['year'] = data['year'].astype(int)

print("Reading masks")
masks = MaskDataset(f"{data_path}/mask/field_based_vegetation_indices_masks_16x16.h5")

print("Reading satellite images")
satellite_imgs = SentinelDataset(f"{data_path}/sentinelhub/field_based_images/field_based_vegetation_indices_16x16.h5")

print(f"Loaded {len(masks.labels)} masks")
print(f"Loaded {len(satellite_imgs.labels) * NUM_IMGS} images")



data = data.loc[data['planted'] != 'erter']
data = data.loc[data['planted'] != 'oljefro']
# data = data.loc[data['planted'] != 'rughvete']
# data = data.loc[data['planted'] != 'rug']

data.drop(data[data['area'] < 1500].index, inplace = True)



labels = list(set(data['planted']))
n_classes = len(labels)

print(pd.Series(list(data['planted'])).value_counts())

def add_labels(orgnr, year, data_arg):
    orgnr = int(orgnr[:9])
    year = int(year)

    if orgnr in data['orgnr'].unique() and year in data.loc[data['orgnr'] == orgnr]['year'].unique():
        label = data.loc[data['orgnr'] == orgnr].loc[data['year'] == year]['planted'].iloc[0]
        index = labels.index(label)
        arr = [0 for _ in range(0, len(labels))]
        arr[index] = 1
        return {'class': arr}




train, val = satellite_imgs.to_iterator().split(rand_seed='corn')
train = train.with_data(add_labels, show_progress=True)
val = val.with_data(add_labels, show_progress=True)

masks_it = masks.get_iterator()
mask_dict = {}

for orgnr, year, mask in masks_it:
    mask_dict[f'{orgnr}/{year}'] = mask

print(f"train samples: {len(train)}")
print(f"val samples: {len(val)}")


# COMMAND ----------


def train_generator():
    for orgnr, year, imgs, label in train:
        imgs = apply_mask_to_image_series(mask_dict[f'{orgnr}/{year}'], imgs[5:20], image_size=IMG_SIZE)
        # imgs = apply_mask_to_image_series(mask_dict[f'{orgnr}/{year}'], imgs[9:15, :, :, [1, 6, 7, 8]], image_size=IMG_SIZE)
        yield imgs[5:20], label['class']

def val_generator():
    for orgnr, year, imgs, label in val:
        imgs = apply_mask_to_image_series(mask_dict[f'{orgnr}/{year}'], imgs[5:20], image_size=IMG_SIZE)
        # imgs = apply_mask_to_image_series(mask_dict[f'{orgnr}/{year}'], imgs[9:15, :, :, [1, 6, 7, 8]], image_size=IMG_SIZE)
        yield imgs[5:20], label['class']


train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_types=(tf.dtypes.float64, tf.dtypes.int64),
    output_shapes=((NUM_IMGS, IMG_SIZE, IMG_SIZE, NUM_INDICES), n_classes)
)

val_dataset = tf.data.Dataset.from_generator(
    val_generator,
    output_types=(tf.dtypes.float64, tf.dtypes.int64),
    output_shapes=((NUM_IMGS, IMG_SIZE, IMG_SIZE, NUM_INDICES), n_classes)
)



# COMMAND ----------


def build_convnet(shape=(IMG_SIZE, IMG_SIZE, NUM_INDICES)):
    momentum = 0.9
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape=shape, padding='same', activation='relu'))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(GlobalMaxPool2D())
    return model



def action_model(shape=(NUM_IMGS, IMG_SIZE, IMG_SIZE, NUM_INDICES,), n_classes=n_classes):
    convnet = build_convnet(shape[1:])

    model = Sequential()

    model.add(TimeDistributed(convnet, input_shape=shape))
    model.add(GRU(12))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model


# COMMAND ----------


model = action_model(shape=(NUM_IMGS, IMG_SIZE, IMG_SIZE, NUM_INDICES,))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

cnn_history = model.fit(
        train_dataset.take(10000).batch(32).prefetch(4),
        validation_data=val_dataset.batch(32).prefetch(4),
        epochs=100,
        verbose=1,
)


fig, ax = plt.subplots(figsize=(20, 4))

ax.plot(model.history.history["acc"])
ax.plot(model.history.history["loss"])

ax.plot(model.history.history["val_" + "acc"])
ax.plot(model.history.history["val_" + "loss"])

ax.set_title("Classification Accuracy and Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Value")
ax.legend(["train_acc", "train_loss", "val_acc", "val_loss"])

plt.show()




# COMMAND ----------

acc = model.history.history["acc"]
loss = model.history.history["loss"]

val_acc = model.history.history["val_" + "acc"]
val_loss = model.history.history["val_" + "loss"]

print(acc)
print(loss)
print(val_acc)
print(val_loss)

# COMMAND ----------

cnn_history2 = model.fit(
        train_dataset.take(10000).batch(32).prefetch(4),
        validation_data=val_dataset.batch(32).prefetch(4),
        epochs=100,
        verbose=1,
)

# COMMAND ----------

_, ax = plt.subplots(figsize=(20, 4))

ax.plot(model.history.history["acc"])
ax.plot(model.history.history["loss"])

ax.plot(model.history.history["val_" + "acc"])
ax.plot(model.history.history["val_" + "loss"])

ax.set_title("Classification Accuracy and Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Value")
ax.legend(["train_acc", "train_loss", "val_acc", "val_loss"])

plt.show()
