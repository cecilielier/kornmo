# Databricks notebook source
from tqdm import tqdm
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from src.satellite_images.storage import SentinelDataset
from src.utils import to_rgb

PREDICTED_VALUES_PATH = '../../kornmo-data-files/raw-data/crop-classification-data/week_1_16/predicted_values.csv'
MODEL_PATH = '../src/grain_classification/training/models/classification_1-16.hdf5'
CLASSES = ['bygg', 'rug', 'havre', 'rughvete', 'hvete']
END_DAY = 16

# COMMAND ----------

def read_data():
    all_fields = SentinelDataset('E:/MasterThesisData/Satellite_Images/small_images_all.h5')
    model = load_model(MODEL_PATH)
    all_fields_it = all_fields.to_iterator()

    return all_fields_it, model

# COMMAND ----------

print("Reading data...")
all_fields_it, model = read_data()

# COMMAND ----------

def val_generator():
    for orgnr, year, imgs, label in all_fields_it:
        imgs = imgs[0:END_DAY]
        yield imgs

val_dataset = tf.data.Dataset.from_generator(
    val_generator,
    output_types=tf.dtypes.float64,
    output_shapes=(END_DAY, 16, 16, 12)
)


predicted_values = model.predict(val_dataset.batch(32).prefetch(2), verbose=1)

# COMMAND ----------

import numpy as np

guesses = {'havre': 0, 'bygg': 0, 'rug': 0, 'rughvete': 0, 'hvete': 0}
for pred in predicted_values:
    guesses[CLASSES[np.argmax(pred)]] += 1
print(guesses)

print(predicted_values.shape)
predicted_values_dataset = []
for vals, pred_arr in tqdm(zip(all_fields_it, predicted_values), total=len(predicted_values)):
    orgnr = int(vals[0][0:9])
    field_id = int(vals[0][9:])
    year = int(vals[1])
    prediction = CLASSES[np.argmax(pred_arr)]

    # print(orgnr, field_id, year, prediction)
    predicted_values_dataset.append([orgnr, field_id, year, prediction])

predicted_values_dataset_df = pd.DataFrame(predicted_values_dataset, columns=['orgnr', 'field_id', 'year', 'crop_type'])
predicted_values_dataset_df.to_csv(PREDICTED_VALUES_PATH)
predicted_values_dataset_df.head()


