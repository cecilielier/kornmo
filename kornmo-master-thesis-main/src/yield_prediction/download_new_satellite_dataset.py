# Databricks notebook source
import os
import sys

#Auto update imports when python files in src is updated
%load_ext autoreload
%autoreload 2
import sys

from src.utils import get_disp_eiendommer, convert_crs, boundingBox, to_rgb
from kornmo.sentinel.sentinel_helpers import download_timeseries_from_bbox
from kornmo.sentinel.sentinel_evalscripts import natural_color, all_bands
from kornmo.sentinel.storage import SentinelDataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


data_location = "../../../kornmo-data-files/raw-data"

# COMMAND ----------

coordinates = pd.read_csv(os.path.join(data_location, 'farm-information/old farm location information/centroid_coordinates_new.csv'))
dataset = pd.read_csv(os.path.join(data_location, '../dataset_filtered-and-normalized.csv'))
disp_properties = get_disp_eiendommer()

disp_properties.head()

# COMMAND ----------

print(len(list(set(disp_properties['orgnr'].tolist()))))
print(len(list(disp_properties['orgnr'])))
row = disp_properties.loc[disp_properties['orgnr'] == str(orgnrs[i])].sort_values(by=['year']).iloc[0]


# COMMAND ----------

def append_orgnr(orgnr):
    file = open('download_progress.txt', 'a')
    file.write(f"{str(orgnr)},")
    file.close()

def get_done_orgnrs():
    file = open('download_progress.txt', 'r')
    nrs = file.read().split(',')[:-1]
    file.close()
    return nrs

# COMMAND ----------

years = [2020]
sd = SentinelDataset('E:/MasterThesisData/Satellite_Images/sentinel_100x100_new_data.h5', create_if_missing=True)
orgnrs = [str(orgnr) for orgnr in list(set(dataset['orgnr'].tolist()))]
disp_orgnrs = list(set(disp_properties['orgnr'].tolist()))

for i in tqdm(range(0, len(orgnrs)), desc=f'Prosessing images...'):
    done_orgnrs = get_done_orgnrs()
    #print(len(done_orgnrs))
    if (orgnrs[i] in disp_orgnrs) and (orgnrs[i] not in done_orgnrs):
        row = disp_properties.loc[disp_properties['orgnr'] == orgnrs[i]].sort_values(by=['year']).iloc[0]
        point = convert_crs([row['geometry'].centroid])[0]
        bbox = boundingBox(point.y, point.x, 1)
        for year in years:
            imgs, _ = download_timeseries_from_bbox(bbox, (year, 3, 1), (year, 10, 1), 30, evalscript=all_bands)
            sd.store_images(imgs, row['orgnr'], year)
        append_orgnr(orgnrs[i])

    else:
        #print(f"Skipping {orgnrs[i]}")
        ...


# COMMAND ----------



# COMMAND ----------


