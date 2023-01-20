# Databricks notebook source
import h5py
import numpy as np
from tqdm import tqdm
from src.satellite_images.storage import SentinelDataset
import pandas as pd
import geopandas as gpd



# explicit function to normalize array
def normalize(arr, t_min=0, t_max=1):
    pixel_min, pixel_max = min(arr), max(arr)

    if pixel_min != pixel_max:
        norm_arr = []
        diff = t_max - t_min
        diff_arr = pixel_max - pixel_min

        for i in arr:
            temp = (((i - pixel_min)*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr
    else:
        return arr


def get_average_index_normalized_in_mask(all_vegetations, vegetation_number, mask):
    vegetation_index = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            vegetation_index[i][j] = all_vegetations[i][j][vegetation_number]

    vegetation_index = vegetation_index * mask

    index_values = []
    for i in range(100):
        for j in range(100):
            if vegetation_index[i][j] != 0:
                index_values.append(vegetation_index[i][j])

    if len(index_values) == 0:
        index_values.append(0)

    normalized_index_values = normalize(index_values)
    average_index = sum(normalized_index_values) / len(normalized_index_values)

    return average_index


def merge_masks(masks):
    complete_mask = np.zeros((100, 100))
    for one_mask in masks:
        complete_mask = complete_mask + one_mask

    for i in range(100):
        for j in range(100):
            if complete_mask[i][j] > 1:
                complete_mask[i][j] = 1

    return complete_mask



# COMMAND ----------

# ------------------------------ Creating Average Indices -------------------------------------

indices_dataset = SentinelDataset('E:/MasterThesisData/Satellite_Images/classification_indices.h5')

vegetation_indices = ["NDVI", "NDRE", "EVI", "SIPI3", "PVR", "GARI", "GRNDVI", "SIWSI", "LSWI", "NDSVI"]

columns = ["orgnr", "year"]
for indices_number in vegetation_indices:
    for day in range(1, 31):
        columns.append(f"{indices_number}_{day}")

average_vegetation_indices = pd.DataFrame(columns=columns)

skipped_years = 0


with h5py.File("../../kornmo-data-files/raw-data/crop-classification-data/classification_field_masks.h5", "r") as f:

    all_masks = f['masks']

    # For each organization numbers:
    p_bar = tqdm(all_masks.keys(), total=len(all_masks))
    for _, orgnum in enumerate(p_bar):

        # For each year:
        for year in all_masks[orgnum]:
            if indices_dataset.contains(orgnum, year):
                yearly_complete_mask = merge_masks(all_masks[orgnum][year])
                farmers_indices = indices_dataset.get_images(orgnum, year)

                default_data = {'orgnr': int(orgnum), 'year': int(year)}
                row = pd.Series(data=default_data, index=['orgnr', 'year'])

                # For each vegetation index:
                for indices_number in range(len(vegetation_indices)):

                    # For each day:
                    day_number = 1
                    for daily_image in farmers_indices:
                        average_vegetation_index = get_average_index_normalized_in_mask(daily_image, indices_number, yearly_complete_mask)
                        row[f"{vegetation_indices[indices_number]}_{day_number}"] = average_vegetation_index
                        day_number = day_number + 1

                average_vegetation_indices = pd.concat([average_vegetation_indices, row.to_frame().T])

            else:
                skipped_years = skipped_years + 1
                p_bar.set_description_str(f"Skippet {skipped_years} sets of years")

    average_vegetation_indices.reset_index(drop=True, inplace=True)
    average_vegetation_indices.to_csv("../../kornmo-data-files/raw-data/crop-classification-data/average_vegetation_indices.csv")




# COMMAND ----------


# -------------------------------- Adding crop type as feature ---------------------------------------
def equal_values_in_column(s):
    a = s.to_numpy()
    return (a[0] == a).all()


average_indices = pd.read_csv("../../kornmo-data-files/raw-data/crop-classification-data/average_vegetation_indices.csv")
average_indices.drop("Unnamed: 0", inplace=True, axis=1)
average_indices['planted'] = np.nan

fields = gpd.read_file('../../kornmo-data-files/raw-data/crop-classification-data/training_data.gpkg')
fields.drop(fields[fields['area'] < 1500].index, inplace = True)

for index, row in tqdm(average_indices.iterrows(), total=len(average_indices)):
    farmers_fields = fields[(fields['orgnr'] == row['orgnr']) & (fields['year'] == row['year'])]

    if equal_values_in_column(farmers_fields['planted']):
        row['planted'] = farmers_fields['planted'].head(1).values[0]
        average_indices.loc[index] = row

    else:
        average_indices.drop(index, inplace=True)

average_indices.to_csv("../../kornmo-data-files/raw-data/crop-classification-data/average_vegetation_indices_planted.csv")




# COMMAND ----------

average_indices = pd.read_csv("../../kornmo-data-files/raw-data/crop-classification-data/average_vegetation_indices_planted.csv")
average_indices.drop("Unnamed: 0", inplace=True, axis=1)

all_field_masks = SentinelDataset('../../kornmo-data-files/raw-data/crop-classification-data/classification_field_masks.h5')
indices_dataset = SentinelDataset('E:/MasterThesisData/Satellite_Images/classification_indices_fixed3.h5')

vegetation_indices = ["NDVI", "NDRE", "EVI", "SIPI3", "PVR", "GARI", "GRNDVI", "SIWSI", "LSWI", "NDSVI"]

