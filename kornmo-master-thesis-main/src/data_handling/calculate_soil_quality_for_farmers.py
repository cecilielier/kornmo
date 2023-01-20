# Databricks notebook source
import numpy as np
import pandas as pd
import statistics
import requests
from tqdm import tqdm
from src.utils import get_disp_eiendommer, convert_crs

# COMMAND ----------

all_farmers = pd.read_csv("../../kornmo-data-files/raw-data/farm-information/farmers_with_address_and_coordinates.csv", usecols=['orgnr', 'longitude', 'latitude', 'elevation'])
print(all_farmers.columns)

soil_quality = pd.read_csv("../../kornmo-data-files/raw-data/soil-data/jordsmonn.csv", usecols=['id', 'JORDKVALIT', 'JORDKVALITET'])
print(soil_quality.columns)

farmers_fields = pd.read_csv("../../kornmo-data-files/raw-data/fields_per_farm.csv", usecols=['field_id', 'municipal_nr', 'orgnr'])
print(farmers_fields.columns)

# COMMAND ----------


number_of_farmers = len(all_farmers)
farmers_missing_fields = 0

p_bar = tqdm(total=number_of_farmers, iterable=all_farmers.iterrows())
for farm_index, farmer in p_bar:

    # Get all field IDs for one farmer
    field_ids = farmers_fields['field_id'].loc[farmers_fields['orgnr'] == farmer['orgnr']].tolist()
    farmers_soil_quality = np.zeros(len(field_ids))


    # If farmer has some field IDs connected to it:
    if len(field_ids) != 0:

        # Get all soil quality for that farmer's field IDs
        for i in range(len(field_ids)):
            farmers_soil_quality[i] = int(soil_quality['JORDKVALITET'].loc[soil_quality['id'] == field_ids[i]])

        farmers_soil_quality = farmers_soil_quality.tolist()

        # If we found a soil quality for each of the farmers fields:
        if int(farmers_soil_quality.count(0)) == 0:
            all_farmers.loc[farm_index, 'mean_soil_quality'] = statistics.mean(farmers_soil_quality)
            all_farmers.loc[farm_index, 'soil_quality_1'] = int(farmers_soil_quality.count(1))
            all_farmers.loc[farm_index, 'soil_quality_2'] = int(farmers_soil_quality.count(2))
            all_farmers.loc[farm_index, 'soil_quality_3'] = int(farmers_soil_quality.count(3))

        else:
            print(f"Farmer {farmer} is missing {int(farmers_soil_quality.count(0))} soil qualities in jordsmonn.csv")

    else:
        farmers_missing_fields = farmers_missing_fields + 1

    p_bar.set_description(f"Done with {farm_index} of {number_of_farmers}")


print(f"There are {farmers_missing_fields} farmers without a field in fields_per_farm.csv")

if all_farmers.isna().sum().sum() != 0:
    print(f"There are {all_farmers.isna().sum().sum()} NaN values in the dataset")
    print(all_farmers.isna().sum())

# all_farmers.to_csv("../../kornmo-data-files/raw-data/farm-information/all-farmers-with-soil_quality.csv")

