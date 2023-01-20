# Databricks notebook source

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use("default")


# COMMAND ----------


average_indices_planted = pd.read_csv("../../kornmo-data-files/raw-data/crop-classification-data/average_vegetation_indices_planted_fixed.csv")
average_indices_planted.drop("Unnamed: 0", inplace=True, axis=1)

vegetation_indices = ["NDVI", "NDRE", "EVI", "SIPI3", "PVR", "GARI", "GRNDVI", "SIWSI", "LSWI", "NDSVI"]

crop_types = average_indices_planted['planted'].unique()

columns = ["index", "planted"]
for day in range(1, 31):
    columns.append(f"day_{day}")

final_dataset = pd.DataFrame(columns=columns)


for crop_type in crop_types:
    average_indices_for_crop = average_indices_planted[(average_indices_planted['planted'] == crop_type)]
    print(f"{crop_type}: {len(average_indices_for_crop)}")

    for vegetation_index in vegetation_indices:
        default_data = {'index': str(vegetation_index), 'planted': str(crop_type)}
        row = pd.Series(data=default_data, index=['index', 'planted'])

        for day in range(1, 31):
            temp_values = average_indices_for_crop[f"{vegetation_index}_{day}"].tolist()
            all_average_index = []

            for value in temp_values:
                if 0 <= value <= 1:
                    all_average_index.append(value)

            single_average_index = sum(all_average_index) / len(all_average_index)

            if single_average_index < 0 or single_average_index > 1:
                print(f"Min: {min(all_average_index)}, Max: {max(all_average_index)}, Average: {single_average_index}")

            row[f"day_{day}"] = single_average_index

        final_dataset = pd.concat([final_dataset, row.to_frame().T])




# COMMAND ----------


new_crop_types = ['Barley', 'Wheat', 'Oat', 'Rye', 'Rye Wheat', 'Oilseeds', 'Peas']
crop_types = ['bygg', 'hvete', 'havre', 'rug', 'rughvete']
# vegetation_indices = ["NDVI", "NDRE", "EVI", "SIPI3", "PVR", "GARI", "GRNDVI", "SIWSI", "LSWI", "NDSVI"]
vegetation_indices = ["NDRE", "GRNDVI", "SIWSI", "LSWI"]

        # 10 = 19       15 = 24
y= []   #9, 17
for i in range(18, 26):
    y.append(i)

for vegetation_index in vegetation_indices:

    counting = 0
    for crop_type in crop_types:

        x = final_dataset[(final_dataset['index'] == vegetation_index) & (final_dataset['planted'] == crop_type)].values.flatten().tolist()
                    # 10, 18
        plt.plot(y, x[10:18], label=new_crop_types[counting])
        counting = counting + 1


    plt.rcParams["figure.figsize"] = (10, 5)
    plt.xlabel('Week')
    plt.ylabel(f'{x[0]} values')
    plt.title(f'Average {x[0]} Vegetation Index')
    plt.rcParams.update({'axes.facecolor':'azure'})


    # Change how many lines to show
    plt.xticks(np.arange(19, 25, 1))
    plt.grid()
    plt.legend()
    plt.show()

