# Databricks notebook source

import h5py
from tqdm import tqdm
import os
from src.data_handling import create_mask_dataset as mask_code
import pandas as pd
import geopandas as gpd


runtime_name = "week_1_16"
data_path = "../../kornmo-data-files/raw-data/crop-classification-data"
crop_types = ['hvete', 'bygg', 'havre', 'rug_og_rughvete']


masks_path = f"{data_path}/{runtime_name}/{runtime_name}_masks.h5"
mask_code.create_mask_file(masks_path)

all_fields = gpd.read_file(f"{data_path}/all_data.gpkg")
all_predicted_labels = pd.read_csv(f"{data_path}/{runtime_name}/predicted_values.csv")
all_bounding_boxes = gpd.read_file('../../kornmo-data-files/raw-data/farm-information/farm-properties/bounding-boxes-previous-students/disponerte_eiendommer_bboxes.shp')



# COMMAND ----------


def create_crop_specific_mask(orgnum, year, crop_type):

    if crop_type == "rug_og_rughvete":
        fields = all_predicted_labels[(all_predicted_labels['orgnr'] == int(orgnum)) & (all_predicted_labels['year'] == int(year)) & ((all_predicted_labels['crop_type'] == "rug") | (all_predicted_labels['crop_type'] == "rughvete"))]

    else:
        fields = all_predicted_labels[(all_predicted_labels['orgnr'] == int(orgnum)) & (all_predicted_labels['year'] == int(year)) & (all_predicted_labels['crop_type'] == crop_type)]

    field_ids = fields["field_id"].tolist()


    if len(field_ids) > 0:

        all_property = all_bounding_boxes[all_bounding_boxes['orgnr'] == int(orgnum)]

        if len(all_property) >= 1:
            property = all_property[all_property['year'] == int(year)]

            if len(property) != 1:
                property = all_property.head(1)


            property_polygon = mask_code.convert_crs(property['geometry'])[0]
            bbox = mask_code.boundingBox(property_polygon.centroid.y, property_polygon.centroid.x, 1)
            bbox = mask_code.box(bbox[0], bbox[1], bbox[2], bbox[3])


            field_geometries = []
            total_area = 0
            for i in range(len(field_ids)):
                row = all_fields.loc[field_ids[i]]
                total_area = total_area + row['area']
                field_geometries.append(row['geometry'])


            mask = mask_code.generate_mask_image_from_polygons(bbox, field_geometries)


            return mask, total_area

        else:
            print(f"Found satellite images for {orgnum}, but no property")


    return -1, -1



# COMMAND ----------


satellite_image_location = "E:/MasterThesisData/Satellite_Images/"
field_areas = pd.DataFrame(columns=['orgnr', 'year', 'crop_type', 'area'])
counter = 0


for filename in ['sentinel_100x100_0.h5', 'sentinel_100x100_1.h5']:
    with h5py.File(os.path.join(satellite_image_location, filename), "r") as file:

        images = file['images']
        for _, orgnum in enumerate(tqdm(images.keys(), total=len(images))):
            for year in images[orgnum]:
                    for crop_type in crop_types:

                        new_crop_mask, area = create_crop_specific_mask(orgnum, year, crop_type)

                        if area != -1:
                            mask_code.insert_mask(masks_path, f"{int(orgnum)}/{int(year)}/{crop_type}", new_crop_mask)

                            data = {'orgnr': int(orgnum), 'year': int(year), 'crop_type': crop_type, 'area':area}
                            row = pd.Series(data=data, index=['orgnr', 'year', 'crop_type', 'area'])
                            field_areas = pd.concat([field_areas, row.to_frame().T])

                        else:
                            counter = counter + 1

    file.close()

field_areas.to_csv(f"{data_path}/{runtime_name}/field_areas.csv")

print("Done")
print(f"Skipped {counter} sets of [farm, year, type]")



# COMMAND ----------


