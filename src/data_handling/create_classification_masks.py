# Databricks notebook source

import numpy as np
from src.data_handling.create_mask_dataset import *
from src.satellite_images.satellite_images import get_images_by_orgnr



classification_fields = gpd.read_file('../../kornmo-data-files/raw-data/crop-classification-data/training_data.gpkg')
classification_fields.drop(classification_fields[classification_fields['area'] < 1500].index, inplace = True)

masks_path = "../../kornmo-data-files/raw-data/crop-classification-data/classification_field_masks.h5"
create_mask_file(masks_path)

all_bounding_boxes = gpd.read_file('../../kornmo-data-files/raw-data/farm-information/farm-properties/bounding-boxes-previous-students/disponerte_eiendommer_bboxes.shp')

all_orgnr = classification_fields['orgnr'].drop_duplicates().reset_index(drop=True)

p_bar = tqdm(total=len(all_orgnr), iterable=all_orgnr.iteritems())
for _, orgnr in p_bar:
    farmers_fields = classification_fields[classification_fields['orgnr'] == orgnr]

    all_years = []
    for temp_year in farmers_fields['year'].value_counts().iteritems():
        all_years.append(temp_year[0])

    for year in all_years:
        yearly_fields = farmers_fields[farmers_fields['year'] == year].reset_index(drop=True)
        field_masks = np.zeros((len(yearly_fields), 100, 100))

        bounding_box = all_bounding_boxes[all_bounding_boxes['orgnr'] == int(orgnr)]

        if len(bounding_box) >= 1:
            one_bounding_box = bounding_box[bounding_box['year'] == int(year)]

            if len(one_bounding_box) != 1:
                one_bounding_box = bounding_box.head(1)

            bounding_box_polygon = convert_crs(one_bounding_box['geometry'])[0]
            bbox = boundingBox(bounding_box_polygon.centroid.y, bounding_box_polygon.centroid.x, 1)
            bbox = box(bbox[0], bbox[1], bbox[2], bbox[3])

            for index, field in yearly_fields.iterrows():
                field_polygon = convert_crs([field['geometry']])[0]
                field_masks[index] = generate_mask_image(bbox, field_polygon)

            insert_mask(masks_path, f"{int(orgnr)}/{int(year)}", field_masks)

        else:
            images = get_images_by_orgnr(str(int(orgnr)))

            if len(images.keys()) != 0:
                print(f"Found satellite images for {orgnr}, but no bounding boxes")
                print(images.keys())
                print(f"{len(yearly_fields)} fields are dropped")



# COMMAND ----------


