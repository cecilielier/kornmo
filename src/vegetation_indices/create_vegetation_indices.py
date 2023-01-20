# Databricks notebook source
import numpy
import h5py
import numpy as np
from tqdm import tqdm
from kornmo.sentinel.storage import SentinelDataset


# COMMAND ----------


def normalize_2d_img(img, new_max):
    min = np.min(img)
    max = np.max(img)
    new_img = []
    for i, row in enumerate(img):
        new_row = []
        for j, pixel in enumerate(row):
            new_pixel = (pixel - min) / (max - min) * new_max
            new_row.append(new_pixel)
        new_img.append(new_row)
    return np.array(new_img)


# Extracts the specified channel from the 12-band satellite images and normalizes the values.
def extract_channel(image, channel, normalize=True):
    shape = image.shape
    newImg = []
    for row in range(0, shape[0]):
        newRow = []
        for col in range(0, shape[1]):
            newRow.append(image[row][col][channel])
        newImg.append(newRow)

    if normalize:
        return normalize_2d_img(newImg, 1)
    else:
        return newImg


# Allow division by zero
numpy.seterr(divide='ignore', invalid='ignore')


def create_new_image(image):
    all_indices = np.zeros((100, 100, 10))

    band2 = extract_channel(image, 1)       # Blue channel
    band3 = extract_channel(image, 2)       # Green channel
    band4 = extract_channel(image, 3)       # Red channel
    band5 = extract_channel(image, 4)       # Red-Edge channel
    band7 = extract_channel(image, 6)       # VNIR channel
    band8 = extract_channel(image, 7)       # Main VNIR channel
    band8a = extract_channel(image, 8)      # VNIR channel
    band11 = extract_channel(image, 10)     # SWIR channel

    # Calculating NDVI: (band8 - band4) / (band8 + band4)
    NDVI = (band8 - band4) / (band8 + band4)

    # Calculating NDRE: (band7 - band5) / (band7 + band5)
    NDRE = (band7 - band5) / (band7 + band5)

    # Calculating EVI: 2.5 * ((band8 - band4) / ((band8 + 6*band4 - 7.5*band2) + 1))
    EVI = 2.5 * ((band8 - band4) / ((band8 + 6*band4 - 7.5*band2) + 1))

    # Calculating SIPI3: (band8 − band2) / (band8 − band4)
    SIPI3 = (band8 - band2) / (band8 - band4)

    # Calculating PVR: (band3 − band4) / (band3 + band4)
    PVR = (band3 - band4) / (band3 + band4)

    # Calculating GARI: (Band8 − (Band3 − (Band2 − Band4))) / (Band8 − (Band3 + (Band2 − Band4)))
    GARI = (band8 - (band3 - (band2 - band4))) / (band8 - (band3 + (band2 - band4)))

    # Calculating GRNDVI: (Band8 − (Band3 + Band5)) / (Band8 + (Band3 + Band5))
    GRNDVI = band8 - (band3 + band5) / (band8 + (band3 + band5))

    # Calculating SIWSI: (Band8a − Band11) / (Band8a + Band11)
    SIWSI = (band8a - band11) / (band8a + band11)

    # Calculating LSWI: (nir - swir) / (nir + swir)
    LSWI = (band8 - band11) / (band8 + band11)

    # Calculating NDSVI: (band11 - band4) / (band11 + band4)
    NDSVI = (band11 - band4) / (band11 + band4)

    for i in range(100):
        for j in range(100):
            all_indices[i][j][0] = NDVI[i][j]
            all_indices[i][j][1] = NDRE[i][j]
            all_indices[i][j][2] = EVI[i][j]
            all_indices[i][j][3] = SIPI3[i][j]
            all_indices[i][j][4] = PVR[i][j]
            all_indices[i][j][5] = GARI[i][j]
            all_indices[i][j][6] = GRNDVI[i][j]
            all_indices[i][j][7] = SIWSI[i][j]
            all_indices[i][j][8] = LSWI[i][j]
            all_indices[i][j][9] = NDSVI[i][j]

    return all_indices



def get_missing_index(image):
    band2 = extract_channel(image, 1)       # Blue channel
    band3 = extract_channel(image, 2)       # Green channel
    band4 = extract_channel(image, 3)       # Red channel
    band8 = extract_channel(image, 7)       # Main VNIR channel

    # Calculating GARI: (Band8 − (Band3 − (Band2 − Band4))) / (Band8 − (Band3 + (Band2 − Band4)))
    GARI = (band8 - (band3 - (band2 - band4))) / (band8 - (band3 + (band2 - band4)))

    return GARI




# COMMAND ----------


indices_dataset = SentinelDataset('E:/MasterThesisData/Satellite_Images/classification_indices.h5', create_if_missing=True)

with h5py.File('E:/MasterThesisData/Satellite_Images/satellite_images_train.h5', "r+") as f:

    images = f['images']
    for _, orgnum in enumerate(tqdm(images.keys(), total=len(images))):
        for year in images[orgnum]:
            new_farm_images = []

            # Check if images exists
            if not indices_dataset.contains(orgnum, year):
                for image in images[orgnum][year][()]:
                    indices_image = create_new_image(image)
                    new_farm_images.append(indices_image)

                indices_dataset.store_images(new_farm_images, orgnum, year)


# COMMAND ----------


# Added missing GARI indices

new_indices_dataset = SentinelDataset('E:/MasterThesisData/Satellite_Images/classification_indices_fixed3.h5', create_if_missing=True)
old_indices_dataset = SentinelDataset('E:/MasterThesisData/Satellite_Images/classification_indices.h5', create_if_missing=False)

with h5py.File('E:/MasterThesisData/Satellite_Images/satellite_images_train.h5', "r+") as f:

    images = f['images']
    for _, orgnum in enumerate(tqdm(images.keys(), total=len(images))):
        for year in images[orgnum]:
            vegetation_indices = old_indices_dataset.get_images(orgnr=orgnum, year=year)

            # Check if images exists
            if not new_indices_dataset.contains(orgnum, year):
                counter = 0
                for image in images[orgnum][year][()]:
                    gari = get_missing_index(image)

                    for i in range(100):
                        for j in range(100):
                            vegetation_indices[counter][i][j][5] = gari[i][j]

                    counter = counter + 1

                new_indices_dataset.store_images(vegetation_indices, orgnum, year)

# COMMAND ----------



new_indices_dataset = SentinelDataset('E:/MasterThesisData/Satellite_Images/classification_indices.h5', create_if_missing=False)
old_indices_dataset = SentinelDataset('E:/MasterThesisData/Satellite_Images/classification_indices_fixed3.h5', create_if_missing=False)


# COMMAND ----------

# Validating new dateset

print(old_indices_dataset.__len__())
print(new_indices_dataset.__len__())


with h5py.File('E:/MasterThesisData/Satellite_Images/classification_indices.h5', "r+") as file:

    old_images = file['images']
    print(len(old_images))
    for _, org in enumerate(tqdm(old_images.keys(), total=len(old_images))):
        for year in old_images[org]:
            if old_indices_dataset.contains(org, year):
                if not new_indices_dataset.contains(org, year):
                    print(f"Found org and year in old dataset that are not in the new. {org}, {year}")
                else:
                    old_indices = old_indices_dataset.get_images(orgnr=org, year=year)
                    new_indices = old_indices_dataset.get_images(orgnr=org, year=year)

                    if old_indices.shape != new_indices.shape:
                        print(f"Found org and year with different shapes. {org}, {year}, {old_indices.shape}, {new_indices.shape}")
            else:
                print(f"Could not find {org} {year} in old dataset!")



# COMMAND ----------


