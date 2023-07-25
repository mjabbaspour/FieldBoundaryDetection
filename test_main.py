from utils import bin_calculator
from test_engine import extracting_patch_polygons, merging_patches, post_processing1, post_processing2, post_processing3
from pathlib import Path
from matplotlib import pyplot as plt
import sys
from PIL import Image, ImageDraw
import os
import time
import copy
import torch
from tkinter.filedialog import askopenfilename
from google_map_downloader import GoogleMapDownloader
import geopandas as gpd
from shapely.geometry import Polygon

# Constants

angles = [0, 90, 180, 270]
stride = 500
patch_width = 1000
patch_height = 1000
DoSave = True
Main_dir = Path.cwd()
Main_Dir = copy.deepcopy(Main_dir)
Image_Source = 'online'
lat_lng = [29.965254974894275, 52.80428189845966]  # Enter if Image_Source = 'online'
MODEL_NAMES = ['mask_rcnn_rotated', 'mask_rcnn_BigDataset', 'mask_rcnn_selected_iran']
RawImages_Dir = "Data\\TestResults\\RawImages"
Shp_Files_Dir = "Data\\TestResults\\ShapeFiles"
Results_Dir = "Data\\TestResults\\Results"


def loading_models(names=MODEL_NAMES, address=Main_Dir):

    sys.path.append(address)
    sys.path.append(os.path.join(address, 'src'))
    sys.path.append(os.path.join(address, 'src\\agoro_field_boundary_detector'))
    sys.path.append(os.path.join(address, 'src\\agoro_field_boundary_detector\\field_detection'))

    from src.agoro_field_boundary_detector.field_detection.model import FieldBoundaryDetector

    # Loading Trained Models

    models = []

    for model_name in names:
        name = Path.cwd() / 'models' / model_name
        models.append(FieldBoundaryDetector(name))

    return models


def loading_image(image_source=Image_Source, address=Main_Dir, display='yes'):

    if image_source == 'offline':
        data = Image.open(address)

    if image_source == 'online':
        zoom = 18
        gmd = GoogleMapDownloader(address[0], address[1], zoom)
        data = gmd.generateImage()

    if display == 'yes':

        plt.figure(figsize=(15, 15))
        plt.imshow(data)
        plt.show()

    return data


# main code

if __name__ == "__main__":

    models = loading_models(names=MODEL_NAMES, address=Main_Dir)

    # offline addressing
    if Image_Source == 'offline':
        # offline addressing
        address = askopenfilename(initialdir=os.path.join(Main_Dir, RawImages_Dir))
        tmp = address.split("/")
        tmp = tmp[-1]
        name_to_save = tmp[:-4]
        data = loading_image(image_source=Image_Source, address=address, display='yes')
        data.save(os.path.join(Main_Dir, Results_Dir, name_to_save + '.png'))

    elif Image_Source == 'online':
        # online addressing

        address = lat_lng
        data = loading_image(image_source=Image_Source, address=address, display='yes')
        name_to_save = str(lat_lng[0]) + '_' + str(lat_lng[0])
        if DoSave:
            data.save(os.path.join(Main_Dir, RawImages_Dir, name_to_save + '.png'))
            data.save(os.path.join(Main_Dir, Results_Dir, name_to_save + '.png'))

    im_size = data.size
    image_width = data.size[0]
    image_height = data.size[1]
    # patch_width = im_size[0]
    # patch_height = im_size[1]

    patch_h_bins = bin_calculator(patch_width, stride, image_width)
    patch_v_bins = bin_calculator(patch_height, stride, image_height)

    num_h_patch = len(patch_h_bins)
    num_v_patch = len(patch_v_bins)

    # print(patch_h_bins)
    # print(patch_v_bins)
    # print(image_width, image_height)

    start = time.time()
    all_polygons = extracting_patch_polygons(data, patch_h_bins, patch_v_bins, models, angles)
    end = time.time()
    # with open('all_polygons_' + name_to_save, 'wb') as f:
    #     pickle.dump(all_polygons, f)
    print(f'All patches are passed through the model. elapsed time: {end - start}')

    # with open('all_polygons_img2_1000_500_whole_mix_patch_dilated', 'rb') as f:
    #     all_polygons = pickle.load(f)
    # print('all polygons are loaded.')

    masks = merging_patches(data, all_polygons, patch_v_bins, patch_h_bins)

    new_all_polygons = post_processing1(data, masks, display='yes')

    new_all_polygons = post_processing2(data, new_all_polygons, display='yes')

    new_all_polygons = post_processing3(data, new_all_polygons, display='yes')

    if DoSave:

        image = copy.deepcopy(data)

        for i in range(len(new_all_polygons)):
            ImageDraw.Draw(image).line(new_all_polygons[i], width=4, fill="yellow")

        image.save(os.path.join(Main_Dir, Results_Dir, name_to_save + '_processed.png'))

        gdr = gpd.GeoDataFrame({'feature': [i for i in range(len(new_all_polygons))],
                                'geometry': [Polygon(new_all_polygons[i]) for i in range(len(new_all_polygons))]})
        gdr.to_file(os.path.join(Main_Dir, Shp_Files_Dir, name_to_save + '.shp'))


