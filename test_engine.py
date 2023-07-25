import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageFilter
from utils import patch_selector, scale_polygon, change_coordinates
import time
import copy
from matplotlib import pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageChops
from shapely.geometry import Polygon


def extracting_patch_polygons(data, patch_h_bins, patch_v_bins, models, angles):
    all_polygons = []

    for i in range(len(patch_v_bins)):
        for j in range(len(patch_h_bins)):
            im_crop = patch_selector(data, j, i, patch_h_bins, patch_v_bins)
            polygons = one_patch_process(im_crop, models, angles)
            all_polygons.append(polygons)
            print(
                f'Patch with horizental {j + 1}/{len(patch_h_bins)} and vertical {i + 1}/{len(patch_v_bins)} coordinates is processed.')

    return all_polygons


def one_patch_process(patch, models, angles):
    im_size = patch.size
    mask = Image.new('L', im_size, color=0)
    fill_val = len(angles) * len(models)

    for angle in angles:

        for i, model in enumerate(models):

            image = patch.rotate(angle)
            _, polygons = model.get_all_polygons(image)

            temp_mask = Image.new('L', im_size, color=0)

            for polygon in polygons:
                ImageDraw.Draw(temp_mask).line(polygon, width=8, fill=int(255 / fill_val))

            mask = ImageChops.add(mask, temp_mask.rotate(-1 * angle))

    mask = np.array(mask)
    mask[mask < (3 * int(255 / fill_val))] = 0

    mask = 255 - mask
    mask[mask < 255] = 0
    mask = np.dstack((mask, mask, mask))
    mask = Image.fromarray(mask)
    image2 = ImageChops.multiply(patch, mask)
    model = models[0]
    _, polygons = model.get_all_polygons(image2)

    new_polygons = scale_polygon(polygons, 10, 'dilate')

    return new_polygons


def merging_patches(data, all_polygons, patch_v_bins, patch_h_bins):

    start = time.time()

    masks = Image.new('L', data.size, color=0)

    for v_ind in range(len(patch_v_bins)):

        for h_ind in range(len(patch_h_bins)):

            current_index = v_ind * len(patch_h_bins) + h_ind
            polygons = copy.deepcopy(all_polygons[current_index])
            shifted_polygons = change_coordinates(polygons, v_ind, h_ind, patch_v_bins, patch_h_bins)

            new_polygons = scale_polygon(shifted_polygons, 15, 'erode')

            mask = Image.new('L', data.size, color=0)

            for polygon in new_polygons:

                ImageDraw.Draw(mask).polygon(polygon, outline=0, fill=255)

            masks = ImageChops.add(masks, mask)

    masks = np.array(masks)
    masks[masks > 0] = 255

    end = time.time()

    print(f'Merging Time: {end - start}')

    return masks


def post_processing1(data, masks, display='yes'):

    start = time.time()

    contours, _ = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt_polygons = []

    for object in contours:
        cnt_polygon = []
        for point in object:
            cnt_polygon.append((point[0][0], point[0][1]))

        if len(cnt_polygon) > 10:
            cnt_polygons.append(cnt_polygon)

    final_polygons = scale_polygon(cnt_polygons, 10, 'dilate')

    end = time.time()
    print(f'First PostProcessing Time: {end - start}')

    if display == 'yes':
        image = copy.deepcopy(data)

        for i in range(len(final_polygons)):
            ImageDraw.Draw(image).line(final_polygons[i], width=4, fill="yellow")

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.show()

    return final_polygons


def post_processing2(data, polygons, display='yes'):

    start = time.time()

    index_to_delete = []
    index_to_erode = []
    extra_mask = Image.new('L', data.size, color=0)

    for i, polygon_ in enumerate(polygons):
        temp_mask = Image.new('L', data.size, color=0)
        ImageDraw.Draw(temp_mask).polygon(polygon_, outline=0, fill=50)
        extra_mask = ImageChops.add(extra_mask, temp_mask)

    extra_mask = np.array(extra_mask)
    extra_mask[extra_mask < 51] = 0
    extra_mask[extra_mask > 0] = 255
    extra_mask = Image.fromarray(extra_mask)

    final_polygons = []

    for i, polygon in enumerate(polygons):

        current_mask = Image.new('L', data.size, color=0)
        ImageDraw.Draw(current_mask).polygon(polygon, outline=0, fill=255)
        val = np.sum(np.array(ImageChops.multiply(extra_mask, current_mask))) / (np.sum(np.array(current_mask)))

        if val >= 0.9:
            index_to_delete.append(i)

        if (val > 0.1) & (val < 0.9):
            index_to_erode.append(i)
            final_polygons.append(scale_polygon([polygon], 5, 'erode')[0])

        if val <= 0.1:
            final_polygons.append(polygon)

    end = time.time()

    print(f'Second PostProcessing Time: {end - start}')

    if display == 'yes':
        image = copy.deepcopy(data)

        for i in range(len(final_polygons)):
            ImageDraw.Draw(image).line(final_polygons[i], width=4, fill="yellow")

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.show()

    return final_polygons


def post_processing3(data, polygons, display='yes'):

    start = time.time()

    final_polygons = []
    for plg in range(len(polygons)):
        polygon = Polygon(polygons[plg])
        new_polygon = polygon.simplify(5, preserve_topology=True)
        x, y = new_polygon.exterior.coords.xy
        temp = []
        for i in range(len(x)):
            temp.append((x[i], y[i]))
        final_polygons.append(temp)

    end = time.time()
    print(f'Third PostProcessing Time: {end - start}')

    if display == 'yes':
        image = copy.deepcopy(data)

        for i in range(len(final_polygons)):
            ImageDraw.Draw(image).line(final_polygons[i], width=4, fill="yellow")

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.show()

    return final_polygons


