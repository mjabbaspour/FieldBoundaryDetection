import numpy as np


def bin_calculator(patch_step, stride, image_length):
    ind = 0
    patch_bins = []
    continue_flag = True

    while(continue_flag):

        left = ind * (patch_step - stride)
        ind += 1
        if (image_length - (left + patch_step - stride) < patch_step):
            right = image_length
            continue_flag = False
        else:
            right = left + patch_step

        patch_bins.append([left, right])

    return patch_bins


def patch_selector(data, h_ind, v_ind, patch_h_bins, patch_v_bins):
    left = patch_h_bins[h_ind][0]
    right = patch_h_bins[h_ind][1]
    upper = patch_v_bins[v_ind][0]
    lower = patch_v_bins[v_ind][1]

    im_crop = data.crop((left, upper, right, lower))

    return im_crop


def change_coordinates(polygons, v_ind, h_ind, patch_v_bins, patch_h_bins):
    new_polygons = []
    shift_h = patch_h_bins[h_ind][0]
    shift_v = patch_v_bins[v_ind][0]

    for plg in range(len(polygons)):

        polygon = polygons[plg]

        for point in range(len(polygon)):
            polygon[point] = (polygon[point][0] + shift_h, polygon[point][1] + shift_v)

        new_polygons.append(polygon)

    return new_polygons


def scale_polygon(polygons, offset, d_or_e):
    new_polygons = []

    for polygon in polygons:

        x = []
        y = []
        new_polygon = []

        for point in polygon:
            x.append(point[0])
            y.append(point[1])

        x_mean = np.mean(np.array(x[:-1]))
        y_mean = np.mean(np.array(y[:-1]))

        x -= x_mean
        y -= y_mean

        for x_point, y_point in zip(x, y):
            scale = offset / np.sqrt(x_point ** 2 + y_point ** 2) + 1
            if d_or_e == 'erode':
                point = (x_point / scale + x_mean, y_point / scale + y_mean)
            if d_or_e == 'dilate':
                point = (x_point * scale + x_mean, y_point * scale + y_mean)
            new_polygon.append(point)

        new_polygons.append(new_polygon)

    return new_polygons




