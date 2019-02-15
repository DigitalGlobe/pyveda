import numpy as np
import scipy.ndimage as ndi


def rescale_toa(arr, dtype=np.float32):
    """
    Rescale any multi-dimensional array of shape (D, M, N) by first subtracting
    the min from each (M, N) array along axis 0, then divide each by the
    resulting maximum to get (float32) distributions between [0, 1].
    Optionally spec uint8 for 0-255 ints.
    """
    # First look at raw value dists along bands

    arr_trans = np.subtract(arr, arr.min(axis=(1, 2))[
                            :, np.newaxis, np.newaxis])
    arr_rs = np.divide(arr_trans, arr_trans.max(
        axis=(1, 2))[:, np.newaxis, np.newaxis])
    if dtype == np.uint8:
        arr_rs = np.array(arr_rs * 255, dtype=np.uint8)
    return arr_rs

# Augmentation Functions Definition- written by Bernardo Henz
# https://github.com/bernardohenz/ExtendableImageDatagen/blob/master/util/extendable_datagen.py


def apply_transform(x, transform_matrix, channel_index=0,
                    fill_mode='nearest', cval=0.):
    """
    Helper function for randomly rotating an image between 1 and 359 degrees.
    """
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel,
                                                         final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    """
    Helper function for randomly rotating an image between 1 and 359 degrees.
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def random_rotation_f(x, rg, row_index=1, col_index=2, channel_index=0,
                      fill_mode='nearest', cval=0.):
    """
    Randomly rotate an image between 1 and 359 degrees for data augmentation.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def flipped_h(shp, bbox):
    """
    Horizontally flip one bounding box.
    """
    shp_center = (np.array(shp[1:]) / 2)[0]
    if (shp_center - bbox[0]) < 0:
        xmin_new = shp_center + (shp_center - bbox[0])
    else:
        xmin_new = shp_center - (shp_center - bbox[0])
    if (shp_center - bbox[2]) < 0:
        xmax_new = shp_center + (shp_center - bbox[2])
    else:
        xmax_new = shp_center - (shp_center - bbox[2])
    return [(xmin_new), bbox[1], (xmax_new), bbox[3]]


def flip_labels_horizontal(shp, klass_lst):
    """
    Adjust object detection image labels if image is horizontally rotated.
    """
    def flipp_klass(feature_lst):
        return [bbox if not bbox else flipped_h(
            shp, bbox) for bbox in feature_lst]
    return [flipp_klass(klass) for klass in klass_lst]


def flipped_v(shp, bbox):
    """
    Vertically flip one bounding box.
    """
    shp_center = (np.array(shp[1:]) / 2)[0]
    if (shp_center - bbox[1]) < 0:
        ymin_new = shp_center + (shp_center - bbox[1])
    else:
        ymin_new = shp_center - (shp_center - bbox[1])
    if (shp_center - bbox[3]) < 0:
        ymax_new = shp_center + (shp_center - bbox[3])
    else:
        ymax_new = shp_center - (shp_center - bbox[3])
    return [bbox[0], int(ymin_new), bbox[2], int(ymax_new)]


def flip_labels_vertical(shp, klass_lst):
    """
    Adjust object detection image labels if image is vertically rotated.
    """
    def flipp_klass(feature_lst):
        return [bbox if not bbox else flipped_v(
            shp, bbox) for bbox in feature_lst]
    return [flipp_klass(klass) for klass in klass_lst]
