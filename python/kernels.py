import autograd.numpy as np


def eval_cubic_kernel(x, a):
    abs_x = np.abs(x)
    result_12 = np.where(np.logical_and(abs_x < 2, abs_x >=1),
                         a * abs_x**3 - 5 * a * abs_x**2 + 8 * a * abs_x - 4 * a,
                         np.zeros_like(x))

    result_01 = np.where(np.logical_and(abs_x < 1, abs_x >= 0),
                         (a + 2) * abs_x**3 - (a + 3) * abs_x**2 + 1,
                         np.zeros_like(x))

    return result_01 + result_12


# TODO: add shape checking.
def eval_cubic_interpolation(w_locs, h_locs, image_padded, a):
    w_ind = [ int(x) for x in np.trunc(w_locs) ]
    h_ind = [ int(x) for x in np.trunc(h_locs) ]
    w_delta = w_locs - w_ind
    h_delta = h_locs - h_ind

    cubic_offsets = np.array([-2, -1, 0, 1])

    w_offsets = np.array([ w_delta + offset for offset in cubic_offsets ])
    h_offsets = np.array([ h_delta + offset for offset in cubic_offsets ])

    # The row is the shift, the column is the axis.
    w_kern = eval_cubic_kernel(w_offsets, a)
    h_kern = eval_cubic_kernel(h_offsets, a)

    #def eval_at_offset_index(wi, hi):
    #    wo = cubic_offsets[wi]
    #    ho = cubic_offsets[hi]
    #    return np.outer(w_kern[wi, :], h_kern[hi, :]) * image_padded[np.ix_(w_ind + wo, h_ind + ho)]

    offset_range = range(len(cubic_offsets))
    image_interp_offsets = np.array(
        [[ np.outer(w_kern[wi, :], h_kern[hi, :]) * \
           image_padded[np.ix_(w_ind + cubic_offsets[wi], h_ind + cubic_offsets[hi])]
           for wi in offset_range ] \
           for hi in offset_range ])

    return np.sum(image_interp_offsets, (0, 1))



# Shift an image on a grid with the same scale as the original image.
# This is more efficient since it doesn't require a kernel evaluation
# for each grid point.
#
# The resulting image will be bigger than the original image due to the fact that the
# cubic interpolation spreads the image out.
# In order to express the larger image as a sum of shifted versions of the original image,
# you must doubly pad the original image.
#
# To put it another way, this only interpolates inside a padded image.  It
# reduces the padded image size by two pixels in each dimension, which remains
# two pixels larger than the original image.
#
# TODO: add shape checking.
def eval_cubic_interpolation_same_spacing(shift, image_padded, a):
    cubic_offsets = np.array([-2, -1, 0, 1])

    w_offsets = np.array([ shift[0] + offset for offset in cubic_offsets ])
    h_offsets = np.array([ shift[1] + offset for offset in cubic_offsets ])

    w_kern = eval_cubic_kernel(w_offsets, a)
    h_kern = eval_cubic_kernel(h_offsets, a)

    image_interp_offsets = np.array(
        [[ w_kern[wi] * h_kern[hi] * \
           image_padded[slice(2 + cubic_offsets[wi],
                              image_padded.shape[0] - 2 + cubic_offsets[wi]),
                        slice(2 + cubic_offsets[hi],
                              image_padded.shape[0] - 2 + cubic_offsets[hi])]
           for wi in range(len(cubic_offsets)) ] \
           for hi in range(len(cubic_offsets)) ])

    return np.sum(image_interp_offsets, (0, 1))
