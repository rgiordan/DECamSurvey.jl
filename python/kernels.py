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


def cubic_shift_image(shift, image, new_image, a):
    # This shift is intended to be the decimal part of the only of the full shift.
    cubic_offsets = np.array([-2, -1, 0, 1])

    all_shifts = np.array([ shift + offset for offset in cubic_offsets ])
    # The row is the shift, the column is the axis.
    kern = eval_cubic_kernel(all_shifts, a)

    # Think of it this way: the kernel defines a function in the coordinates of the original
    # image at non-integer locations, where integer locations are the centers of the old
    # pixel values.  Since the cubic kernel is four uints wide, the expanded image is larger
    # than the original image.  This is evaluating the kernel function on an evenly spaced grid
    # offset from the integer locations by an amount -<shift>.
    new_image.fill(0.0)
    for xi in range(len(cubic_offsets)):
        for yi in range(len(cubic_offsets)):
            xo = cubic_offsets[xi]
            yo = cubic_offsets[yi]
            x_slice = slice(1 - xo, 1 + image.shape[0] - xo)
            y_slice = slice(1 - yo, 1 + image.shape[1] - yo)
            new_image[x_slice, y_slice] += kern[xi, 0] * kern[yi, 1] * image


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
