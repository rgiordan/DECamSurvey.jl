

function lanczos_kernel{NumType <: Number}(x::NumType, a::Float64)
    abs(x) < a ? sinc(x) * sinc(x / a) : zero(promote_type(NumType, Float64))
end


function sinc_with_derivatives{NumType <: Number}(x::NumType)
    x_pi = pi * x
    sinc_x = sinc(x)
    sinc_x_d = (cos(x_pi) - sinc_x) / x
    sinc_x_h = -pi * (pi * sinc_x + 2 * sinc_x_d / x_pi)
    return sinc_x, sinc_x_d, sinc_x_h
end


# A function without checking a.  Factored out for testing with ForwardDiff..
function lanczos_kernel_with_derivatives_nocheck{NumType <: Number}(
    x::NumType, a::Float64)
    sinc_x, sinc_x_d, sinc_x_h = sinc_with_derivatives(x)
    sinc_xa, sinc_xa_d, sinc_xa_h = sinc_with_derivatives(x / a)

    return sinc_x * sinc_xa,
           sinc_x_d * sinc_xa + sinc_x * sinc_xa_d / a,
           sinc_x_h * sinc_xa + 2 * sinc_x_d * sinc_xa_d / a +
              sinc_x * sinc_xa_h / (a ^ 2)
end


function lanczos_kernel_with_derivatives{NumType <: Number}(x::NumType, a::Float64)
    T = promote_type(NumType, Float64)
    if abs(x) > a
        return T(0), T(0), T(0)
    end
    return lanczos_kernel_with_derivatives_nocheck(x, a)
end


function bspline_kernel_with_derivatives{NumType <: Number}(x::NumType)
    abs_x = abs(x)
    sign_x = sign(x)
    if abs_x > 2
        return zero(x)/1, zero(x)/1, zero(x)/1
    elseif abs_x > 1
        return (-1 * abs_x^3 + 6 * abs_x^2 - 12 * abs_x + 8) / 6,
               (-3 * abs_x^2 + 12 * abs_x  - 12) * sign_x / 6,
               (-6 * abs_x + 12) / 6
    else
        return (3 * abs_x^3 - 6 * abs_x^2 + 4) / 6,
               (9 * abs_x^2 - 12 * abs_x) * sign_x / 6,
               (18 * abs_x  - 12) / 6
    end
end


function cubic_kernel_with_derivatives{NumType <: Number}(x::NumType, a::Float64)
    T = promote_type(NumType, Float64)
    abs_x = abs(x)
    sign_x = sign(x)
    if abs_x > 2
        return T(0), T(0), T(0)
    elseif abs_x > 1
        return a * abs_x^3      -5 * a * abs_x^2 +  8 * a * abs_x - 4 * a,
               (3 * a * abs_x^2  -10 * a * abs_x +   8 * a) * sign_x,
               6 * a * abs_x    -10 * a
    else
        return (a + 2) * abs_x^3 -    (a + 3) * abs_x^2 + 1,
               (3 * (a + 2) * abs_x^2  -2 * (a + 3) * abs_x) * sign_x,
               6 * (a + 2) * abs_x    -2 * (a + 3)
    end
end


function cubic_kernel{NumType <: Number}(x::NumType, a::Float64)
    T = promote_type(NumType, Float64)
    abs_x = abs(x)
    sign_x = sign(x)
    if abs_x > 2
        return T(0)
    elseif abs_x > 1
        return a * abs_x^3 -5 * a * abs_x^2 +  8 * a * abs_x - 4 * a
    else
        return (a + 2) * abs_x^3 - (a + 3) * abs_x^2 + 1
    end
end



function add_interpolation_to_image!{NumType <: Number}(
        kernel, # A 1-d kernel function
        kernel_width::Int, # The width of the non-zero part of the kernel
        image::Matrix{NumType},
        psf_image::Matrix{Float64},
        h_range::UnitRange{Int64}, # h range in the image
        w_range::UnitRange{Int64}, # w range in the image
        object_loc::Vector{NumType},
        brightness::NumType)

    h_psf_width = (size(psf_image, 1) + 1) / 2.0
    w_psf_width = (size(psf_image, 2) + 1) / 2.0

    # h, w are pixel coordinates.
    for h in h_range, w in w_range
        # h_psf, w_psf are in psf coordinates.
        # The PSF is centered at object_loc + psf_width.
        h_psf = h - object_loc[1] + h_psf_width
        w_psf = w - object_loc[2] + w_psf_width

        # Centers of indices of the psf matrix, i.e., integer psf coordinates.
        h_ind0, w_ind0 = Int(floor(h_psf)), Int(floor(w_psf))
        h_lower = max(h_ind0 - kernel_width + 1, 1)
        h_upper = min(h_ind0 + kernel_width, size(psf_image, 1))
        for h_ind = (h_lower:h_upper)
            lh_v = kernel(h_psf - h_ind)
            if lh_v != 0
                w_lower = max(w_ind0 - kernel_width + 1, 1)
                w_upper = min(w_ind0 + kernel_width, size(psf_image, 2))
                for w_ind = (w_lower:w_upper)
                    lw_v = kernel(w_psf - w_ind)
                    if lw_v != 0
                        image[h, w] +=
                            brightness * lh_v * lw_v * psf_image[h_ind, w_ind]
                    end
                end
            end
        end
    end
end
