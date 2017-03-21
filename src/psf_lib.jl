function get_ra_dec_brick(ra, dec, bricks)
    row = (bricks[:ra1] .< ra .< bricks[:ra2]) & (bricks[:dec1] .< dec .< bricks[:dec2])
    return bricks[row, :]
end

function get_pixel_box(ra, dec, pixel_radius, wcs, image_size)
    pix_loc = floor(WCS.world_to_pix(wcs, [ra, dec]))
    pix_loc_h = Int(pix_loc[1])
    pix_loc_w = Int(pix_loc[2])
    pix_lower_h = max(1, pix_loc_h - pixel_radius)
    pix_lower_w = max(1, pix_loc_w - pixel_radius)
    pix_upper_h = min(image_size[1], pix_loc_h + pixel_radius)
    pix_upper_w = min(image_size[2], pix_loc_w + pixel_radius)
    return pix_lower_h:pix_upper_h, pix_lower_w:pix_upper_w
end

function load_catalog(f_tractor::FITSIO.FITS, wcs::WCS.WCSTransform, image_size::Tuple{Int64, Int64})
    catalog = DataFrame(objid=FITSIO.read(f_tractor[2], "objid"));
    catalog_cols =
        [ "type", "ra", "dec", "cpu_source", "decam_flux", "wise_flux",
          "fracdev", "shapeexp_r", "shapedev_r" ]
    for par in catalog_cols
        tab = FITSIO.read(f_tractor[2], par)
        if ndims(tab) == 1
            catalog[Symbol(par)] = tab
        else
            for row in 1:size(tab, 1)
                catalog[Symbol(par * "$row")] = tab[row, :]
            end
        end
    end
    world_loc = Array(catalog[[:ra, :dec]]);
    pix_loc = WCS.world_to_pix(wcs, world_loc');
    catalog[:pix_h] = pix_loc[1, :];
    catalog[:pix_w] = pix_loc[2, :];
    image_rows = 1 .<= (catalog[:pix_h] .<= size(image, 1)) & (1 .<= catalog[:pix_w] .<= size(image, 2));
    catalog = catalog[image_rows, :]
    return catalog
end

# Define star ranges.
type PixelRange
    objid::Int32
    h_range::UnitRange{Int64}
    w_range::UnitRange{Int64}
end



"""
Convert an (n - 1)-vector of real numbers to an n-vector on the simplex, where
the last entry implicitly has the untransformed value 1.
"""
function constrain_to_simplex(x)
    m = maximum(x)
    k = length(x) + 1
    @assert m < Inf
    z = similar(x, k)
    z[1:(k - 1)] = exp.(x .- m)
    z[k] = exp(-m)
    z = z / sum(z)
    return z
end


"""
Convert an n-vector on the simplex to an (n - 1)-vector in R^{n -1}.  Entries
are expressed relative to the last element.
"""
function unconstrain_simplex(z)
    n = length(z)
    [ log(z[i]) - log(z[n]) for i = 1:(n - 1)]
end


function simplexify_parameter(free_param, lb::Float64, scale::Float64)
    # Broadcasting doesn't work with DualNumbers and Floats. :(
    # z_sim is on an unconstrained simplex.
    n = length(free_param) + 1
    z_sim = constrain_to_simplex([ p / scale for p in free_param ])
    param = [ (1 - n * lb) * p + lb for p in z_sim ]

    param
end


"""
Invert the transformation simplexify_parameter() by converting an n-vector
on a simplex to R^{n - 1}.
"""
function unsimplexify_parameter(param, lb::Float64, scale::Float64)
    n = length(param)
    @assert all(param .>= lb)
    @assert(abs(sum(param) - 1) < 1e-14, abs(sum(param) - 1))

    # z_sim is on an unconstrained simplex.
    # Broadcasting doesn't work with DualNumbers and Floats. :(
    z_sim = [ (p - lb) / (1 - n * lb) for p in param ]
    free_param = [ p * scale for p in unconstrain_simplex(z_sim) ]

    free_param
end
