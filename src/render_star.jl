import FITSIO
import WCS
#using PyPlot
using DataFrames
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using Optim

data_path = joinpath(ENV["GIT_REPO_LOC"], "DECamSurvey.jl", "dat")
src_path = joinpath(ENV["GIT_REPO_LOC"], "DECamSurvey.jl", "src")
include(joinpath(src_path, "psf_lib.jl"))
include(joinpath(src_path, "kernels.jl"))

fname_head = "c4d_160302_094418_oo"
fname_tail = "_z_v1.fits"
join([ fname_head, "i", fname_tail ])

f_image = FITSIO.FITS(joinpath(data_path, join([ fname_head, "i", fname_tail ])));
f_weight = FITSIO.FITS(joinpath(data_path, join([ fname_head, "w", fname_tail ])));
f_mask = FITSIO.FITS(joinpath(data_path, join([ fname_head, "d", fname_tail ])));

header = FITSIO.read_header(f_image[1]);
asec_per_pixel = (header["PIXSCAL1"], header["PIXSCAL2"])
header["OBJECT"] # You can use this to look up in DR3 survey-ccds-decals.fits.gz

im_ind = 4;

image = FITSIO.read(f_image[im_ind]);
weight = FITSIO.read(f_weight[im_ind]);
mask = FITSIO.read(f_mask[im_ind]);

image[mask .== 1] = NaN

im_header = FITSIO.read_header(f_image[im_ind]);
im_header_string = FITSIO.read_header(f_image[im_ind], String);
gaina = im_header["GAINA"]
gainb = im_header["GAINB"]

# Get the corners, confirm that WCS matches up with them.
wcs = WCS.from_header(im_header_string)[1];

# Get the brick ids here
# http://legacysurvey.org/dr3/files/
# It appears that bricks are RA, DEC rectangles.
# survey-bricks.fits.gz
f_bricks = FITSIO.FITS(joinpath(data_path, "survey-bricks.fits"))
FITSIO.read_header(f_bricks[2]);
f_bricks[2]
bricks = DataFrame(
    brickname=FITSIO.read(f_bricks[2], "brickname"),
    brickid=FITSIO.read(f_bricks[2], "brickid"),
    dec1=FITSIO.read(f_bricks[2], "dec1"),
    dec2=FITSIO.read(f_bricks[2], "dec2"),
    ra1=FITSIO.read(f_bricks[2], "ra1"),
    ra2=FITSIO.read(f_bricks[2], "ra2"));

# brick_row = get_ra_dec_brick(obj_loc[1], obj_loc[2], bricks);

# Load the catalogs with the brick ids
# http://legacysurvey.org/dr3/catalogs/
# │ Row │ brickname  │ brickid │ dec1   │ dec2   │ ra1     │ ra2     │
# ├─────┼────────────┼─────────┼────────┼────────┼─────────┼─────────┤
# │ 1   │ "1984p110" │ 394182  │ 10.875 │ 11.125 │ 198.305 │ 198.559 │

f_tractor = FITSIO.FITS(joinpath(data_path, "tractor-1984p110.fits"));
tractor_header = FITSIO.read_header(f_tractor[2]);

catalog = load_catalog(f_tractor, wcs, size(image));
star_rows = catalog[:type] .== "PSF";

# NaN out pixels that are near galaxies.
gal_catalog = catalog[!star_rows, :];
filter_image = deepcopy(image);
for row in 1:nrow(gal_catalog)
    ra = gal_catalog[row, :ra]
    dec = gal_catalog[row, :dec]
    radius = Int(ceil(8 * max(gal_catalog[row, :shapeexp_r],
                              gal_catalog[row, :shapedev_r])))
    radius = max(radius, 3)
    h_range, w_range = get_pixel_box(ra, dec, radius, wcs, size(image))
    filter_image[h_range, w_range] = NaN
end


star_catalog = catalog[star_rows & (catalog[:decam_flux5] .> 50), :];
pixel_ranges = PixelRange[]
for row in 1:nrow(star_catalog)
    ra = star_catalog[row, :ra]
    dec = star_catalog[row, :dec]
    objid = star_catalog[row, :objid]
    h_range, w_range = get_pixel_box(ra, dec, 10, wcs, size(image))
    push!(pixel_ranges, PixelRange(objid, h_range, w_range))
end

star_image = fill(NaN, size(image));
for pr in pixel_ranges
    star_image[pr.h_range, pr.w_range] = image[pr.h_range, pr.w_range]
end

function gaussian_at_point(pix_loc::Vector{Float64},
                           pix_center::Vector{Float64},
                           radius_pix::Float64)
    r = pix_loc - pix_center
    return exp(-0.5 * dot(r, r) / (radius_pix ^ 2))
end

psf_image_orig = fill(0.0, (12, 12));
psf_radius = 1.5
psf_center = 0.5 * Float64[size(psf_image_orig, 1) + 1, size(psf_image_orig, 2) + 1];
for h in 1:size(psf_image_orig, 1), w in 1:size(psf_image_orig, 2)
    psf_image_orig[h, w] = gaussian_at_point(Float64[h, w], psf_center, psf_radius)
end
psf_image_orig = psf_image_orig / sum(psf_image_orig);
psf_image = deepcopy(psf_image_orig);

rendered_image = fill(NaN, size(image));
function render_image!(rendered_image, psf_image, scale)
    rendered_image[:] = NaN
    for pr in pixel_ranges
        row = findfirst(star_catalog[:objid] .== pr.objid)
        object_loc = Array(star_catalog[row, [:pix_h, :pix_w]])[:]
        object_brightness = star_catalog[row, :decam_flux5] * scale

        # Just a guess
        #object_brightness *= im_header["SATURATA"]

        # Another guess
        # object_brightness *= header["MAGZERO"]

        kernel_width = 3.0
        for h in pr.h_range, w in pr.w_range
            if isnan(rendered_image[h, w])
                rendered_image[h, w] = 0.0
            end
        end
        add_interpolation_to_image!(
            x -> cubic_kernel(x, kernel_width),
            Int(kernel_width),
            rendered_image,
            psf_image,
            pr.h_range,
            pr.w_range,
            object_loc,
            object_brightness)
    end
    rendered_image += im_header["AVSKY"];

    return true
end



if false
    render_image!(rendered_image, psf_image, 1000.0)
    PyPlot.close("all")
    plot(star_image[:], rendered_image[:], "k.")
    plot(maximum(star_image), maximum(star_image), "ro")

    PyPlot.close("all")
    matshow(rendered_image); colorbar(); title("rendered")
    matshow(star_image); colorbar(); title("image")

    PyPlot.close("all")
    matshow(star_image - rendered_image); colorbar(); title("residual")
end


#################################
# Autodiff

# Set a lower bound and make sure the psf is greater than the lower bound
# but still normalized.
psf_lb = 1e-10
n = prod(size(psf_image_orig))
psf_image = psf_image_orig * (1 - 2 * psf_lb * n) + 2 * psf_lb;

# Get a permutation that sets the middle of the PSF to be the last index,
# since that's the one that's set to zero in the unconstrained simplex encoding.
psf_center = (Int(floor(0.5 * size(psf_image, 1) + 1)),
              Int(floor(0.5 * size(psf_image, 2) + 1)))
psf_center_ind = sub2ind(size(psf_image),psf_center...)
psf_inds = setdiff(1:length(psf_image), psf_center_ind);
push!(psf_inds, psf_center_ind);

function encode_params(psf_image, scale)
    par = fill(NaN, length(psf_image) + 1)

    offset = 0
    psf_free = unsimplexify_parameter(psf_image[psf_inds], psf_lb, 1.0);
    par[offset + (1:length(psf_free))] = psf_free
    offset += length(psf_free)

    par[offset + 1] = log(scale)

    return par
end

function decode_params(par)
    psf_size = length(par) - 1

    # Check that it's square
    psf_dim = Int(floor(sqrt(psf_size)))
    @assert abs(psf_dim - sqrt(psf_size)) < 1e-16

    offset = 0
    psf_free = par[offset + (1:(psf_size - 1))]
    offset += psf_size - 1

    psf_vec = simplexify_parameter(psf_free, psf_lb, 1.0);
    ipermute!(psf_vec, psf_inds);
    psf_image = reshape(psf_vec, (psf_dim, psf_dim));

    scale = exp(par[offset + 1])

    return psf_image, scale
end


image_diff = similar(star_image);
active_pixel_indices = find(!isnan(star_image));
function objective(psf_image, scale)
    # rendered_image = fill(NaN, size(image));
    rendered_image = similar(psf_image, size(star_image));
    # rendered_image = similar(psf_image, (5, 5));
    rendered_image[:] = NaN
    render_image!(rendered_image, psf_image, scale)
    image_diff = rendered_image - star_image
    result = 0.
    for pix_ind in active_pixel_indices
        result += image_diff[pix_ind] ^ 2
    end
    result /= length(active_pixel_indices)
    println(result)
    return result
end

objective(psf_image, 1000.0)

function objective_wrap(par)
    psf_image, scale = decode_params(par)
    return objective(psf_image, scale)
end


scale = 1000.0

par = encode_params(psf_image, scale);
psf_image_test, scale_test = decode_params(par);
@assert maximum(abs(psf_image - psf_image_test)) < 1e-12
@assert abs(scale - scale_test) < 1e-12
objective_wrap(par);

objective_wrap_tape = GradientTape(objective_wrap, par);
objective_wrap_hess_tape = ReverseDiff.HessianTape(objective_wrap, par);
# compiled_objective_wrap_tape = compile(objective_wrap_tape)

results = (similar(par));
gradient!(results, objective_wrap_tape, par);

# Doesn't work:
# hess_results = similar(par, (length(par), length(par)));
# ReverseDiff.hessian!(hess_results, objective_wrap_hess_tape, par);
# ReverseDiff.hessian(objective_wrap_hess_tape, par);


function objective_grad!(par, results)
    gradient!(results, objective_wrap_tape, par)
end
objective_grad!(par, results);
maximum(abs(results))

epsilon = 1e-8
psf_image_eps, scale_eps = decode_params(par + epsilon * results);
if false
    matshow(psf_image_eps - psf_image); colorbar()
end

optim_res = Optim.optimize(
    objective_wrap, objective_grad!, par, LBFGS(),
    Optim.Options(f_tol=1e-12, iterations=100,
    store_trace = true, show_trace = true))

psf_image_opt, scale_opt = decode_params(optim_res.minimizer);

rendered_image = similar(psf_image_opt, size(star_image));
rendered_image[:] = NaN
render_image!(rendered_image, psf_image_opt, scale_opt);
image_diff = rendered_image - star_image;
if false matshow(image_diff) end


ind = 0

# PyPlot.close("all")
ind = ind + 1
pr = pixel_ranges[ind]




using RCall
R"""
library(ggplot2)
"""

ind = ind + 1
pr = pixel_ranges[ind]

@rput image_diff;
h_range = pr.h_range;
w_range = pr.w_range;
@rput h_range;
@rput w_range;
@rput ind;


R"""
img_melt <- melt(image_diff[h_range, w_range])
ggplot(img_melt, aes(x=Var1, y=Var2, fill=value)) +
    geom_raster() + scale_fill_gradient2(midpoint=0) +
    ggtitle(ind)
"""

#
# R"""
# library(ggplot2)
# library(gridExtra)
# library(reshape2)
#
# grid.arrange(
#     ggplot(melt(raw_image), aes(Var1, Var2, value)) + geom_raster(),
#     ggplot(melt(raw_image), aes(Var1, Var2, value)) + geom_raster(),
#     ncol=2)
# """










#
