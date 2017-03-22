import FITSIO
import WCS
#using PyPlot
using DataFrames
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using Optim

using RCall
R"""
library(ggplot2)
library(gridExtra)
library(reshape2)
library(dplyr)

PlotMatrix <- function(image) {
    img_melt <- melt(image)
    ggplot(img_melt, aes(x=Var1, y=Var2, fill=value)) +
        geom_raster() + scale_fill_gradient2(midpoint=0)
}

PlotRaster <- function(img_melt) {
    ggplot(img_melt, aes(x=h, y=w, fill=val)) +
        geom_raster() + scale_fill_gradient2(midpoint=0)
}
"""

# Make a melted data table
function melt_image(image, pixel_ranges)
    # Just to get it started
    df = DataFrame(h=1, w=1, h0=0, w0=0, objid=Int32(0), val=0.0)
    for pr in pixel_ranges, h in pr.h_range, w in pr.w_range
        h0 = minimum(pr.h_range)
        w0 = minimum(pr.w_range)
        objid = pr.objid
        if (isnan(image[h, w])) continue end
        append!(df, DataFrame(h=h, w=w, h0=h0, w0=w0, objid=objid,
                              val=Float64(image[h, w])))
    end
    df = df[2:end, :]
    return df
end

data_path = joinpath(ENV["GIT_REPO_LOC"], "DECamSurvey.jl", "dat")
src_path = joinpath(ENV["GIT_REPO_LOC"], "DECamSurvey.jl", "src")
include(joinpath(src_path, "objectives.jl"))
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
wcs = WCS.from_header(FITSIO.read_header(f_image[im_ind], String))[1];

bricknames = get_bricknames_for_image(
    joinpath(data_path, "survey-bricks.fits"), im_header);
catalog_vec = [ load_brickname(brickname, wcs, image) for brickname in bricknames ];
catalog = reduce(vcat, catalog_vec);



###################
# Get pixels that we're going to process

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

# Choose stars of a minimum brighntess
star_catalog = catalog[star_rows & (catalog[:decam_flux5] .> 300), :];
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

star_locs = reduce(vcat,
    [ DataFrame(objid=pr.objid, h0=minimum(pr.h_range),
    w0=minimum(pr.w_range)) for pr in pixel_ranges]);
# Show the star locations
if false
    @rput star_locs
    R"""
    ggplot(star_locs) + geom_point(aes(x=h0, y=w0), size=3, color="red")
    """
end

# Make an initial PSF guess
psf_image_orig = fill(0.0, (12, 12));
psf_radius = 1.5
psf_center = 0.5 * Float64[size(psf_image_orig, 1) + 1, size(psf_image_orig, 2) + 1];
for h in 1:size(psf_image_orig, 1), w in 1:size(psf_image_orig, 2)
    psf_image_orig[h, w] = gaussian_at_point(Float64[h, w], psf_center, psf_radius)
end
psf_image_orig = psf_image_orig / sum(psf_image_orig);
psf_image = deepcopy(psf_image_orig);

rendered_image = fill(NaN, size(image));


#################################
# Autodiff

# Set a lower bound and make sure the psf is greater than the lower bound
# but still normalized.
psf_lb = 1e-10
psf_image = enforce_psf_lower_bound(psf_image_orig, psf_lb);
encode_params, decode_params, objective, objective_wrap,  objective_grad! =
    get_single_flux_single_psf_objectives(
        star_image, psf_image, psf_lb, pixel_ranges, star_catalog, im_header);

objective(psf_image, 1000.0)

scale = 617.0
par = encode_params(psf_image, scale);
results = similar(par);
objective_grad!(par, results);

optim_res = Optim.optimize(
    objective_wrap, objective_grad!, par, LBFGS(),
    Optim.Options(f_tol=1e-4, iterations=200,
    store_trace = true, show_trace = true))

psf_image_opt, scale_opt = decode_params(optim_res.minimizer);

@rput psf_image_opt
R"""
PlotMatrix(psf_image_opt)
"""

###########################
# Look at results

rendered_image = similar(psf_image_opt, size(star_image));
rendered_image[:] = NaN
render_image!(rendered_image, psf_image_opt, scale_opt,
              pixel_ranges, star_catalog, im_header);
image_diff = rendered_image - star_image;

ind = 0

ind = ind + 1
pr = pixel_ranges[ind]

@rput image_diff;
h_range = pr.h_range;
w_range = pr.w_range;
@rput h_range;
@rput w_range;
@rput ind;
row = findfirst(star_catalog[:objid] .== pr.objid);
object_brightness = star_catalog[row, :decam_flux5]
object_h = star_catalog[row, :pix_h]
object_w = star_catalog[row, :pix_w]
@rput object_brightness
@rput object_h
@rput object_w
im_size_h = size(image, 1)
im_size_w = size(image, 2)
@rput im_size_h
@rput im_size_w
R"""
grid.arrange(
    PlotMatrix(image_diff[h_range, w_range]) +
        ggtitle(paste(ind, object_brightness, sep=": "))
,
    ggplot() + geom_point(aes(x=object_h, y=object_w), size=2, color="red") +
        xlim(1, im_size_h) + ylim(1, im_size_w),
ncol=2
)
"""


ind = 0

ind = ind + 1
pr = pixel_ranges[ind]
@rput image_diff;
@rput image;
@rput rendered_image;
h_range = pr.h_range;
w_range = pr.w_range;
@rput h_range;
@rput w_range;
@rput ind;
row = findfirst(star_catalog[:objid] .== pr.objid);
object_brightness = star_catalog[row, :decam_flux5]
@rput object_brightness

R"""
grid.arrange(
PlotMatrix(image[h_range, w_range]) +
    ggtitle(paste("Original image", ind, object_brightness, sep=": "))
,
PlotMatrix(rendered_image[h_range, w_range]) +
    ggtitle(paste("Rendered", ind, object_brightness, sep=": "))
,
PlotMatrix(image_diff[h_range, w_range]) +
    ggtitle(paste("Residual", ind, object_brightness, sep=": "))
, ncol=3)
"""


image_melt = melt_image(image, pixel_ranges);
@rput image_melt;

image_diff_melt = melt_image(image_diff, pixel_ranges);
@rput image_diff_melt;

objids = [pr.objid for pr in pixel_ranges ];
h0 = [minimum(pr.h_range) for pr in pixel_ranges ];
w0 = [minimum(pr.w_range) for pr in pixel_ranges ];
w_perm = sortperm(w0)

ind = 0

ind = ind + 1
this_objid = objids[w_perm[ind]]
@rput this_objid

R"""
PlotRaster(filter(image_diff_melt, objid==this_objid)) +
    ggtitle(sprintf("objid=%d", this_objid))
"""

###########
# Try finite differences

using Calculus
function objective_grad(par)
    results = similar(par)
    gradient!(results, objective_wrap_tape, par)
    return results
end

hess = jacobian(objective_grad, optim_res.minimizer, :central);





#
