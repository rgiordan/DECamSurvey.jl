import FITSIO
import WCS
using PyPlot
using DataFrames

data_path = joinpath(ENV["GIT_REPO_LOC"], "DECamSurvey.jl", "dat")
src_path = joinpath(ENV["GIT_REPO_LOC"], "DECamSurvey.jl", "src")
include(joinpath(src_path, "psf_lib.jl"))

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

# Get the corners, confirm that WCS matches up with them.
corners = [ [ im_header["COR$(i)RA1"], im_header["COR$(i)DEC1"]] for i in 1:4 ]
center = [ im_header["CENRA1"], im_header["CENDEC1"]];
wcs = WCS.from_header(im_header_string)[1];
WCS.world_to_pix(wcs, corners[1])
WCS.world_to_pix(wcs, corners[2])
WCS.world_to_pix(wcs, corners[3])
WCS.world_to_pix(wcs, corners[4])
WCS.world_to_pix(wcs, center)
size(image)


# arcmin
[x for x in asec_per_pixel] .* [x for x in size(image)] / 60

function get_trimmed_image(image; trim_quantile=0.9999, num_sd=2)
    trim_level = quantile(image[ !isnan(image[:]) ][:], trim_quantile)
    noise_level =
        median(image[ !isnan(image[:]) ][:]) + num_sd * std(image[!isnan(image)])
    image_trim = deepcopy(image);
    image_trim[image_trim .> trim_level] = trim_level;
    image_trim[image_trim .< noise_level] = 0
    return image_trim
end

# Note that relative to the legacy viewer
# http://legacysurvey.org
# this seems to be flipped left-to-right.  Also, the legacy viewer is in some
# kind of very sensitive (logarithmic or more?) brightness scale.
# matshow(get_trimmed_image(image, trim_quantile=1, num_sd=-Inf)); colorbar()
obj_loc = WCS.pix_to_world(wcs, Float64[1398, 683])

# Get the brick ids here
# http://legacysurvey.org/dr3/files/
# It appears that bricks are RA, DEC rectangles.
# survey-bricks.fits.gz
f_bricks = FITSIO.FITS(joinpath(data_path, "survey-bricks.fits"))
FITSIO.read_header(f_bricks[2])
f_bricks[2]
bricks = DataFrame(
    brickname=FITSIO.read(f_bricks[2], "brickname"),
    brickid=FITSIO.read(f_bricks[2], "brickid"),
    dec1=FITSIO.read(f_bricks[2], "dec1"),
    dec2=FITSIO.read(f_bricks[2], "dec2"),
    ra1=FITSIO.read(f_bricks[2], "ra1"),
    ra2=FITSIO.read(f_bricks[2], "ra2"))

brick_row = get_ra_dec_brick(obj_loc[1], obj_loc[2], bricks)

# Load the catalogs with the brick ids
# http://legacysurvey.org/dr3/catalogs/
# │ Row │ brickname  │ brickid │ dec1   │ dec2   │ ra1     │ ra2     │
# ├─────┼────────────┼─────────┼────────┼────────┼─────────┼─────────┤
# │ 1   │ "1984p110" │ 394182  │ 10.875 │ 11.125 │ 198.305 │ 198.559 │

f_tractor = FITSIO.FITS(joinpath(data_path, "tractor-1984p110.fits"));
FITSIO.read_header(f_tractor[2])

catalog = load_catalog(f_tractor, wcs, size(image));

bright_rows = catalog[:decam_flux5] .> 500;
star_rows = catalog[:type] .== "PSF";

# look at galaxies where we don't know whether they're exp or dev
catalog[ (catalog[:type] .!= "PSF") & (0 .< catalog[:fracdev] .< 1),
         [:type, :ra, :dec, :decam_flux5, :fracdev, :shapeexp_r, :shapedev_r]]

# look at galaxies
sort!(catalog, cols = [:shapeexp_r])
catalog[ !star_rows,
         [:type, :ra, :dec, :pix_h, :pix_w, :decam_flux5, :fracdev, :shapeexp_r, :shapedev_r]]

function RowsCloseTo(ra, dec, catalog, pix_radius)
    pix_loc = WCS.world_to_pix(wcs, [ra, dec])
    return (
        (pix_loc[1] - pix_radius .< catalog[:pix_h] .< pix_loc[1] + pix_radius) &
        (pix_loc[2] - pix_radius .< catalog[:pix_w] .< pix_loc[2] + pix_radius))
end

row_num = 346
world_loc = Array(catalog[!star_rows, :][row_num, [:ra, :dec]])[:]
ra, dec = world_loc[1], world_loc[2]
catalog[RowsCloseTo(ra, dec, catalog, 10), :]
pix_loc = WCS.world_to_pix(wcs, [ra, dec])
hrange, wrange = get_pixel_box(ra, dec, 40, wcs, size(image))
matshow(image[hrange, wrange])
plot(pix_loc[1] - minimum(hrange) + 1, pix_loc[2] - minimum(wrange) + 1, "bo")


hrange, wrange = get_pixel_box(ra, dec, 40, wcs, size(image))
matshow(image[hrange, wrange])

# I'm concerned wither pix_loc is in 0-indexed or 1-indexed coordinates.
# It appears to be 1-indexed.

matshow(log(get_trimmed_image(image, trim_quantile=1, num_sd=-Inf))); colorbar()
plot(pix_loc[2, image_rows & bright_rows] - 1, pix_loc[1, bright_rows] - 1, "ro")

# Gaia.  What is the meaning of the chunk numbers?
f_gaia = FITSIO.FITS(joinpath(data_path, "chunk-00001.fits"));
gaia_ra = FITSIO.read(f_gaia[2], "ra");
gaia_dec = FITSIO.read(f_gaia[2], "dec");
plot(gaia_ra, gaia_dec, "bo")

PyPlot.close("all")

# First pass

# NaN out pixels that are near galaxies.
gal_catalog = catalog[!star_rows, :];
filter_image = deepcopy(image);
for row in 1:nrow(gal_catalog)
    ra = gal_catalog[row, :ra]
    dec = gal_catalog[row, :dec]
    radius = Int(ceil(8 * max(gal_catalog[row, :shapeexp_r], gal_catalog[row, :shapedev_r])))
    radius = max(radius, 3)
    h_range, w_range = get_pixel_box(ra, dec, radius, wcs, size(image))
    filter_image[h_range, w_range] = NaN
end

# Define star ranges.
type PixelRange
    objid::Int32
    h_range::UnitRange{Int64}
    w_range::UnitRange{Int64}
end

star_catalog = catalog[star_rows & (catalog[:decam_flux5] .> 50), :];
pixel_ranges = PixelRange[]
for row in 1:nrow(star_catalog)
    ra = star_catalog[row, :ra]
    dec = star_catalog[row, :dec]
    objid = star_catalog[row, :objid]
    h_range, w_range = get_pixel_box(ra, dec, 20, wcs, size(image))
    push!(pixel_ranges, PixelRange(objid, h_range, w_range))
end

star_image = fill(NaN, size(image));
for pr in pixel_ranges
    star_image[pr.h_range, pr.w_range] = image[pr.h_range, pr.w_range]
end

function render_star_at_point(pix_loc::Vector{Float64}, pix_center::Vector{Float64})
    radius_pix = 5.0
    r = pix_loc - pix_center
    return exp(-0.5 * dot(r, r) / (radius_pix ^ 2))
end
render_star_at_point([5., 6.], [3., 2.])

rendered_image = fill(NaN, size(image));
for pr in pixel_ranges
    row = findfirst(star_catalog[:objid] .== pr.objid)
    pix_center = Array(star_catalog[row, [:pix_h, :pix_w]])[:]
    for h in pr.h_range, w in pr.w_range
        if (isnan(star_image[h, w])) continue end
        if (isnan(rendered_image[h, w])) rendered_image[h, w] = 0. end
        rendered_image[h, w] += render_star_at_point(Float64[h, w], pix_center)
    end
end

matshow(star_image - rendered_image)
