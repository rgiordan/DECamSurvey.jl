import FITSIO
import WCS
using PyPlot

data_path = joinpath(ENV["GIT_REPO_LOC"], "DECamSurvey.jl", "dat")

fname_head = "c4d_160302_094418_oo"
fname_tail = "_z_v1.fits"
join([ fname_head, "i", fname_tail ])

f_image = FITSIO.FITS(joinpath(data_path, join([ fname_head, "i", fname_tail ])));
f_weight = FITSIO.FITS(joinpath(data_path, join([ fname_head, "w", fname_tail ])));
f_mask = FITSIO.FITS(joinpath(data_path, join([ fname_head, "d", fname_tail ])));

header = FITSIO.read_header(f_image[1]);
asec_per_pixel = (header["PIXSCAL1"], header["PIXSCAL2"])

im_ind = 4;

image = FITSIO.read(f_image[im_ind]);
weight = FITSIO.read(f_weight[im_ind]);
mask = FITSIO.read(f_mask[im_ind]);

image[mask .== 1] = NaN

im_header = FITSIO.read_header(f_image[im_ind]);
im_header_string = FITSIO.read_header(f_image[im_ind], String);

# Get the corners, confirm that WCS matches up with them.
corners = [ [ im_header["COR$(i)RA1"], im_header["COR$(i)DEC1"]] for i in 1:4 ]
wcs = WCS.from_header(im_header_string)[1]
WCS.pix_to_world(wcs, Float64[1,1])
WCS.pix_to_world(wcs, Float64[x for x in size(image)])
corners

# arcmin
[x for x in asec_per_pixel] .* [x for x in size(image)] / 60

function get_trimmed_image(image; trim_quantile=0.9999, num_sd=2)
    trim_level = quantile(image[ !isnan(image[:]) ][:], trim_quantile)
    noise_level = median(image[ !isnan(image[:]) ][:]) + num_sd * std(image[!isnan(image)])
    image_trim = deepcopy(image);
    image_trim[image_trim .> trim_level] = trim_level;
    image_trim[image_trim .< noise_level] = 0
    return image_trim
end

# Note that relative to the legacy viewer
# http://legacysurvey.org
# this seems to be flipped left-to-right.  Also, the legacy viewer is in some
# kind of very sensitive (logarithmic or more?) brightness scale.
matshow(get_trimmed_image(image, trim_quantile=1, num_sd=-Inf)); colorbar()
WCS.pix_to_world(wcs, Float64[1398, 683])

