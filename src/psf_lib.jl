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
