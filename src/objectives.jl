###################################
# Objective functions

using ReverseDiff, ForwardDiff

# Thanks to Jarrett for this function.  Returns a function that evaluates
# hessian(x) * v using a mixture of forward and reverse diff.
function get_hess_vec_prod_function(f, x)
    # A vector of single-perturbation dual numbers

    x_dual_v = similar(x, ForwardDiff.Dual{1, eltype(x)})
    f_grad_tape = ReverseDiff.GradientTape(f, x_dual_v);

    # hess(x) * v
    function hess_vec_prod(x, v)
        result = similar(x, ForwardDiff.Dual{1, eltype(x)})

        # use the `x` entries for the primal values, and `v` entries as the
        # perturbation coefficients
        for i in eachindex(x_dual_v)
            x_dual_v[i] = ForwardDiff.Dual(x[i], v[i])
        end

        # returns a gradient of Duals where the perturbation coefficients
        # are the result of the Hessian-vector product
        hess_v_prod_duals = deepcopy(x_dual_v)
        ReverseDiff.gradient!(hess_v_prod_duals, f_grad_tape, x_dual_v);

        # ReverseDiff.gradient(f, x)
        # hess_v_prod_duals = ReverseDiff.gradient(f, x_dual_v);
        return Float64[ d.partials[1] for d in hess_v_prod_duals ]
    end

    return hess_vec_prod
end



# A objective as a function of a single PSF matrix and common scale.
function get_single_psf_objectives(
    star_image, psf_image, psf_lb, pixel_ranges, star_catalog, im_header, scale)

    # Get a permutation that sets the middle of the PSF to be the last index,
    # since that's the one that's set to zero in the unconstrained simplex encoding.
    psf_center = (Int(floor(0.5 * size(psf_image, 1) + 1)),
                  Int(floor(0.5 * size(psf_image, 2) + 1)))
    psf_center_ind = sub2ind(size(psf_image), psf_center...)
    psf_inds = setdiff(1:length(psf_image), psf_center_ind);
    push!(psf_inds, psf_center_ind);

    function encode_params(psf_image)
        par = fill(NaN, length(psf_image) - 1)

        offset = 0
        psf_free = unsimplexify_parameter(psf_image[psf_inds], psf_lb, 1.0);
        par[offset + (1:length(psf_free))] = psf_free
        offset += length(psf_free)

        return par
    end

    function decode_params(par)
        psf_size = length(par) + 1

        # Check that it's square
        psf_dim = Int(floor(sqrt(psf_size)))
        @assert abs(psf_dim - sqrt(psf_size)) < 1e-16

        offset = 0
        psf_free = par[offset + (1:(psf_size - 1))]
        offset += psf_size - 1

        psf_vec = simplexify_parameter(psf_free, psf_lb, 1.0);
        ipermute!(psf_vec, psf_inds);
        psf_image = reshape(psf_vec, (psf_dim, psf_dim));

        return psf_image
    end

    image_diff = similar(star_image);
    active_pixel_indices = find(!isnan(star_image));
    function objective(psf_image)
        rendered_image = similar(psf_image, size(star_image));
        rendered_image[:] = NaN
        render_catalog_image!(rendered_image, psf_image, scale,
                              pixel_ranges, star_catalog, im_header)
        image_diff = rendered_image - star_image
        result = 0.
        for pix_ind in active_pixel_indices
            result += image_diff[pix_ind] ^ 2
        end
        result /= length(active_pixel_indices)
        println(result)
        return result
    end


    function objective_wrap(par)
        psf_image = decode_params(par)
        return objective(psf_image)
    end


    par = encode_params(psf_image);
    psf_image_test = decode_params(par);
    @assert maximum(abs(psf_image - psf_image_test)) < 1e-12

    print("Building gradient tape...")
    objective_wrap_tape = GradientTape(objective_wrap, par);
    println("done.")
    function objective_grad!(par, results)
        gradient!(results, objective_wrap_tape, par)
    end

    objective_hess_vec_prod = get_hess_vec_prod_function(objective_wrap, par)

    return encode_params, decode_params, objective, objective_wrap,
        objective_grad!, objective_hess_vec_prod
end



# A objective as a function of a single PSF matrix and common scale.
function get_single_flux_single_psf_objectives(
    star_image, psf_image, psf_lb, pixel_ranges, star_catalog, im_header)

    # Get a permutation that sets the middle of the PSF to be the last index,
    # since that's the one that's set to zero in the unconstrained simplex encoding.
    psf_center = (Int(floor(0.5 * size(psf_image, 1) + 1)),
                  Int(floor(0.5 * size(psf_image, 2) + 1)))
    psf_center_ind = sub2ind(size(psf_image), psf_center...)
    psf_inds = setdiff(1:length(psf_image), psf_center_ind);
    push!(psf_inds, psf_center_ind);

    function encode_params(psf_image, scale)
        par = fill(NaN, length(psf_image) -1 + 1)

        offset = 0
        psf_free = unsimplexify_parameter(psf_image[psf_inds], psf_lb, 1.0);
        par[offset + (1:length(psf_free))] = psf_free
        offset += length(psf_free)

        par[offset + 1] = log(scale)

        return par
    end

    function decode_params(par)
        psf_size = length(par) - 1 + 1

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
        rendered_image = similar(psf_image, size(star_image));
        rendered_image[:] = NaN
        render_catalog_image!(rendered_image, psf_image, scale,
                              pixel_ranges, star_catalog, im_header)
        image_diff = rendered_image - star_image
        result = 0.
        for pix_ind in active_pixel_indices
            result += image_diff[pix_ind] ^ 2
        end
        result /= length(active_pixel_indices)
        println(result)
        return result
    end


    function objective_wrap(par)
        psf_image, scale = decode_params(par)
        return objective(psf_image, scale)
    end


    scale = 1000.0
    par = encode_params(psf_image, scale);
    psf_image_test, scale_test = decode_params(par);
    @assert maximum(abs(psf_image - psf_image_test)) < 1e-12
    @assert abs(scale - scale_test) < 1e-12

    print("Building gradient tape...")
    objective_wrap_tape = GradientTape(objective_wrap, par);
    println("done.")
    function objective_grad!(par, results)
        gradient!(results, objective_wrap_tape, par)
    end

    # Doesn't work:
    # objective_wrap_hess_tape = ReverseDiff.HessianTape(objective_wrap, par);
    # compiled_objective_wrap_tape = compile(objective_wrap_tape)
    # hess_results = similar(par, (length(par), length(par)));
    # ReverseDiff.hessian!(hess_results, objective_wrap_hess_tape, par);
    # ReverseDiff.hessian(objective_wrap_hess_tape, par);

    return encode_params, decode_params, objective, objective_wrap, objective_grad!
end


# A objective as a function of a single PSF matrix and common scale.
function get_multi_flux_single_psf_objectives(
    star_image, psf_image, psf_lb, num_stars, pixel_ranges, star_catalog, im_header)

    # Get a permutation that sets the middle of the PSF to be the last index,
    # since that's the one that's set to zero in the unconstrained simplex encoding.
    psf_center = (Int(floor(0.5 * size(psf_image, 1) + 1)),
                  Int(floor(0.5 * size(psf_image, 2) + 1)))
    psf_center_ind = sub2ind(size(psf_image), psf_center...)
    psf_inds = setdiff(1:length(psf_image), psf_center_ind);
    push!(psf_inds, psf_center_ind);

    function encode_params(psf_image, scales)
        par = fill(NaN, length(psf_image) -1 + length(scales))

        offset = 0
        psf_free = unsimplexify_parameter(psf_image[psf_inds], psf_lb, 1.0);
        par[offset + (1:length(psf_free))] = psf_free
        offset += length(psf_free)

        par[(offset + 1):end] = log(scale)

        return par
    end

    function decode_params(par)
        psf_size = length(par) - length(scales) + 1

        # Check that the PSF is square
        psf_dim = Int(floor(sqrt(psf_size)))
        @assert abs(psf_dim - sqrt(psf_size)) < 1e-16

        offset = 0
        psf_free = par[offset + (1:(psf_size - 1))]
        offset += psf_size - 1

        psf_vec = simplexify_parameter(psf_free, psf_lb, 1.0);
        ipermute!(psf_vec, psf_inds);
        psf_image = reshape(psf_vec, (psf_dim, psf_dim));

        scales = exp(par[(offset + 1):end])

        return psf_image, scales
    end

    image_diff = similar(star_image);
    active_pixel_indices = find(!isnan(star_image));
    function objective(psf_image, scales)
        rendered_image = similar(psf_image, size(star_image));
        rendered_image[:] = NaN
        @assert length(scales) == length(pixel_ranges)

        object_brightnesses = []
        object_locs = fill(NaN, length(pixel_ranges), 2)
        for ind in 1:length(pixel_ranges)
            pr = pixel_ranges[ind]
            row = findfirst(star_catalog[:objid] .== pr.objid)
            object_loc = Array(star_catalog[row, [:pix_h, :pix_w]])[:]
            object_brightness = star_catalog[row, :decam_flux5] * scales[ind]
            append!(object_brightnesses, object_brightness)
            object_locs[ind, :] = object_loc
        end

        render_image!(rendered_image, psf_image, pixel_ranges,
                       obj_locs, object_brightnesses, im_header["AVSKY"])

        image_diff = rendered_image - star_image
        result = 0.
        for pix_ind in active_pixel_indices
            result += image_diff[pix_ind] ^ 2
        end
        result /= length(active_pixel_indices)
        println(result)
        return result
    end


    function objective_wrap(par)
        psf_image, scales = decode_params(par)
        return objective(psf_image, scales)
    end


    par = encode_params(psf_image, scales);
    psf_image_test, scales_test = decode_params(par);
    @assert maximum(abs(psf_image - psf_image_test)) < 1e-12
    @assert abs(scales - scales_test) < 1e-12

    print("Building gradient tape...")
    objective_wrap_tape = GradientTape(objective_wrap, par);
    println("done.")
    function objective_grad!(par, results)
        gradient!(results, objective_wrap_tape, par)
    end

    # Doesn't work:
    # objective_wrap_hess_tape = ReverseDiff.HessianTape(objective_wrap, par);
    # compiled_objective_wrap_tape = compile(objective_wrap_tape)
    # hess_results = similar(par, (length(par), length(par)));
    # ReverseDiff.hessian!(hess_results, objective_wrap_hess_tape, par);
    # ReverseDiff.hessian(objective_wrap_hess_tape, par);

    return encode_params, decode_params, objective, objective_wrap, objective_grad!
end
