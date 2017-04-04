using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile

function encode_params(psf_matrix, scale)
    par = fill(NaN, length(psf_matrix) + 1)

    offset = 0
    par[offset + (1:length(psf_matrix))] = psf_matrix[:]
    offset += length(psf_matrix)

    par[offset + 1] = log(scale)

    return par
end


function decode_params(par)
    psf_size = length(par) - 1

    # Check that it's square
    psf_dim = Int(floor(sqrt(psf_size)))
    @assert abs(psf_dim - sqrt(psf_size)) < 1e-16

    offset = 0
    psf_matrix = reshape(par[offset + (1:psf_size)], psf_dim, psf_dim)
    offset += psf_size

    scale = exp(par[offset + 1])

    return psf_matrix, scale
end


function objective(psf_matrix, scale)
    return scale * sum(psf_matrix)
end


function objective_wrap(par)
    psf_matrix, scale = decode_params(par)
    return objective(psf_matrix, scale)
end


psf_matrix = rand(5, 5);
scale = 4.0

par = encode_params(psf_matrix, scale)
decode_params(par)
objective_wrap(par)

const objective_wrap_tape = GradientTape(objective_wrap, par)
const compiled_objective_wrap_tape = compile(objective_wrap_tape)

results = (similar(par))
gradient!(results, compiled_objective_wrap_tape, par)


###################################
# Hessian vector products

using ReverseDiff, ForwardDiff

function get_hess_p_function(f, x)
    # A vector of single-perturbation dual numbers

    x_dual_v = similar(x, ForwardDiff.Dual{1, eltype(x)})
    f_grad_tape = ReverseDiff.GradientTape(f, x_dual_v);

    # hess(x) * v
    function hess_p(x, v)
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

    return hess_p
end

function f(x)
    sum(x .* x)
end

x = rand(4) # my input value
v = rand(4) # the `v` in `H * v`


hess_p = get_hess_p_function(f, x)

f_hess = ReverseDiff.hessian(f, x);
f_hess * v

hess_p(x, v)

















#
