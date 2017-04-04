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

k = 50
A = rand(k, k)
A = 0.5 * (A + A') + eye(k)
function f(x)
    0.5 * dot(x, A * x)
end

x = rand(k) # my input value
v = rand(k) # the `v` in `H * v`


hess_p = get_hess_p_function(f, x)

f_hess = ReverseDiff.hessian(f, x);
f_hess * v

hess_p(x, v)


############
# Conjugate gradient

b = copy(v)
x0 = rand(k)

# Get hess^{-1} v starting at x0
function mult_a(v)
    hess_p(x, v)
end

function conjugate_gradient(mult_a, b, x0, tol=1e-6)
    r0 = mult_a(x0) - b
    p0 = copy(-r0)
    tol = 1e-6

    ind = 0
    x1 = copy(x0)
    while sqrt(dot(r0, r0)) > tol
        alpha = dot(r0, r0) / dot(p0, mult_a(p0))
        x1 = x0 + alpha * p0
        r1 = r0 + alpha * mult_a(p0)
        beta = dot(r1, r1) / dot(r0, r0)
        p1 = -r1 + beta * p0
        r0 = copy(r1)
        p0 = copy(p1)
        x0 = copy(x1)
        ind = ind + 1
        println(ind, " ", sqrt(dot(r0, r0)))
    end

    return x1
end

soln = conjugate_gradient(mult_a, b, x0);
maximum(abs(soln - (f_hess \ b)))
maximum(abs((A \ b) - soln))




    #
