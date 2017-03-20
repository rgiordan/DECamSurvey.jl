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
