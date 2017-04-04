

k = 10
H = rand(k, k)
H = 0.5 * (H + H') + eye(k)
g = rand(k)

function mult_h(v)
    return H * v
end

tj = g
qjm1 = fill(0.0, length(g))

iters = 5
Q = fill(NaN, k, iters)
gamma_vec = fill(0.0, iters)
delta_vec = fill(0.0, iters)

for j in 1:iters
    gamma = sqrt(dot(tj, tj))
    qj = tj / gamma
    delta = dot(qj, mult_h(qj))
    tjp1 = mult_h(qj) - delta * qj - gamma * qjm1

    gamma_vec[j] = gamma
    Q[:, j] = qj
    delta_vec[j] = delta

    qjm1 = copy(qj)
    tj = copy(tjp1)
end

Tmat = SymTridiagonal(delta_vec, gamma_vec[2:end])

Q' * H * Q - Tmat













#
