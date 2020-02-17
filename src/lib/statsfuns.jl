import .StatsFuns
using .StatsFuns: xlogx, logistic, logit, log1psq, log1pexp, logsumexp

@adjoint function xlogx(x::Real)
    y = xlogx(x)
    return y, function(Δ::Real)
        return (x > zero(x) ? Δ * (log(x) + one(y)) : zero(y),)
    end
end

@adjoint function logistic(x::Real)
    y = logistic(x)
    return y, Δ->(Δ * y * (1 - y),)
end

@adjoint logit(x::Real) = logit(x), Δ->(Δ / (x * (1 - x)),)

@adjoint log1psq(x::Real) = log1psq(x), Δ->(Δ * 2x / (1 + abs2(x)),)

@adjoint function log1pexp(x::Float64)
    return log1pexp(x), Δ->(Δ * (x < 18.0 ? logistic(x) : x < 33.3 ? 1 - exp(-x) : 1),)
end

@adjoint function log1pexp(x::Float32)
    return log1pexp(x), Δ->(Δ * (x < 9f0 ? logistic(x) : x < 16f0 ? 1 - exp(-x) : 1),)
end

@adjoint function logsumexp(X::AbstractArray{<:Real}; dims=:)
    lse = logsumexp(X; dims=dims)
    return lse, Δ -> (Δ .* exp.(X .- lse),)
end
