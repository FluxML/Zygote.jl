import .StatsFuns
using .StatsFuns: xlogx, xlogy, logistic, logit, log1psq, log1pexp,
    logsumexp, logaddexp, logsubexp
using Base.Broadcast: broadcasted

@adjoint function xlogx(x::Real)
    result, dx = ∇xlogx(x)
    back(δ) = (dx * δ,)
    return result, back
end
function _pullback(__context__::AContext, ::typeof(broadcasted), ::typeof(xlogx), x::Numeric)
    result, dx = ∇xlogx(x)
    _back(::Union{Nothing,AbstractZero}) = Zero()
    _back(Δ) = (DoesNotExist(), DoesNotExist(), unbroadcast(x, Δ .* dx))
    return result, _back
end
function ∇xlogx(x::Numeric)
    logx = log.(x)
    xlogx = x .* logx
    result = @. ifelse(iszero(x), zero(xlogx), xlogx)
    dx = @. one(x) + logx
    return result, dx
end

@adjoint function logistic(x::Real)
    y = logistic(x)
    return y, Δ->(Δ * y * (1 - y),)
end

@adjoint logit(x::Real) = logit(x), Δ->(Δ / (x * (1 - x)),)

@adjoint log1psq(x::Real) = log1psq(x), Δ->(Δ * 2x / (1 + abs2(x)),)

@adjoint function log1pexp(x::Real)
    dx = ∂log1pexp(x)
    return log1pexp(x), δ -> (δ * dx,)
end
function _pullback(__context__::AContext, ::typeof(broadcasted), ::typeof(log1pexp), x::Numeric)
    dx = ∂log1pexp.(x)
    _back(::Union{Nothing,AbstractZero}) = Zero()
    _back(Δ) = (DoesNotExist(), DoesNotExist(), unbroadcast(x, Δ .* dx))
    return log1pexp.(x), _back
end
∂log1pexp(x::Real)    = x < 18.0 ? logistic(x) : x < 33.3 ? one(x) - exp(-x) : oftype(exp(x), 1)
∂log1pexp(x::Float32) = x < 9f0  ? logistic(x) : x < 16f0 ? one(x) - exp(-x) : oftype(exp(x), 1)

@adjoint function logsumexp(X::AbstractArray{<:Real}; dims=:)
    lse = logsumexp(X; dims=dims)
    return lse, Δ -> (Δ .* exp.(X .- lse),)
end

@adjoint function xlogy(x::Real, y::Real)
    result, dx, dy = ∇xlogy(x, y)
    back(δ) = (δ * dx, δ * dy)
    return result, back
end
function _pullback(__context__::AContext, ::typeof(broadcasted), ::typeof(xlogy), x::Numeric, y::Numeric)
    result, dx, dy = ∇xlogy(x, y)
    _back(::Union{Nothing,AbstractZero}) = Zero()
    _back(Δ) = (DoesNotExist(), DoesNotExist(), unbroadcast(x, Δ .* dx), unbroadcast(y, Δ .* dy))
    return result, _back
end
function ∇xlogy(x::Numeric, y::Numeric)
    dx = logy = log.(y)
    dy = x ./ y
    xlogy = x .* logy
    result = @. ifelse(iszero(x) & !isnan(y), zero(xlogy), xlogy)
    return result, dx, dy
end

@adjoint function logaddexp(x::Real, y::Real)
    result, dx, dy = ∇logaddexp(x, y)
    back(δ) = (δ * dx, δ * dy)
    return result, back
end
function _pullback(__context__::AContext, ::typeof(broadcasted), ::typeof(logaddexp), x::Numeric, y::Numeric)
    result, dx, dy = ∇logaddexp(x, y)
    _back(::Union{Nothing,AbstractZero}) = Zero()
    _back(Δ) = (DoesNotExist(), DoesNotExist(), unbroadcast(x, Δ .* dx), unbroadcast(y, Δ .* dy))
    return result, _back
end
function ∇logaddexp(x::Numeric, y::Numeric)
    result = logaddexp.(x, y)
    t = @. exp(-abs(x - y))
    dx, dy = select(x .≥ y, inv.(one.(t) .+ t), t ./ (one.(t) .+ t))
    return result, dx, dy
end

@adjoint function logsubexp(x::Real, y::Real)
    result, dx, dy = ∇logsubexp(x, y)
    back(δ) = (δ * dx, δ * dy)
    return result, back
end
function _pullback(__context__::AContext, ::typeof(broadcasted), ::typeof(logsubexp), x::Numeric, y::Numeric)
    result, dx, dy = ∇logsubexp(x, y)
    _back(::Union{Nothing,AbstractZero}) = Zero()
    _back(Δ) = (DoesNotExist(), DoesNotExist(), unbroadcast(x, Δ .* dx), unbroadcast(y, Δ .* dy))
    return result, _back
end
function ∇logsubexp(x::Numeric, y::Numeric)
    result = logsubexp.(x, y)
    t = @. -inv(expm1(-abs(x - y)))
    dx, dy = select(x .≥ y, t, one.(t) .- t)
    return result, dx, dy
end

"""
	select(cond, x, y)

The call `a, b = select(cond, x, y)` constructs two arrays `a, b`, where
`a[i], b[i] = x[i], y[i]` if `cond[i]` is `true`, and `a[i], b[i] = y[i], x[i]`
 if `cond[i]` is `false`.
"""
select(cond, x, y) = ifelse.(cond, x, y), ifelse.(cond, y, x)

