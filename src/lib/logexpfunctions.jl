using LogExpFunctions: xlogx, xlogy, logistic, log1pexp, logsumexp, logaddexp, logsubexp
using Base.Broadcast: broadcasted

@adjoint function xlogx(x::Real)
    result, dx = ∇xlogx(x)
    back(δ) = (dx * δ,)
    return result, back
end
@adjoint function broadcasted(::typeof(xlogx), x::Numeric)
    result, dx = ∇xlogx(x)
    back(δ) = (nothing, unbroadcast(x, δ .* dx))
    return result, back
end
function ∇xlogx(x::Numeric)
    logx = log.(x)
    xlogx = x .* logx
    result = @. ifelse(iszero(x), zero(xlogx), xlogx)
    dx = @. one(x) + logx
    return result, dx
end

@adjoint function broadcasted(::typeof(log1pexp), x::Numeric)
    dx = logistic.(x)
    return log1pexp.(x), δ -> (nothing, unbroadcast(x, δ .* dx))
end

@adjoint function xlogy(x::Real, y::Real)
    result, dx, dy = ∇xlogy(x, y)
    back(δ) = (δ * dx, δ * dy)
    return result, back
end
@adjoint function broadcasted(::typeof(xlogy), x::Numeric, y::Numeric)
    result, dx, dy = ∇xlogy(x, y)
    back(δ) = (nothing, unbroadcast(x, δ .* dx), unbroadcast(y, δ .* dy))
    return result, back
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
@adjoint function broadcasted(::typeof(logaddexp), x::Numeric, y::Numeric)
    result, dx, dy = ∇logaddexp(x, y)
    back(δ) = (nothing, unbroadcast(x, δ .* dx), unbroadcast(y, δ .* dy))
    return result, back
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
@adjoint function broadcasted(::typeof(logsubexp), x::Numeric, y::Numeric)
    result, dx, dy = ∇logsubexp(x, y)
    back(δ) = (nothing, unbroadcast(x, δ .* dx), unbroadcast(y, δ .* dy))
    return result, back
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

