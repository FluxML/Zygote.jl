using LogExpFunctions: xlogx, xlogy, logistic, log1pexp, logsumexp, logaddexp, logsubexp
using Base.Broadcast: broadcasted

# The scalar rules are `rrule`s, as an `@adjoint` would unthunk a deferred cotangent (#1741).

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(xlogx), x::Real)
    result, dx = ∇xlogx(x)
    xlogx_pullback(δ) = (NoTangent(), dx * δ)
    return result, xlogx_pullback
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

for f in (:xlogx, :log1pexp)  # see the `AbstractFill` methods in broadcast.jl
    @eval @adjoint broadcasted(::typeof($f), x::AbstractFill{<:Real}) = _broadcast_fill(__context__, $f, x)
end

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(xlogy), x::Real, y::Real)
    result, dx, dy = ∇xlogy(x, y)
    xlogy_pullback(δ) = (NoTangent(), δ * dx, δ * dy)
    return result, xlogy_pullback
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

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(logaddexp), x::Real, y::Real)
    result, dx, dy = ∇logaddexp(x, y)
    logaddexp_pullback(δ) = (NoTangent(), δ * dx, δ * dy)
    return result, logaddexp_pullback
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

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(logsubexp), x::Real, y::Real)
    result, dx, dy = ∇logsubexp(x, y)
    logsubexp_pullback(δ) = (NoTangent(), δ * dx, δ * dy)
    return result, logsubexp_pullback
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

