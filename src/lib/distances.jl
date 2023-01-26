using .Distances
import .ChainRules: NoTangent, rrule, rrule_via_ad

function rrule(::ZygoteRuleConfig, ::SqEuclidean, x::AbstractVector, y::AbstractVector)
  δ = x .- y
  function sqeuclidean_rrule(Δ::Real)
    x̄ = (2 * Δ) .* δ
    return NoTangent(), x̄, -x̄
  end
  return sum(abs2, δ), sqeuclidean_rrule
end

function rrule(::ZygoteRuleConfig, ::typeof(colwise), s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
  function colwise_SqEuclidean_rrule(Δ::AbstractVector)
    x̄ = 2 .* Δ' .* (x .- y)
    return NoTangent(), NoTangent(), x̄, -x̄
  end
  return colwise(s, x, y), colwise_SqEuclidean_rrule
end

function rrule(::ZygoteRuleConfig, ::typeof(pairwise), s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix; dims::Int=2)
  if dims==1
    return pairwise(s, x, y; dims=1), ∇pairwise(s, transpose(x), transpose(y), transpose)
  else
    return pairwise(s, x, y; dims=dims), ∇pairwise(s, x, y, identity)
  end
end

∇pairwise(s, x, y, f) =
  function pairwise_sqeuclidean_rrule(Δ)
    x̄ = 2 .* (x * Diagonal(vec(sum(Δ; dims=2))) .- y * transpose(Δ))
    ȳ = 2 .* (y * Diagonal(vec(sum(Δ; dims=1))) .- x * Δ)
    return NoTangent(), NoTangent(), f(x̄), f(ȳ)
  end

function rrule(::ZygoteRuleConfig, ::typeof(pairwise), s::SqEuclidean, x::AbstractMatrix; dims::Int=2)
  if dims==1
    return pairwise(s, x; dims=1), ∇pairwise(s, transpose(x), transpose)
  else
    return pairwise(s, x; dims=dims), ∇pairwise(s, x, identity)
  end
end

∇pairwise(s, x, f) =
  function pairwise_sqeuclidean_rrule(Δ)
    d1 = Diagonal(vec(sum(Δ; dims=1)))
    d2 = Diagonal(vec(sum(Δ; dims=2)))
    return NoTangent(), NoTangent(), x * (2 .* (d1 .+ d2 .- Δ .- transpose(Δ))) |> f
  end

function rrule(::ZygoteRuleConfig, ::Euclidean, x::AbstractVector, y::AbstractVector)
  D = x .- y
  δ = sqrt(sum(abs2, D))
  function euclidean_rrule(Δ::Real)
    x̄ = ifelse(iszero(δ), D, (Δ / δ) .* D)
    return NoTangent(), x̄, -x̄
  end
  return δ, euclidean_rrule
end

function rrule(::ZygoteRuleConfig, ::typeof(colwise), s::Euclidean, x::AbstractMatrix, y::AbstractMatrix)
  d = colwise(s, x, y)
  function colwise_Euclidean_rrule(Δ::AbstractVector)
    x̄ = (Δ ./ max.(d, eps(eltype(d))))' .* (x .- y)
    return NoTangent(), NoTangent(), x̄, -x̄
  end
  return d, colwise_Euclidean_rrule
end

_sqrt_if_positive(d, δ) = d > δ ? sqrt(d) : zero(d)

function rrule(config::ZygoteRuleConfig, ::typeof(pairwise), dist::Euclidean, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
  # Modify the forwards-pass slightly to ensure stability on the reverse.
  function _pairwise_euclidean(sqdist::SqEuclidean, X, Y)
    D2 = pairwise(sqdist, X, Y; dims)
    δ = eps(eltype(D2))
    return _sqrt_if_positive.(D2, δ)
  end
  return rrule_via_ad(config, _pairwise_euclidean, SqEuclidean(dist.thresh), X, Y)
end

function rrule(config::ZygoteRuleConfig, ::typeof(pairwise), dist::Euclidean, X::AbstractMatrix; dims=2)
  # Modify the forwards-pass slightly to ensure stability on the reverse.
  function _pairwise_euclidean(sqdist::SqEuclidean, X)
    D2 = pairwise(sqdist, X; dims)
    δ = eps(eltype(D2))
    return _sqrt_if_positive.(D2, δ)
  end
  return rrule_via_ad(config, _pairwise_euclidean, SqEuclidean(dist.thresh), X)
end
