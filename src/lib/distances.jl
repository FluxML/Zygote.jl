using .Distances
import .ChainRules: NoTangent, rrule, rrule_via_ad

function rrule(::SqEuclidean, x::AbstractVector, y::AbstractVector)
  δ = x .- y
  function sqeuclidean(Δ::Real)
    x̄ = (2 * Δ) .* δ
    return NoTangent(), x̄, -x̄
  end
  return sum(abs2, δ), sqeuclidean
end

function rrule(::typeof(colwise), s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
  return colwise(s, x, y), function (Δ::AbstractVector)
    x̄ = 2 .* Δ' .* (x .- y)
    return NoTangent(), NoTangent(), x̄, -x̄
  end
end

function rrule(::typeof(pairwise), s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix; dims::Int=2)
  if dims==1
    return pairwise(s, x, y; dims=1), ∇pairwise(s, transpose(x), transpose(y), transpose)
  else
    return pairwise(s, x, y; dims=dims), ∇pairwise(s, x, y, identity)
  end
end

∇pairwise(s, x, y, f) =
  function(Δ)
    x̄ = 2 .* (x * Diagonal(vec(sum(Δ; dims=2))) .- y * transpose(Δ))
    ȳ = 2 .* (y * Diagonal(vec(sum(Δ; dims=1))) .- x * Δ)
    return NoTangent(), NoTangent(), f(x̄), f(ȳ)
  end

function rrule(::typeof(pairwise), s::SqEuclidean, x::AbstractMatrix; dims::Int=2)
  if dims==1
    return pairwise(s, x; dims=1), ∇pairwise(s, transpose(x), transpose)
  else
    return pairwise(s, x; dims=dims), ∇pairwise(s, x, identity)
  end
end

∇pairwise(s, x, f) =
  function(Δ)
    d1 = Diagonal(vec(sum(Δ; dims=1)))
    d2 = Diagonal(vec(sum(Δ; dims=2)))
    return NoTangent(), NoTangent(), x * (2 .* (d1 .+ d2 .- Δ .- transpose(Δ))) |> f
  end

function rrule(::Euclidean, x::AbstractVector, y::AbstractVector)
  D = x .- y
  δ = sqrt(sum(abs2, D))
  function euclidean(Δ::Real)
    x̄ = ifelse(iszero(δ), D, (Δ / δ) .* D)
    return NoTangent(), x̄, -x̄
  end
  return δ, euclidean
end

function rrule(::typeof(colwise), s::Euclidean, x::AbstractMatrix, y::AbstractMatrix)
  d = colwise(s, x, y)
  return d, function (Δ::AbstractVector)
    x̄ = (Δ ./ max.(d, eps(eltype(d))))' .* (x .- y)
    return NoTangent(), NoTangent(), x̄, -x̄
  end
end

function rrule(config::ZygoteRuleConfig, ::typeof(pairwise), ::Euclidean, X::AbstractMatrix, Y::AbstractMatrix; dims=2)

  # Modify the forwards-pass slightly to ensure stability on the reverse.
  function _pairwise_euclidean(X, Y)
    δ = eps(promote_type(eltype(X), eltype(Y)))^2
    return sqrt.(max.(pairwise(SqEuclidean(), X, Y; dims=dims), δ))
  end
  D, back = rrule_via_ad(config, _pairwise_euclidean, X, Y)

  return D, function(Δ)
    return (NoTangent(), NoTangent(), back(Δ)...)
  end
end

function rrule(config::ZygoteRuleConfig, ::typeof(pairwise), ::Euclidean, X::AbstractMatrix; dims=2)
  D, back = rrule_via_ad(config, X -> pairwise(SqEuclidean(), X; dims = dims), X)
  D .= sqrt.(D)
  return D, function(Δ)
    Δ = Δ ./ (2 .* max.(D, eps(eltype(D))))
    Δ[diagind(Δ)] .= 0
    return (NoTangent(), NoTangent(), first(back(Δ)))
  end
end
