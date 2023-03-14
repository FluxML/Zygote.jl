module ZygoteDistancesExt

if isdefined(Base, :get_extension)
    using Zygote
    using Distances
    using LinearAlgebra
else
    using ..Zygote
    using ..Distances
    using ..LinearAlgebra
end

using Zygote: @adjoint, AContext, _pullback

@adjoint function (::SqEuclidean)(x::AbstractVector, y::AbstractVector)
  δ = x .- y
  function sqeuclidean(Δ::Real)
    x̄ = (2 * Δ) .* δ
    return x̄, -x̄
  end
  return sum(abs2, δ), sqeuclidean
end

@adjoint function colwise(s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
  return colwise(s, x, y), function (Δ::AbstractVector)
    x̄ = 2 .* Δ' .* (x .- y)
    return nothing, x̄, -x̄
  end
end

@adjoint function pairwise(s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix; dims::Int=2)
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
    return (nothing, f(x̄), f(ȳ))
  end

@adjoint function pairwise(s::SqEuclidean, x::AbstractMatrix; dims::Int=2)
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
    return (nothing, x * (2 .* (d1 .+ d2 .- Δ .- transpose(Δ))) |> f)
  end

@adjoint function (::Euclidean)(x::AbstractVector, y::AbstractVector)
  D = x .- y
  δ = sqrt(sum(abs2, D))
  function euclidean(Δ::Real)
    x̄ = ifelse(iszero(δ), D, (Δ / δ) .* D)
    return x̄, -x̄
  end
  return δ, euclidean
end

@adjoint function colwise(s::Euclidean, x::AbstractMatrix, y::AbstractMatrix)
  d = colwise(s, x, y)
  return d, function (Δ::AbstractVector)
    x̄ = (Δ ./ max.(d, eps(eltype(d))))' .* (x .- y)
    return nothing, x̄, -x̄
  end
end

_sqrt_if_positive(d, δ) = d > δ ? sqrt(d) : zero(d)

function Zygote._pullback(cx::AContext, ::Core.kwftype(typeof(pairwise)),
                   kws::@NamedTuple{dims::Int}, ::typeof(pairwise), dist::Euclidean,
                   X::AbstractMatrix, Y::AbstractMatrix)
  # Modify the forwards-pass slightly to ensure stability on the reverse.
  dims = kws.dims
  function _pairwise_euclidean(sqdist::SqEuclidean, X, Y)
    D2 = pairwise(sqdist, X, Y; dims=dims)
    δ = eps(eltype(D2))
    return _sqrt_if_positive.(D2, δ)
  end
  res, back = _pullback(cx, _pairwise_euclidean, SqEuclidean(dist.thresh), X, Y)
  pairwise_Euclidean_pullback(Δ) = (nothing, nothing, back(Zygote.unthunk_tangent(Δ))...)
  return res, pairwise_Euclidean_pullback
end

function Zygote._pullback(cx::AContext, ::Core.kwftype(typeof(pairwise)),
                   kws::@NamedTuple{dims::Int}, ::typeof(pairwise), dist::Euclidean,
                   X::AbstractMatrix)
  # Modify the forwards-pass slightly to ensure stability on the reverse.
  dims = kws.dims
  function _pairwise_euclidean(sqdist::SqEuclidean, X)
    D2 = pairwise(sqdist, X; dims=dims)
    δ = eps(eltype(D2))
    return _sqrt_if_positive.(D2, δ)
  end
  res, back = _pullback(cx, _pairwise_euclidean, SqEuclidean(dist.thresh), X)
  pairwise_Euclidean_pullback(Δ) = (nothing, nothing, back(Zygote.unthunk_tangent(Δ))...)
  return res, pairwise_Euclidean_pullback
end

end
