using .Distances

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

  # Modify the forwards-pass slightly to ensure stability on the reverse.
  function _colwise_euclidean(x, y)
    δ = eps(promote_type(eltype(x), eltype(y)))^2
    return sqrt.(max.(colwise(SqEuclidean(), x, y), δ))
  end

  D, pb = pullback(_colwise_euclidean, x, y)

  colwise_Euclidean_binary_pullback(Δ) = (nothing, pb(Δ)...)

  return D, colwise_Euclidean_binary_pullback
end

@adjoint function pairwise(::Euclidean, X::AbstractMatrix, Y::AbstractMatrix; dims=2)

  # Modify the forwards-pass slightly to ensure stability on the reverse.
  function _pairwise_euclidean(X, Y)
    δ = eps(promote_type(eltype(X), eltype(Y)))^2
    return sqrt.(max.(pairwise(SqEuclidean(), X, Y; dims=dims), δ))
  end

  D, pb = pullback(_pairwise_euclidean, X, Y)

  pairwise_Euclidean_binary_pullback(Δ) = (nothing, pb(Δ)...)

  return D, pairwise_Euclidean_binary_pullback
end

@adjoint function pairwise(::Euclidean, X::AbstractMatrix; dims=2)

  # Modify the forwards-pass slightly to ensure stability on the reverse.
  function _pairwise_euclidean(X)
    return sqrt.(max.(pairwise(SqEuclidean(), X; dims=dims), eps(eltype(X))^2))
  end

  D, pb = pullback(_pairwise_euclidean, X)

  pairwise_Euclidean_unary_pullback(Δ) = (nothing, first(pb(Δ)))

  return D, pairwise_Euclidean_unary_pullback
end
