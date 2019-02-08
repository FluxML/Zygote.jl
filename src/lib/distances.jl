using .Distances

@adjoint function sqeuclidean(x::AbstractVector, y::AbstractVector)
  δ = x .- y
  return sum(abs2, δ), function(Δ::Real)
    x̄ = (2 * Δ) .* δ
    return x̄, -x̄
  end
end

@adjoint function colwise(s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
  return colwise(s, x, y), function (Δ::AbstractVector)
    x̄ = 2 .* Δ' .* (x .- y)
    return nothing, x̄, -x̄
  end
end

@adjoint function pairwise(s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
  return pairwise(s, x, y), function(Δ)
    x̄ = 2 .* (x * Diagonal(vec(sum(Δ; dims=2))) .- y * Δ')
    ȳ = 2 .* (y * Diagonal(vec(sum(Δ; dims=1))) .- x * Δ)
    return nothing, x̄, ȳ
  end
end

@adjoint function pairwise(s::SqEuclidean, X::AbstractMatrix)
  D = pairwise(s, X)
  return D, function(Δ)
    d1, d2 = Diagonal(vec(sum(Δ; dims=1))), Diagonal(vec(sum(Δ; dims=2)))
    return (nothing, X * (2 .* (d1 .+ d2 .- Δ .- Δ')))
  end
end
