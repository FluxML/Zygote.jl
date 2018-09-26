@grad (::Type{T})(args...) where T<:Array = T(args...), Δ -> nothing

@nograd size, length, eachindex, Colon(), findfirst

@grad Base.vect(xs...) = Base.vect(xs...), Δ -> (Δ...,)

Base.zero(xs::AbstractArray{Any}) = fill!(similar(xs), nothing)

@grad function getindex(xs::Array, i...)
  xs[i...], function (Δ)
    Δ′ = zero(xs)
    Δ′[i...] = Δ
    (Δ′, map(_ -> nothing, i)...)
  end
end

@grad! setindex!(xs::AbstractArray, x...) = setindex!(xs, x...),
  _ -> error("Mutating arrays is not supported")

# General

@grad collect(x) = collect(x), Δ -> (Δ,)

@grad permutedims(xs, dims) = permutedims(xs, dims),
  Δ -> (permutedims(Δ, invperm(dims)), nothing)

@grad reshape(xs, dims...) = reshape(xs, dims...),
  Δ -> (reshape(Δ, size(xs)),map(_->nothing,dims)...)

@grad function repeat(xs; inner=ntuple(_->1, ndims(xs)), outer=ntuple(_->1, ndims(xs)))
  repeat(xs, inner = inner, outer = outer), function (Δ)
    Δ′ = zero(xs)
    S = size(xs)

    # Loop through each element of Δ, calculate source dimensions, accumulate into Δ′
    for (dest_idx, val) in pairs(IndexCartesian(), Δ)
        # First, round dest_idx[dim] to nearest gridpoint defined by inner[dim], then
        # wrap around based on original size S.
        src_idx = [mod1(div(dest_idx[dim] - 1, inner[dim]) + 1, S[dim]) for dim in 1:length(S)]
        Δ′[src_idx...] += val
    end
    return (Δ′,)
  end
end

# Reductions

fill_similar_array(xs, v) = similar(xs) .= Δ
@grad sum(xs::AbstractArray; dims = :) =
  sum(xs, dims = dims), Δ -> (fill_similar_array(xs, Δ),)

@grad prod(xs; dims) = prod(xs, dims = dims),
  Δ -> (reshape(.*(circshift.([reshape(xs, length(xs))], 1:length(xs)-1)...), size(xs)) .* Δ,)

@grad prod(xs) = prod(xs), Δ -> (prod(xs) ./ xs .* Δ,)

@grad function maximum(xs; dims = :)
  maximum(xs, dims = dims), function (Δ)
    Δ′ = zero(xs)
    _, i = findmax(xs, dims = dims)
    Δ′[i] = Δ
    return (Δ′,)
  end
end

@grad function minimum(xs; dims = :)
  minimum(xs, dims = dims), function (Δ)
    Δ′ = zero(xs)
    _, i = findmin(xs, dims = dims)
    Δ′[i] = Δ
    return (Δ′,)
  end
end

# LinAlg

@grad a::AbstractVecOrMat * b::AbstractVecOrMat = a * b,
  Δ -> (Δ * transpose(b), transpose(a) * Δ)

@grad transpose(x) = transpose(x), Δ -> (transpose(Δ),)
@grad adjoint(x) = adjoint(x), Δ -> (adjoint(Δ),)

function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end

@grad kron(a::AbstractMatrix, b::AbstractMatrix) = forward(_kron, a, b)
