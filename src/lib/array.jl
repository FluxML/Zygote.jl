@adjoint (::Type{T})(args...) where T<:Array = T(args...), Δ -> nothing

@nograd size, length, eachindex, Colon(), findfirst, randn, ones, zeros, one, zero


@adjoint Base.vect(xs...) = Base.vect(xs...), Δ -> (Δ...,)

Base.zero(xs::AbstractArray{Any}) = fill!(similar(xs), nothing)

@adjoint function getindex(xs::Array, i...)
  xs[i...], function (Δ)
    Δ′ = zero(xs)
    Δ′[i...] = Δ
    (Δ′, map(_ -> nothing, i)...)
  end
end

@adjoint! setindex!(xs::AbstractArray, x...) = setindex!(xs, x...),
  _ -> error("Mutating arrays is not supported")

# General

@adjoint collect(x) = collect(x), Δ -> (Δ,)

@adjoint permutedims(xs, dims) = permutedims(xs, dims),
  Δ -> (permutedims(Δ, invperm(dims)), nothing)

@adjoint reshape(xs, dims...) = reshape(xs, dims...),
  Δ -> (reshape(Δ, size(xs)),map(_->nothing,dims)...)

@adjoint function hvcat(rows::Tuple{Vararg{Int}}, xs::T...) where T<:Number
  hvcat(rows, xs...), ȳ -> (nothing, ȳ...)
end

pull_block_vert(sz, Δ, A::AbstractVector) = Δ[sz-length(A)+1:sz]
pull_block_vert(sz, Δ, A::AbstractMatrix) = Δ[sz-size(A, 1)+1:sz, :]
@adjoint function vcat(A::Union{AbstractVector, AbstractMatrix}...)
  sz = cumsum([size.(A, 1)...])
  return vcat(A...), Δ->(map(n->pull_block_vert(sz[n], Δ, A[n]), eachindex(A))...,)
end

pull_block_horz(sz, Δ, A::AbstractVector) = Δ[:, sz]
pull_block_horz(sz, Δ, A::AbstractMatrix) = Δ[:, sz-size(A, 2)+1:sz]
@adjoint function hcat(A::Union{AbstractVector, AbstractMatrix}...)
  sz = cumsum([size.(A, 2)...])
  return hcat(A...), Δ->(map(n->pull_block_horz(sz[n], Δ, A[n]), eachindex(A))...,)
end


@adjoint function repeat(xs; inner=ntuple(_->1, ndims(xs)), outer=ntuple(_->1, ndims(xs)))
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

@adjoint function sum(xs::AbstractArray; dims = :)
  if dims === (:)
    sum(xs), Δ -> (FillArray(Δ, size(xs)),)
  else
    sum(xs, dims = dims), Δ -> (similar(xs) .= Δ,)
  end
end

function _forward(cx::Context, ::typeof(sum), f, xs::AbstractArray)
  y, back = forward(cx, (xs -> sum(f.(xs))), xs)
  y, ȳ -> (nothing, nothing, back(ȳ)...)
end

@adjoint function prod(xs; dims = :)
  if dims === (:)
    prod(xs), Δ -> (prod(xs) ./ xs .* Δ,)
  else
    prod(xs, dims = dims),
      Δ -> (reshape(.*(circshift.([reshape(xs, length(xs))], 1:length(xs)-1)...), size(xs)) .* Δ,)
  end
end

@adjoint function maximum(xs; dims = :)
  max, i = findmax(xs, dims = dims)
  max, function (Δ)
    Δ isa Real && Δ <= sqrt(eps(float(Δ))) && return nothing
    Δ′ = zero(xs)
    Δ′[i] = Δ
    return (Δ′,)
  end
end

@adjoint function minimum(xs; dims = :)
  min, i = findmin(xs, dims = dims)
  min, function (Δ)
    Δ′ = zero(xs)
    Δ′[i] = Δ
    return (Δ′,)
  end
end

# LinAlg

@adjoint a::AbstractVecOrMat * b::AbstractVecOrMat = a * b,
  Δ -> (Δ * transpose(b), transpose(a) * Δ)

@adjoint transpose(x) = transpose(x), Δ -> (transpose(Δ),)
@adjoint Base.adjoint(x) = x', Δ -> (Δ',)
@adjoint parent(x::LinearAlgebra.Adjoint) = parent(x), ȳ -> (LinearAlgebra.Adjoint(ȳ),)

function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end

@adjoint kron(a::AbstractMatrix, b::AbstractMatrix) = forward(_kron, a, b)

@adjoint iterate(r::UnitRange, i...) = iterate(r, i...), _ -> nothing

@adjoint diag(A::AbstractMatrix) = diag(A), Δ->(Diagonal(Δ),)

@adjoint function \(A::AbstractMatrix, B::AbstractVecOrMat)
    Y = A \ B
    return Y, function(Ȳ)
        B̄ = A' \ Ȳ
        return (-B̄ * Y', B̄)
    end
end

@adjoint function /(A::AbstractMatrix, B::AbstractMatrix)
    Y = A / B
    return Y, function(Ȳ)
        Ā = Ȳ / B'
        return (Ā, -Y' * Ā)
    end
end

_symmetric_back(Δ) = UpperTriangular(Δ) + LowerTriangular(Δ)' - Diagonal(Δ)
_symmetric_back(Δ::UpperTriangular) = Δ
@adjoint function Symmetric(A::AbstractMatrix)
    back(Δ::AbstractMatrix) = (_symmetric_back(Δ),)
    back(Δ::NamedTuple) = (_symmetric_back(Δ.data),)
    return Symmetric(A), back
end

# Implementation due to Seeger, Matthias, et al. "Auto-differentiating linear algebra."
@adjoint function cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}})
    C = cholesky(Σ)
    return C, function(Δ)
        U, Ū = C.U, Δ.factors
        Σ̄ = Ū * U'
        Σ̄ = copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)
        @inbounds for n in diagind(Σ̄)
            Σ̄[n] /= 2
        end
        return (UpperTriangular(Σ̄),)
    end
end

# Various sensitivities for `literal_getproperty`, depending on the 2nd argument.
@adjoint function literal_getproperty(C::Cholesky, ::Val{:uplo})
    return literal_getproperty(C, Val(:uplo)), function(Δ)
        return ((uplo=nothing, info=nothing, factors=nothing),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:info})
    return literal_getproperty(C, Val(:info)), function(Δ)
        return ((uplo=nothing, info=nothing, factors=nothing),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:U})
    return literal_getproperty(C, Val(:U)), function(Δ)
        Δ_factors = C.uplo == 'U' ? UpperTriangular(Δ) : LowerTriangular(copy(Δ'))
        return ((uplo=nothing, info=nothing, factors=Δ_factors),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:L})
    return literal_getproperty(C, Val(:L)), function(Δ)
        Δ_factors = C.uplo == 'L' ? LowerTriangular(Δ) : UpperTriangular(copy(Δ'))
        return ((uplo=nothing, info=nothing, factors=Δ_factors),)
    end
end

@adjoint function logdet(C::Cholesky)
    return logdet(C), function(Δ)
        return ((uplo=nothing, info=nothing, factors=Diagonal(2 .* Δ ./ diag(C.factors))),)
    end
end

@adjoint function +(A::AbstractMatrix, S::UniformScaling)
    return A + S, Δ->(Δ, (λ=sum(view(Δ, diagind(Δ))),))
end
