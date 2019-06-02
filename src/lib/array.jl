using FillArrays

@adjoint (::Type{T})(::UndefInitializer, args...) where T<:Array = T(undef, args...), Δ -> nothing

@nograd size, length, eachindex, Colon(), findfirst, randn, ones, zeros, one, zero,
  print, println


@adjoint Base.vect(xs...) = Base.vect(xs...), Δ -> (Δ...,)

@adjoint copy(x::AbstractArray) = copy(x), ȳ -> (ȳ,)

# Array Constructors
@adjoint (::Type{T})(x::T) where T<:Array = T(x), ȳ -> (ȳ,)
@adjoint (::Type{T})(x::Number, sz) where {T <: Fill} = Fill(x, sz), Δ -> (sum(Δ), nothing)

_zero(xs::AbstractArray{<:Integer}) = fill!(similar(xs, float(eltype(xs))), false)
_zero(xs::AbstractArray{<:Number}) = zero(xs)
_zero(xs::AbstractArray) = Any[nothing for x in xs]

@adjoint function getindex(xs::Array, i...)
  xs[i...], function (Δ)
    Δ′ = _zero(xs)
    Δ′[i...] = Δ
    (Δ′, map(_ -> nothing, i)...)
  end
end

@adjoint! setindex!(xs::AbstractArray, x...) = setindex!(xs, x...),
  _ -> error("Mutating arrays is not supported")

# General

@adjoint collect(x::Array) = collect(x), Δ -> (Δ,)

@adjoint permutedims(xs, dims) = permutedims(xs, dims),
  Δ -> (permutedims(Δ, invperm(dims)), nothing)

@adjoint reshape(xs, dims...) = reshape(xs, dims...),
  Δ -> (reshape(Δ, size(xs)),map(_->nothing,dims)...)

@adjoint function hvcat(rows::Tuple{Vararg{Int}}, xs::T...) where T<:Number
    hvcat(rows, xs...), ȳ -> (nothing, transpose(ȳ)...)
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

@adjoint getindex(i::Int, j::Int) = i[j], _ -> nothing

function unzip(tuples)
  map(1:length(first(tuples))) do i
      map(tuple -> tuple[i], tuples)
  end
end
@adjoint function map(f, args::AbstractArray...)
  ys_and_backs = map((args...) -> _forward(__context__, f, args...), args...)
  ys, backs = unzip(ys_and_backs)
  ys, function (Δ)
    Δf_and_args_zipped = map((f, δ) -> f(δ), backs, Δ)
    Δf_and_args = unzip(Δf_and_args_zipped)
    Δf = reduce(accum, Δf_and_args[1])
    (Δf, Δf_and_args[2:end]...)
  end
end

function _forward(cx::Context, ::typeof(collect), g::Base.Generator)
  y, back = _forward(cx, map, g.f, g.iter)
  y, function (ȳ)
    _, f̄, x̄ = back(ȳ)
    (nothing, (f = f̄, iter = x̄),)
  end
end

@adjoint iterate(r::UnitRange, i...) = iterate(r, i...), _ -> nothing

# Reductions

@adjoint function sum(xs::AbstractArray; dims = :)
  if dims === (:)
    sum(xs), Δ -> (Fill(Δ, size(xs)),)
  else
    sum(xs, dims = dims), Δ -> (similar(xs) .= Δ,)
  end
end

function _forward(cx::Context, ::typeof(sum), f, xs::AbstractArray)
  y, back = forward(cx, (xs -> sum(f.(xs))), xs)
  y, ȳ -> (nothing, nothing, back(ȳ)...)
end

@adjoint function sum(::typeof(abs2), X::AbstractArray; dims = :)
  return sum(abs2, X; dims=dims), Δ::Union{Number, AbstractArray}->(nothing, ((2Δ) .* X))
end

@adjoint function prod(xs; dims = :)
  if dims === (:)
    prod(xs), Δ -> (prod(xs) ./ xs .* Δ,)
  else
    prod(xs, dims = dims),
      Δ -> (reshape(.*(circshift.([reshape(xs, length(xs))], 1:length(xs)-1)...), size(xs)) .* Δ,)
  end
end

function _forward(cx::Context, ::typeof(prod), f, xs::AbstractArray)
  y, back = forward(cx, (xs -> prod(f.(xs))), xs)
  y, ȳ -> (nothing, nothing, back(ȳ)...)
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

@adjoint function mean(xs::AbstractArray; dims = :)
  return mean(xs, dims=dims), Δ -> (_backmean(xs,Δ,dims),)
end
_backmean(xs, Δ, ::Colon) = zero(xs) .+ Δ ./ length(xs)
_backmean(xs, Δ, dims) = zero(xs) .+ Δ ./ mapreduce(i -> size(xs,i),*,dims)

# LinAlg
# ======

@adjoint function(a::AbstractVecOrMat * b::AbstractVecOrMat)
  return a * b, function(Δ)
    return (reshape(Δ * b', size(a)), reshape(a' * Δ, size(b)))
  end
end

@adjoint transpose(x) = transpose(x), Δ -> (transpose(Δ),)
@adjoint Base.adjoint(x) = x', Δ -> (Δ',)
@adjoint parent(x::LinearAlgebra.Adjoint) = parent(x), ȳ -> (LinearAlgebra.Adjoint(ȳ),)

@adjoint dot(x::AbstractArray, y::AbstractArray) = dot(x, y), Δ->(Δ .* y, Δ .* x)

function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end

@adjoint kron(a::AbstractMatrix, b::AbstractMatrix) = forward(_kron, a, b)

@adjoint function Diagonal(d::AbstractVector)
  back(Δ::NamedTuple) = (Δ.diag,)
  back(Δ::AbstractMatrix) = (diag(Δ),)
  return Diagonal(d), back
end

@adjoint diag(A::AbstractMatrix) = diag(A), Δ->(Diagonal(Δ),)

@adjoint det(xs) = det(xs), Δ -> (Δ * det(xs) * transpose(inv(xs)),)

@adjoint logdet(xs) = logdet(xs), Δ -> (Δ * transpose(inv(xs)),)

@adjoint logabsdet(xs) = logabsdet(xs), Δ -> (Δ[1] * transpose(inv(xs)),)

@adjoint function inv(A)
    return inv(A), function (Δ)
        Ainv = inv(A)
        ∇A = - Ainv' * Δ * Ainv'
        return (∇A, )
    end
end

@adjoint function \(A::AbstractMatrix, B::AbstractVecOrMat)
  Y = A \ B
  return Y, function(Ȳ)
      B̄ = A' \ Ȳ
      return (-B̄ * Y', B̄)
  end
end

function _forward(cx::Context, ::typeof(norm), x::AbstractArray, p::Real = 2)
  fallback = (x, p) -> sum(abs.(x).^p .+ eps(0f0))^(1/p) # avoid d(sqrt(x))/dx == Inf at 0
  _forward(cx, fallback, x, p)
end

# LinAlg Matrix Types
# ===================

# This is basically a hack while we don't have a working `ldiv!`.
@adjoint function \(A::Cholesky, B::AbstractVecOrMat)
  Y, back = Zygote.forward((U, B)->U \ (U' \ B), A.U, B)
  return Y, function(Ȳ)
    Ā_factors, B̄ = back(Ȳ)
    return ((uplo=nothing, status=nothing, factors=Ā_factors), B̄)
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
_symmetric_back(Δ::Union{Diagonal, UpperTriangular}) = Δ
@adjoint function Symmetric(A::AbstractMatrix)
  back(Δ::AbstractMatrix) = (_symmetric_back(Δ),)
  back(Δ::NamedTuple) = (_symmetric_back(Δ.data),)
  return Symmetric(A), back
end

@adjoint function cholesky(Σ::Real)
  C = cholesky(Σ)
  return C, Δ::NamedTuple->(Δ.factors[1, 1] / (2 * C.U[1, 1]),)
end

@adjoint function cholesky(Σ::Diagonal)
  C = cholesky(Σ)
  return C, Δ::NamedTuple->(Diagonal(diag(Δ.factors) .* inv.(2 .* C.factors.diag)),)
end

# Implementation due to Seeger, Matthias, et al. "Auto-differentiating linear algebra."
@adjoint function cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}})
  C = cholesky(Σ)
  return C, function(Δ::NamedTuple)
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

@adjoint function lyap(A::AbstractMatrix, C::AbstractMatrix)
  X = lyap(A, C)
  return X, function (X̄)
    C̄ = lyap(collect(A'), X̄)
    Ā = C̄*X' + C̄'*X
    return (Ā, C̄)
  end
end

# Adjoint based on the Theano implementation, which uses the differential as described
# in Brančík, "Matlab programs for matrix exponential function derivative evaluation"
@adjoint exp(A::AbstractMatrix) = exp(A), function(F̄)
  n = size(A, 1)
  E = eigen(A)
  w = E.values
  ew = exp.(w)
  X = [i==j ? ew[i] : (ew[i]-ew[j])/(w[i]-w[j]) for i in 1:n,j=1:n]
  VT = transpose(E.vectors)
  VTF = factorize(collect(VT))
  Ā = real.(VTF\(VT*F̄/VTF.*X)*VT)
  (Ā, )
end

Zygote.@adjoint function LinearAlgebra.tr(x::AbstractMatrix)
  # x is a squre matrix checked by tr,
  # so we could just use Eye(size(x, 1))
  # to create a Diagonal
  tr(x), function (Δ::Number)
    (Diagonal(Fill(Δ, (size(x, 1), ))), )
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
