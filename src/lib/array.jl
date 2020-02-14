using FillArrays, FFTW
using FillArrays: AbstractFill, getindex_value
using Base.Broadcast: broadcasted, broadcast_shape

@adjoint (::Type{T})(::UndefInitializer, args...) where T<:Array = T(undef, args...), Δ -> nothing

@adjoint Array(xs::AbstractArray) = Array(xs), ȳ -> (ȳ,)
@adjoint Array(xs::Array) = Array(xs), ȳ -> (ȳ,)

@nograd size, length, eachindex, axes, Colon(), findfirst, findlast, findall, randn, ones, zeros, one, zero,
  print, println, any, all

@adjoint rand(dims::Integer...) = rand(dims...), _ -> nothing

@adjoint Base.vect(xs...) = Base.vect(xs...), Δ -> (Δ...,)

@adjoint copy(x::AbstractArray) = copy(x), ȳ -> (ȳ,)

# Array Constructors
@adjoint (::Type{T})(x::T) where T<:Array = T(x), ȳ -> (ȳ,)
@adjoint function (::Type{T})(x::Number, sz) where {T <: Fill}
    back(Δ::AbstractArray) = (sum(Δ), nothing)
    back(Δ::NamedTuple) = (Δ.value, nothing)
    return Fill(x, sz), back
end

@adjoint (::Type{T})(sz) where {T<:Zeros} = Zeros(sz), Δ->(nothing,)
@adjoint (::Type{T})(sz) where {T<:Ones} = Ones(sz), Δ->(nothing,)

_zero(xs::AbstractArray{<:Number}, T=float(eltype(xs))) = fill!(similar(xs, T), false)
_zero(xs::AbstractArray, T=Any) = Union{Nothing, T}[nothing for x in xs]

@adjoint getindex(x::AbstractArray, inds...) = x[inds...], ∇getindex(x, inds)

@adjoint view(x::AbstractArray, inds...) = view(x, inds...), ∇getindex(x, inds)

∇getindex(x::AbstractArray, inds) = dy -> begin
  if inds isa  NTuple{<:Any,Integer}
    dx = _zero(x, typeof(dy))
    dx[inds...] = dy
  else
    dx = _zero(x, eltype(dy))
    @views dx[inds...] .+= dy
  end
  (dx, map(_->nothing, inds)...)
end

@adjoint! setindex!(xs::AbstractArray, x...) = setindex!(xs, x...),
  _ -> error("Mutating arrays is not supported")

for f in [push!, pop!, pushfirst!, popfirst!]
  @eval @adjoint! $f(xs::Vector, x...) =
    push!(xs, x...), _ -> error("Mutating arrays is not supported")
end

# General

@adjoint collect(x::Array) = collect(x), Δ -> (Δ,)

@adjoint fill(x::Real, dims...) = fill(x, dims...), Δ->(sum(Δ), map(_->nothing, dims)...)

@adjoint function circshift(A, shifts)
  circshift(A, shifts), Δ -> (circshift(Δ, map(-, shifts)), nothing)
end

@adjoint permutedims(xs) = permutedims(xs), Δ -> (permutedims(Δ),)

@adjoint permutedims(xs::AbstractVector) = permutedims(xs), Δ -> (vec(permutedims(Δ)),)

@adjoint permutedims(xs, dims) = permutedims(xs, dims),
  Δ -> (permutedims(Δ, invperm(dims)), nothing)

@adjoint PermutedDimsArray(xs, dims) = PermutedDimsArray(xs, dims),
  Δ -> (PermutedDimsArray(Δ, invperm(dims)), nothing)

@adjoint reshape(xs, dims...) = reshape(xs, dims...),
  Δ -> (reshape(Δ, size(xs)),map(_->nothing,dims)...)

@adjoint function hvcat(rows::Tuple{Vararg{Int}}, xs::T...) where T<:Number
  hvcat(rows, xs...), ȳ -> (nothing, permutedims(ȳ)...)
end

pull_block_vert(sz, Δ, A::AbstractVector) = Δ[sz-length(A)+1:sz]
pull_block_vert(sz, Δ, A::AbstractMatrix) = Δ[sz-size(A, 1)+1:sz, :]
@adjoint function vcat(A::Union{AbstractVector, AbstractMatrix}...)
  sz = cumsum([size.(A, 1)...])
  return vcat(A...), Δ->(map(n->pull_block_vert(sz[n], Δ, A[n]), eachindex(A))...,)
end
@adjoint vcat(xs::Number...) = vcat(xs...), Δ -> (Δ...,)

pull_block_horz(sz, Δ, A::AbstractVector) = Δ[:, sz]
pull_block_horz(sz, Δ, A::AbstractMatrix) = Δ[:, sz-size(A, 2)+1:sz]
@adjoint function hcat(A::Union{AbstractVector, AbstractMatrix}...)
  sz = cumsum([size.(A, 2)...])
  return hcat(A...), Δ->(map(n->pull_block_horz(sz[n], Δ, A[n]), eachindex(A))...,)
end
@adjoint hcat(xs::Number...) = hcat(xs...), Δ -> (Δ...,)

@adjoint function cat(Xs...; dims)
  cat(Xs...; dims = dims), Δ -> begin
    start = ntuple(_ -> 0, ndims(Δ))
    dXs = map(Xs) do x
      move = ntuple(d -> d in dims ? size(x,d) : 0, ndims(Δ))
      x_in_Δ = ntuple(d -> move[d] > 0 ? (start[d]+1:start[d]+move[d]) : Colon(), ndims(Δ))
      start = start .+ move
      dx = reshape(Δ[x_in_Δ...], size(x))
    end
  end
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
function ∇map(cx, f, args...)
  ys_and_backs = map((args...) -> _pullback(cx, f, args...), args...)
  if isempty(ys_and_backs)
    ys_and_backs, _ -> nothing
  else
    ys, backs = unzip(ys_and_backs)
    ys, function (Δ)
      Δf_and_args_zipped = map((f, δ) -> f(δ), backs, Δ)
      Δf_and_args = unzip(Δf_and_args_zipped)
      Δf = reduce(accum, Δf_and_args[1])
      (Δf, Δf_and_args[2:end]...)
    end
  end
end

@adjoint function map(f, args::Union{AbstractArray,Tuple}...)
  ∇map(__context__, f, args...)
end

function _pullback(cx::AContext, ::typeof(collect), g::Base.Generator)
  y, back = ∇map(cx, g.f, g.iter)
  y, function (ȳ)
    f̄, x̄ = back(ȳ)
    (nothing, (f = f̄, iter = x̄),)
  end
end

@adjoint iterate(r::UnitRange, i...) = iterate(r, i...), _ -> nothing

@adjoint function sort(x::AbstractArray)
  p = sortperm(x)
  return x[p], x̄ -> (x̄[invperm(p)],)
end

# Reductions

@adjoint function sum(xs::AbstractArray; dims = :)
  if dims === (:)
    sum(xs), Δ -> (Fill(Δ, size(xs)),)
  else
    sum(xs, dims = dims), Δ -> (similar(xs) .= Δ,)
  end
end

function _pullback(cx::AContext, ::typeof(sum), f, xs::AbstractArray)
  y, back = pullback(cx, (xs -> sum(f.(xs))), xs)
  y, ȳ -> (nothing, nothing, back(ȳ)...)
end

@adjoint function sum(::typeof(abs2), X::AbstractArray; dims = :)
  return sum(abs2, X; dims=dims), Δ::Union{Number, AbstractArray}->(nothing, ((2Δ) .* X))
end

@adjoint function prod(xs; dims = :)
  p = prod(xs; dims = dims)
  p, Δ -> (p ./ xs .* Δ,)
end

function _pullback(cx::AContext, ::typeof(prod), f, xs::AbstractArray)
  y, back = pullback(cx, (xs -> prod(f.(xs))), xs)
  y, ȳ -> (nothing, nothing, back(ȳ)...)
end

@adjoint function maximum(xs::AbstractArray; dims = :)
  max, i = findmax(xs, dims = dims)
  max, function (Δ)
    Δ isa Real && abs(Δ) <= sqrt(eps(float(Δ))) && return nothing
    Δ′ = zero(xs)
    Δ′[i] = Δ
    return (Δ′,)
  end
end

@adjoint function minimum(xs::AbstractArray; dims = :)
  min, i = findmin(xs, dims = dims)
  min, function (Δ)
    Δ′ = zero(xs)
    Δ′[i] = Δ
    return (Δ′,)
  end
end

@adjoint function dropdims(xs::AbstractArray; dims)
  dropdims(xs, dims = dims), Δ -> (reshape(Δ, size(xs)...),)
end

@adjoint real(x::AbstractArray) = real(x), r̄ -> (real(r̄),)
@adjoint conj(x::AbstractArray) = conj(x), r̄ -> (conj(r̄),)
@adjoint imag(x::AbstractArray) = imag(x), ī -> (complex.(0, real.(ī)),)

@adjoint function mean(xs::AbstractArray; dims = :)
  return mean(xs, dims=dims), Δ -> (_backmean(xs,Δ,dims),)
end
_backmean(xs, Δ, ::Colon) = zero(xs) .+ Δ ./ length(xs)
_backmean(xs, Δ, dims) = zero(xs) .+ Δ ./ mapreduce(i -> size(xs,i),*,dims)

@adjoint function Statistics.var(xs::AbstractArray; corrected::Bool=true, dims=:, mean=mean(xs, dims=dims))
    return Statistics.var(xs; corrected=corrected, mean=mean, dims=dims), Δ -> _backvar(xs, Δ, corrected, mean, dims)
end
_backvar(xs, Δ, corrected::Bool, mean, dims)         = _backvar(xs, Δ, mapreduce(i -> size(xs,i),*,dims) - corrected, mean)
_backvar(xs, Δ, corrected::Bool, mean, ::Colon)      = _backvar(xs, Δ, length(xs) - corrected, mean)
_backvar(xs, Δ, N::Int, mean) = (convert(eltype(xs), 2/N) .* Δ .* (xs .- mean),)

@adjoint function Statistics.std(xs::AbstractArray; corrected::Bool=true, dims=:, mean=mean(xs, dims=dims))
    s = Statistics.std(xs; corrected=corrected, mean=mean, dims=dims)
    return s, Δ -> _backvar(xs, Δ ./ (2 .* s), corrected, mean, dims)
end


# LinAlg
# ======

@adjoint function(A::AbstractMatrix * B::AbstractMatrix)
  return A * B, Δ::AbstractMatrix->(Δ * B', A' * Δ)
end

@adjoint function(A::AbstractMatrix * x::AbstractVector)
  return A * x, Δ::AbstractVector->(Δ * x', A' * Δ)
end

@adjoint function *(x::Union{Transpose{<:Any, <:AbstractVector},
                             LinearAlgebra.Adjoint{<:Any, <:AbstractVector}},
                    y::AbstractVector)
  return x * y, Δ->(Δ * y', x' * Δ)
end

@adjoint function(a::AbstractVector * x::AbstractMatrix)
  return a * x, Δ::AbstractMatrix->(vec(Δ * x'), a' * Δ)
end

@adjoint function transpose(x)
  back(Δ) = (transpose(Δ),)
  back(Δ::NamedTuple{(:parent,)}) = (Δ.parent,)
  return transpose(x), back
end

@adjoint function Base.adjoint(x)
  back(Δ) = (Δ',)
  back(Δ::NamedTuple{(:parent,)}) = (Δ.parent,)
  return x', back
end

@adjoint parent(x::LinearAlgebra.Adjoint) = parent(x), ȳ -> (LinearAlgebra.Adjoint(ȳ),)

@adjoint dot(x::AbstractArray, y::AbstractArray) = dot(x, y), Δ->(Δ .* y, Δ .* x)

function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end

@adjoint kron(a::AbstractMatrix, b::AbstractMatrix) = pullback(_kron, a, b)

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
  Ainv = inv(A)
  return Ainv, function (Δ)
    ∇A = - Ainv' * Δ * Ainv'
    return (∇A, )
  end
end

# Defaults for atol and rtol copied directly from LinearAlgebra. See the following for
# derivation:
# Golub, Gene H., and Victor Pereyra. "The differentiation of pseudo-inverses and nonlinear
# least squares problems whose variables separate." SIAM Journal on numerical analysis 10.2
# (1973): 413-432.
@adjoint function pinv(
  A::AbstractMatrix{T};
  atol::Real = 0.0,
  rtol::Real = (eps(real(float(one(T))))*min(size(A)...))*iszero(atol),
) where {T}
  Y = pinv(A)
  return Y, Δ->(-Y' * Δ * Y' + (I - A * Y) * Δ' * Y * Y' + Y' * Y * Δ' * (I - Y * A),)
end

# When `A` is guaranteed to be square, definitely use the simple expression for the adjoint.
@adjoint function \(
  A::Union{
    Diagonal,
    AbstractTriangular,
    LinearAlgebra.Adjoint{<:Any, <:AbstractTriangular},
    Transpose{<:Any, <:AbstractTriangular},
  },
  B::AbstractVecOrMat,
)
  Y = A \ B
  return Y, function(Ȳ)
    B̄ = A' \ Ȳ
    return (-B̄ * Y', B̄)
  end
end

@adjoint function /(A::AbstractMatrix, B::Union{Diagonal, AbstractTriangular})
  Y = A / B
  return Y, function(Ȳ)
    Ā = Ȳ / B'
    return (Ā, -Y' * Ā)
  end
end

@adjoint function \(A::AbstractMatrix, B::AbstractVecOrMat)
  Z = A \ B
  return Z, function(Z̄)
    B̄ = A' \ Z̄
    if size(A, 1) == size(A, 2)
      return (-B̄ * Z', B̄)
    else
      a = -B̄ * Z'
      b = (B - A * Z) * B̄' / A'
      c = A' \ Z * (Z̄' - B̄' * A)
      return (a + b + c, B̄)
    end
  end
end

function _pullback(cx::AContext, ::typeof(norm), x::AbstractArray, p::Real = 2)
  fallback = (x, p) -> sum(abs.(x).^p .+ eps(0f0))^(1/p) # avoid d(sqrt(x))/dx == Inf at 0
  _pullback(cx, fallback, x, p)
end

# LinAlg Matrix Types
# ===================

@adjoint LinearAlgebra.LowerTriangular(A) = LowerTriangular(A), Δ->(LowerTriangular(Δ),)
@adjoint LinearAlgebra.UpperTriangular(A) = UpperTriangular(A), Δ->(UpperTriangular(Δ),)
@adjoint LinearAlgebra.UnitLowerTriangular(A) = UnitLowerTriangular(A), Δ->(UnitLowerTriangular(Δ)-I,)
@adjoint LinearAlgebra.UnitUpperTriangular(A) = UnitUpperTriangular(A), Δ->(UnitUpperTriangular(Δ)-I,)

# This is basically a hack while we don't have a working `ldiv!`.
@adjoint function \(A::Cholesky, B::AbstractVecOrMat)
  Y, back = Zygote.pullback((U, B)->U \ (U' \ B), A.U, B)
  return Y, function(Ȳ)
    Ā_factors, B̄ = back(Ȳ)
    return ((uplo=nothing, status=nothing, factors=Ā_factors), B̄)
  end
end

function _symmetric_back(Δ, uplo)
  L, U, D = LowerTriangular(Δ), UpperTriangular(Δ), Diagonal(Δ)
  return uplo == 'U' ? U .+ transpose(L) - D : L .+ transpose(U) - D
end
_symmetric_back(Δ::Diagonal, uplo) = Δ
_symmetric_back(Δ::UpperTriangular, uplo) = collect(uplo == 'U' ? Δ : transpose(Δ))
_symmetric_back(Δ::LowerTriangular, uplo) = collect(uplo == 'U' ? transpose(Δ) : Δ)

@adjoint function Symmetric(A::AbstractMatrix, uplo=:U)
  S = Symmetric(A, uplo)
  back(Δ::AbstractMatrix) = (_symmetric_back(Δ, S.uplo), nothing)
  back(Δ::NamedTuple) = (_symmetric_back(Δ.data, S.uplo), nothing)
  return S, back
end

_extract_imag(x) = (x->complex(0, imag(x))).(x)
function _hermitian_back(Δ, uplo)
  isreal(Δ) && return _symmetric_back(Δ, uplo)
  L, U, rD = LowerTriangular(Δ), UpperTriangular(Δ), real.(Diagonal(Δ))
  return uplo == 'U' ? U .+ L' - rD : L .+ U' - rD
end
_hermitian_back(Δ::Diagonal, uplo) = real.(Δ)
function _hermitian_back(Δ::LinearAlgebra.AbstractTriangular, uplo)
  isreal(Δ) && return _symmetric_back(Δ, uplo)
  ŪL̄ = Δ .- Diagonal(_extract_imag(diag(Δ)))
  if istriu(Δ)
    return collect(uplo == 'U' ? ŪL̄ : ŪL̄')
  else
    return collect(uplo == 'U' ? ŪL̄' : ŪL̄)
  end
end

@adjoint function LinearAlgebra.Hermitian(A::AbstractMatrix, uplo=:U)
  H = Hermitian(A, uplo)
  back(Δ::AbstractMatrix) = (_hermitian_back(Δ, H.uplo), nothing)
  back(Δ::NamedTuple) = (_hermitian_back(Δ.data, H.uplo), nothing)
  return H, back
end

@adjoint convert(::Type{R}, A::LinearAlgebra.HermOrSym{T,S}) where {T,S,R<:Array} = convert(R, A),
  Δ -> (nothing, convert(S, Δ),)
@adjoint Matrix(A::LinearAlgebra.HermOrSym{T,S}) where {T,S} = Matrix(A),
  Δ -> (convert(S, Δ),)

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

# Matrix of pairwise difference quotients
Base.@propagate_inbounds function _pairdiffquot(f, i, j, x, fx, dfx, d²fx = nothing)
  i == j && return dfx[i]
  Δx = x[i] - x[j]
  T = real(eltype(x))
  if d²fx === nothing
    abs(Δx) ≤ sqrt(eps(T)) && return (dfx[i] + dfx[j]) / 2
  else
    abs(Δx) ≤ eps(T)^(1/3) && return dfx[i] - Δx / 2 * d²fx[i]
  end
  Δfx = fx[i] - fx[j]
  return Δfx / Δx
end

Base.@propagate_inbounds function _pairdiffquotmat(f, n, x, fx, dfx, d²fx = nothing)
  Δfij = (i, j)->_pairdiffquot(f, i, j, x, fx, dfx, d²fx)
  return Δfij.(Base.OneTo(n), Base.OneTo(n)')
end

# Adjoint based on the Theano implementation, which uses the differential as described
# in Brančík, "Matlab programs for matrix exponential function derivative evaluation"
@adjoint exp(A::AbstractMatrix) = exp(A), function(F̄)
  n = size(A, 1)
  E = eigen(A)
  w = E.values
  ew = exp.(w)
  X = _pairdiffquotmat(exp, n, w, ew, ew, ew)
  V = E.vectors
  VF = factorize(V)
  Āc = (V * ((VF \ F̄' * V) .* X) / VF)'
  Ā = isreal(A) && isreal(F̄) ? real(Āc) : Āc
  return (Ā,)
end

@adjoint function LinearAlgebra.eigen(A::LinearAlgebra.RealHermSymComplexHerm)
  dU = eigen(A)
  return dU, function (Δ)
    d, U = dU
    d̄, Ū = Δ
    if Ū === nothing
      P = Diagonal(d̄)
    else
      F = inv.(d' .- d)
      P = F .* (U' * Ū)
      if d̄ === nothing
        P[diagind(P)] .= 0
      else
        P[diagind(P)] = d̄
      end
    end
    return (U * P * U',)
  end
end

@adjoint function LinearAlgebra.eigvals(A::LinearAlgebra.RealHermSymComplexHerm)
  d, U = eigen(A)
  return d, d̄ -> (U * Diagonal(d̄) * U',)
end


# Hermitian/Symmetric matrix functions that can be written as power series
_realifydiag!(A::AbstractArray{<:Real}) = A
function _realifydiag!(A)
  n = LinearAlgebra.checksquare(A)
  for i in 1:n
      @inbounds A[i,i] = real(A[i,i])
  end
  return A
end
@adjoint _realifydiag!(A) = _realifydiag!(A), Δ -> (_realifydiag!(Δ),)

_hasrealdomain(f, x) = true
_hasrealdomain(::Union{typeof.((acos,asin))...}, x) = all(x -> -1 ≤ x ≤ 1, x)
_hasrealdomain(::typeof(acosh), x) = all(x -> x ≥ 1, x)
_hasrealdomain(::Union{typeof.((log,sqrt,^))...}, x) = all(x -> x ≥ 0, x)

_process_series_eigvals(f, λ) = _hasrealdomain(f, λ) ? λ : complex.(λ)

_process_series_matrix(f, fA, A, fλ) = fA
_process_series_matrix(f, fA, ::LinearAlgebra.HermOrSym{<:Real}, fλ) = Symmetric(fA)
_process_series_matrix(f, fA, ::Hermitian{<:Complex}, ::AbstractVector{<:Real}) =
  Hermitian(_realifydiag!(fA))
_process_series_matrix(::typeof(^), fA, ::Hermitian{<:Real}, fλ) = Hermitian(fA)
_process_series_matrix(::typeof(^), fA, ::Hermitian{<:Real}, ::AbstractVector{<:Complex}) = fA
_process_series_matrix(::typeof(^), fA, ::Hermitian{<:Complex}, ::AbstractVector{<:Complex}) = fA

# Compute function on eigvals, thunks for conjugates of 1st and 2nd derivatives,
# and function to pull back adjoints to args
function _pullback_series_func_scalar(f, λ, args...)
  compλ = _process_series_eigvals(f, λ)
  fλ, fback = Zygote.pullback((x,args...) -> f.(x, args...), compλ, args...)
  n = length(λ)
  return (fλ,
          ()->fback(ones(n))[1],
          ()->nothing, # TODO: add 2nd deriv
          isempty(args) ? _ -> () : f̄λ -> tail(fback(f̄λ)))
end

function _pullback_series_func_scalar(f::typeof(^), λ, p)
  compλ = _process_series_eigvals(f, λ)
  r, powλ = isinteger(p) ? (Integer(p), λ) : (p, compλ)
  fλ = powλ .^ r
  return (fλ,
          ()->conj.(r .* powλ .^ (r - 1)),
          ()->conj.((r * (r - 1)) .* powλ .^ (r - 2)),
          f̄λ -> (dot(fλ .* log.(compλ), f̄λ),))
end

function _pullback_series_func_scalar(f::typeof(exp), λ)
  expλ = exp.(λ)
  return expλ, ()->expλ, ()->expλ, _ -> ()
end

_apply_series_func(f, A, args...) = f(A, args...)

@adjoint function _apply_series_func(f, A, args...)
  hasargs = !isempty(args)
  n = LinearAlgebra.checksquare(A)
  λ, U = eigen(A)
  fλ, dfthunk, d²fthunk, argsback = _pullback_series_func_scalar(f, λ, args...)
  fΛ = Diagonal(fλ)
  fA = U * fΛ * U'
  Ω = _process_series_matrix(f, fA, A, fλ)
  return Ω, function (f̄A)
    f̄Λ = U' * f̄A * U
    ārgs = hasargs ? argsback(diag(f̄Λ)) : ()
    P = _pairdiffquotmat(f, n, λ, conj(fλ), dfthunk(), d²fthunk())
    Ā = U * (P .* f̄Λ) * U'
    return (nothing, Ā, ārgs...)
  end
end

_hermsympow(A::Symmetric, p::Integer) = LinearAlgebra.sympow(A, p)
_hermsympow(A::Hermitian, p::Integer) = A^p

@adjoint function _hermsympow(A::Hermitian, p::Integer)
  if p < 0
    B, back = Zygote.pullback(A->Base.power_by_squaring(inv(A), -p), A)
  else
    B, back = Zygote.pullback(A->Base.power_by_squaring(A, p), A)
  end
  Ω = Hermitian(_realifydiag!(B))
  return Ω, function (Ω̄)
    B̄ = _hermitian_back(Ω̄, 'U')
    Ā = back(B̄)[1]
    return (Ā, nothing)
  end
end

_pullback(cx::AContext, ::typeof(^), A::LinearAlgebra.HermOrSym{<:Real}, p::Integer) =
  _pullback(cx, _hermsympow, A, p)
_pullback(cx::AContext, ::typeof(^), A::Symmetric{<:Complex}, p::Integer) =
  _pullback(cx, _hermsympow, A, p)
_pullback(cx::AContext, ::typeof(^), A::Hermitian{<:Complex}, p::Integer) =
  _pullback(cx, _hermsympow, A, p)

function _pullback(cx::AContext,
                   f::typeof(^),
                   A::LinearAlgebra.RealHermSymComplexHerm,
                   p::Real)
  return _pullback(cx, (A, p) -> _apply_series_func(f, A, p), A, p)
end

for func in (:exp, :log, :cos, :sin, :tan, :cosh, :sinh, :tanh, :acos, :asin, :atan, :acosh, :asinh, :atanh, :sqrt)
  @eval begin
    function _pullback(cx::AContext,
                       f::typeof($func),
                       A::LinearAlgebra.RealHermSymComplexHerm)
      return _pullback(cx, A -> _apply_series_func(f, A), A)
    end
  end
end

@adjoint function sincos(A::LinearAlgebra.RealHermSymComplexHerm)
  n = LinearAlgebra.checksquare(A)
  λ, U = eigen(A)
  sλ, cλ = Buffer(λ), Buffer(λ)
  for i in Base.OneTo(n)
    @inbounds sλ[i], cλ[i] = sincos(λ[i])
  end
  sinλ, cosλ = copy(sλ), copy(cλ)
  sinA, cosA = U * Diagonal(sinλ) * U', U * Diagonal(cosλ) * U'
  Ω, processback = Zygote.pullback(sinA, cosA) do s,c
    return (_process_series_matrix(sin, s, A, λ),
            _process_series_matrix(cos, c, A, λ))
  end
  return Ω, function (Ω̄)
    s̄inA, c̄osA = processback(Ω̄)
    s̄inΛ, c̄osΛ = U' * s̄inA * U, U' * c̄osA * U
    PS = _pairdiffquotmat(sin, n, λ, sinλ, cosλ, -sinλ)
    PC = _pairdiffquotmat(cos, n, λ, cosλ, -sinλ, -cosλ)
    Ā = U * (PS .* s̄inΛ .+ PC .* c̄osΛ) * U'
    return (Ā,)
  end
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

@adjoint function Matrix(::UniformScaling, i::Integer, j::Integer)
  return Matrix(I, i, j), Δ -> ((λ=tr(Δ),), nothing, nothing)
end
@adjoint function Matrix(::UniformScaling, ij::NTuple{2, Integer})
  return Matrix(I, ij), Δ -> ((λ=tr(Δ),), nothing)
end
@adjoint function Matrix{T}(::UniformScaling, i::Integer, j::Integer) where {T}
  return Matrix{T}(I, i, j), Δ -> ((λ=tr(Δ),), nothing, nothing)
end
@adjoint function Matrix{T}(::UniformScaling, ij::NTuple{2, Integer}) where {T}
  return Matrix{T}(I, ij), Δ -> ((λ=tr(Δ),), nothing)
end
@adjoint function +(A::AbstractMatrix, S::UniformScaling)
  return A + S, Δ->(Δ, (λ=tr(Δ),))
end
@adjoint function -(S::UniformScaling, A::AbstractMatrix)
  return S - A, Δ->((λ=tr(Δ),), -Δ)
end

@adjoint +(A::AbstractArray, B::AbstractArray) = A + B, Δ->(Δ, Δ)
@adjoint -(A::AbstractArray, B::AbstractArray) = A - B, Δ->(Δ, -Δ)
@adjoint -(A::AbstractArray) = -A, Δ->(-Δ,)

# FFTW
# ===================

# FFTW functions do not work with FillArrays, which are needed
# for some functionality of Zygote. To make it work with FillArrays
# as well, overload the relevant functions
FFTW.fft(x::Fill, dims...) = FFTW.fft(collect(x), dims...)
FFTW.ifft(x::Fill, dims...) = FFTW.ifft(collect(x), dims...)


# the adjoint jacobian of an FFT with respect to its input is the reverse FFT of the
# gradient of its inputs, but with different normalization factor
@adjoint function FFTW.fft(xs)
  return FFTW.fft(xs), function(Δ)
    N = length(xs)
    return (N * FFTW.ifft(Δ),)
  end
end

@adjoint function FFTW.ifft(xs)
  return FFTW.ifft(xs), function(Δ)
    N = length(xs)
    return (1/N* FFTW.fft(Δ),)
  end
end

@adjoint function FFTW.fft(xs, dims)
  return FFTW.fft(xs, dims), function(Δ)
    # dims can be int, array or tuple,
    # convert to collection for use as index
    dims = collect(dims)
    # we need to multiply by all dimensions that we FFT over
    N = prod(collect(size(xs))[dims])
    return (N * FFTW.ifft(Δ, dims), nothing)
  end
end

@adjoint function FFTW.ifft(xs,dims)
  return FFTW.ifft(xs, dims), function(Δ)
    # dims can be int, array or tuple,
    # convert to collection for use as index
    dims = collect(dims)
    # we need to divide by all dimensions that we FFT over
    N = prod(collect(size(xs))[dims])
    return (1/N * FFTW.fft(Δ, dims),nothing)
  end
end

# FillArray functionality
# =======================

@adjoint function broadcasted(op, r::AbstractFill{<:Real})
  y, _back = Zygote.pullback(op, getindex_value(r))
  back(Δ::AbstractFill) = (nothing, Fill(_back(getindex_value(Δ))[1], size(r)))
  back(Δ::AbstractArray) = (nothing, getindex.(_back.(Δ), 1))
  return Fill(y, size(r)), back
end
