using Random, FillArrays, AbstractFFTs
using FillArrays: AbstractFill, getindex_value
using Base.Broadcast: broadcasted, broadcast_shape
using Distributed: pmap, AbstractWorkerPool

@adjoint (::Type{T})(::UndefInitializer, args...) where T<:Array = T(undef, args...), Δ -> nothing

@adjoint Array(xs::AbstractArray) = Array(xs), ȳ -> (ȳ,)
@adjoint Array(xs::Array) = Array(xs), ȳ -> (ȳ,)

@nograd ones, zeros, Base.OneTo, Colon(), one, zero, sizehint!

@adjoint Base.vect(xs...) = Base.vect(xs...), Δ -> (Δ...,)

@adjoint copy(x::AbstractArray) = copy(x), ȳ -> (ȳ,)

@adjoint collect(x::Tuple) = collect(x), dy -> (Tuple(dy),)
@adjoint collect(x::AbstractArray) = collect(x), dy -> (dy,)

# Array Constructors
@adjoint (::Type{T})(x::T) where T<:Array = T(x), ȳ -> (ȳ,)
@adjoint function (::Type{T})(x::Number, sz) where {T <: Fill}
    back(Δ::AbstractArray) = (sum(Δ), nothing)
    back(Δ::NamedTuple) = (Δ.value, nothing)
    return Fill(x, sz), back
end

@adjoint (::Type{T})(sz) where {T<:Zeros} = T(sz), Δ->(nothing,)
@adjoint (::Type{T})(sz) where {T<:Ones} = T(sz), Δ->(nothing,)

@adjoint getindex(x::AbstractArray, inds...) = x[inds...], ∇getindex(x, inds)

@adjoint view(x::AbstractArray, inds...) = view(x, inds...), ∇getindex(x, inds)

∇getindex(x::AbstractArray, inds) = dy -> begin
  if inds isa  NTuple{<:Any, Integer}
    dx = _zero(x, typeof(dy))
    dx[inds...] = dy
  else
    dx = _zero(x, eltype(dy))
    dxv = view(dx, inds...)
    dxv .= accum.(dxv, _droplike(dy, dxv))
  end
  return (dx, map(_->nothing, inds)...)
end

_zero(xs::AbstractArray{<:Number}, T::Type{Nothing}) = fill!(similar(xs), zero(eltype(xs)))
_zero(xs::AbstractArray{<:Number}, T) = fill!(similar(xs, T), false)
_zero(xs::AbstractArray, T) = fill!(similar(xs, Union{Nothing, T}), nothing)

_droplike(dy, dxv) = dy
_droplike(dy::Union{LinearAlgebra.Adjoint, LinearAlgebra.Transpose}, dxv::AbstractVector) =
  dropdims(dy; dims=2)

@adjoint getindex(::Type{T}, xs...) where {T} = T[xs...], dy -> (nothing, dy...)

@adjoint! setindex!(xs::AbstractArray, x...) = setindex!(xs, x...),
  _ -> error("Mutating arrays is not supported")

@adjoint! copyto!(args...) = copyto!(args...),
  _ -> error("Mutating arrays is not supported")

for f in [push!, pop!, pushfirst!, popfirst!]
  @eval @adjoint! $f(xs, x...) =
    push!(xs, x...), _ -> error("Mutating arrays is not supported")
end

# This is kind of bad, but at least we don't materialize the whole
# array. Prefer to use `Buffer`
# function _pullback(cx::Context, ::typeof(push!), xs::AbstractVector{<:AbstractArray}, x::AbstractArray{T}...) where T
@adjoint! function push!(xs::AbstractVector{<:AbstractArray}, x::AbstractArray{T}...) where T
  sz_xs = size.(xs)
  sz_x = size.(x)
  push!(xs, x...), Δ -> begin
    (Δ, map(x -> Ones{T}(x...), sz_x)...)
  end
end

@adjoint! function pop!(xs::AbstractVector{<:AbstractArray{T}}) where T
  sz_xs = size.(xs)
  op = pop!(xs)
  op, Δ -> begin
    ([Ones{T}(sz...) for sz in sz_xs], )
  end
end

# General

@adjoint collect(x::Array) = collect(x), Δ -> (Δ,)

@adjoint fill(x::Real, dims...) = fill(x, dims...), Δ->(sum(Δ), map(_->nothing, dims)...)

@adjoint function circshift(A, shifts)
  circshift(A, shifts), Δ -> (circshift(Δ, map(-, shifts)), nothing)
end

@adjoint function reverse(x::AbstractArray, args...; kwargs...)
  _reverse(t) = reverse(t, args...; kwargs...)
  _reverse(x), Δ->(_reverse(Δ), map(_->nothing, args)...)
end

@adjoint permutedims(xs) = permutedims(xs), Δ -> (permutedims(Δ),)

@adjoint permutedims(xs::AbstractVector) = permutedims(xs), Δ -> (vec(permutedims(Δ)),)

@adjoint permutedims(xs, dims) = permutedims(xs, dims),
  Δ -> (permutedims(Δ, invperm(dims)), nothing)

@adjoint PermutedDimsArray(xs, dims) = PermutedDimsArray(xs, dims),
  Δ -> (PermutedDimsArray(Δ, invperm(dims)), nothing)

@adjoint reshape(xs, dims...) = reshape(xs, dims...),
  Δ -> (reshape(Δ, size(xs)),map(_->nothing,dims)...)

@adjoint function hvcat(rows::Tuple{Vararg{Int}}, xs::Number...)
  hvcat(rows, xs...), ȳ -> (nothing, permutedims(ȳ)...)
end

pull_block_vert(sz, Δ, A::Number) = Δ[sz]
pull_block_vert(sz, Δ, A::AbstractVector) = Δ[sz-length(A)+1:sz]
pull_block_vert(sz, Δ, A::AbstractMatrix) = Δ[sz-size(A, 1)+1:sz, :]
@adjoint function vcat(A::Union{AbstractVector, AbstractMatrix, Number}...)
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
    catdims = Base.dims2cat(dims)
    dXs = map(Xs) do x
      move = ntuple(d -> (d<=length(catdims) && catdims[d]) ? size(x,d) : 0, ndims(Δ))
      x_in_Δ = ntuple(d -> (d<=length(catdims) && catdims[d]) ? (start[d]+1:start[d]+move[d]) : Colon(), ndims(Δ))
      start = start .+ move
      dx = Δ[x_in_Δ...]
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

@adjoint repeat(x::AbstractVector, m::Integer) =
   repeat(x, m), ȳ -> (dropdims(sum(reshape(ȳ, length(x), :); dims=2); dims=2), nothing)

@adjoint function repeat(x::AbstractVecOrMat, m::Integer, n::Integer=1)
   return repeat(x, m, n), function (ȳ)
      ȳ′ = reshape(ȳ, size(x,1), m, size(x,2), n)
      return reshape(sum(ȳ′; dims=(2,4)), size(x)), nothing, nothing
   end
end

@adjoint getindex(i::Int, j::Int) = i[j], _ -> nothing

struct StaticGetter{i} end
(::StaticGetter{i})(v) where {i} = v[i]
@generated function _unzip(tuples, ::Val{N}) where {N}
  Expr(:tuple, (:(map($(StaticGetter{i}()), tuples)) for i ∈ 1:N)...)
end
function unzip(tuples)
  N = length(first(tuples))
  _unzip(tuples, Val(N))
end

# Reverse iteration order when ∇map is applied to vector,
# needed for stateful functions.
# See https://github.com/FluxML/Flux.jl/issues/1209
# Should be generalized to abstract array, but reverse takes a dims keyword there
_tryreverse(m, backs, Δ) = backs, Δ
function _tryreverse(m::typeof(map), backs, Δ::Union{AbstractVector, Tuple})
  return reverse(backs), reverse(Δ)
end
_tryreverse(m, x) = x
_tryreverse(m::typeof(map), x::Union{AbstractVector, Tuple}) = reverse(x)

for (mapfunc,∇mapfunc) in [(:map,:∇map),(:pmap,:∇pmap)]
  @eval function $∇mapfunc(cx, f, args...)
    ys_and_backs = $mapfunc((args...) -> _pullback(cx, f, args...), args...)
    if isempty(ys_and_backs)
      ys_and_backs, _ -> nothing
    else
      ys, backs = unzip(ys_and_backs)
      ys, function (Δ)
        # Apply pullbacks in reverse order. Needed for correctness if `f` is stateful.
        Δf_and_args_zipped = $mapfunc((f, δ) -> f(δ), _tryreverse($mapfunc, backs, Δ)...)
        Δf_and_args = unzip(_tryreverse($mapfunc, Δf_and_args_zipped))
        Δf = reduce(accum, Δf_and_args[1])
        (Δf, Δf_and_args[2:end]...)
      end
    end
  end

  @eval @adjoint function $mapfunc(f, args::Union{AbstractArray,Tuple}...)
    $∇mapfunc(__context__, f, args...)
  end
end

@adjoint function pmap(f, wp::AbstractWorkerPool, args...; kwargs...)
  ys_backs = pmap((x...) -> _pullback(__context__, f, x...), wp, args...; kwargs...)
  ys, backs = unzip(ys_backs)
  ys, function (Δ)
    res = pmap((df,d) -> df(d), wp, backs, Δ; kwargs...)
    Δf_and_args = unzip(res)
    Δf = reduce(accum, Δf_and_args[1])
    (Δf, nothing, Δf_and_args[2:end]..., nothing, nothing)
  end
end

for t in subtypes(AbstractWorkerPool)
  @nograd t
end
@nograd workers

function _pullback(cx::AContext, ::typeof(collect), g::Base.Generator)
  y, back = ∇map(cx, g.f, g.iter)
  y, function (ȳ)
    f̄, x̄ = back(ȳ)
    (nothing, (f = f̄, iter = x̄),)
  end
end

@adjoint iterate(r::UnitRange, i...) = iterate(r, i...), _ -> nothing

@adjoint function sort(x::AbstractArray; by=identity)
  p = sortperm(x, by=by)
  return x[p], x̄ -> (x̄[invperm(p)],)
end

@adjoint function filter(f, x::AbstractVector)
    t = map(f, x)
    x[t], Δ -> begin
        dx = _zero(x, eltype(Δ))
        dx[t] .= Δ
        (nothing, dx)
    end
end

# Reductions
@adjoint function sum(xs::AbstractArray; dims = :)
  if dims === (:)
    sum(xs), Δ -> (Fill(Δ, size(xs)),)
  else
    sum(xs, dims = dims), Δ -> (similar(xs) .= Δ,)
  end
end

@adjoint function sum(xs::AbstractArray{Bool}; dims = :)
  sum(xs, dims = dims), Δ -> (nothing,)
end

_normalize_kws(kws::NamedTuple) = kws
_normalize_kws(kws) = NamedTuple()

function _pullback(cx::AContext, kwtype, kws, ::typeof(sum), f, xs::AbstractArray)
  norm_kws = _normalize_kws(kws)
  @assert !haskey(norm_kws, :init) # TODO add init support (julia 1.6)
  y, back = pullback(cx, (f, xs) -> sum(f.(xs); norm_kws...), f, xs)
  y, ȳ -> (nothing, nothing, nothing, back(ȳ)...)
end

@adjoint function sum(::typeof(abs2), X::AbstractArray; dims = :)
  return sum(abs2, X; dims=dims), Δ::Union{Number, AbstractArray}->(nothing, ((2Δ) .* X))
end

@adjoint function prod(xs::AbstractArray; dims = :)
  p = prod(xs; dims = dims)
  p, Δ -> (p ./ xs .* Δ,)
end

function _pullback(cx::AContext, ::typeof(prod), f, xs::AbstractArray)
  y, back = pullback(cx, ((f, xs) -> prod(f.(xs))), f, xs)
  y, ȳ -> (nothing, back(ȳ)...)
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
@adjoint imag(x::AbstractArray) = imag(x), ī -> (complex.(0, real.(ī)),)

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

@adjoint function cumsum(xs::AbstractVector; dims::Integer = 1)
  dims == 1 || return copy(xs), Δ -> (Δ,)
  cumsum(xs), Δ -> (reverse(cumsum(reverse(Δ))),)
end
@adjoint function cumsum(xs::AbstractArray; dims::Integer)
  dims <= ndims(xs) || return copy(xs), Δ -> (Δ,)
  cumsum(xs; dims=dims), Δ -> begin
    (reverse(cumsum(reverse(Δ, dims=dims), dims=dims), dims=dims),)
  end
end

@adjoint eachrow(x::AbstractVecOrMat) = collect(eachrow(x)), dys -> ∇eachslice(dys, x, 1)
@adjoint eachcol(x::AbstractVecOrMat) = collect(eachcol(x)), dys -> ∇eachslice(dys, x, 2)
@adjoint eachslice(x::AbstractArray; dims::Integer) =
  collect(eachslice(x; dims=dims)), dys -> ∇eachslice(dys, x, dims)

function ∇eachslice(dys, x::AbstractArray, dim::Integer) where {TX}
  i1 = findfirst(dy -> dy isa AbstractArray, dys)
  i1 === nothing && return (zero(x),) # all slices get nothing
  T = promote_type(eltype(dys[i1]), eltype(x))
  dx = similar(x, T)
  for i in axes(x, dim)
    if dys[i] isa AbstractArray
      copyto!(selectdim(dx,dim,i), dys[i])
    else
      selectdim(dx,dim,i) .= 0
    end
  end
  (dx,)
end


# LinearAlgebra
# =============

@adjoint function transpose(x)
  back(Δ) = (transpose(Δ),)
  back(Δ::NamedTuple{(:parent,)}) = (Δ.parent,)
  return transpose(x), back
end

@adjoint function LinearAlgebra.Transpose(x)
  back(Δ) = (LinearAlgebra.Transpose(Δ),)
  back(Δ::NamedTuple{(:parent,)}) = (Δ.parent,)
  return LinearAlgebra.Transpose(x), back
end


@adjoint function Base.adjoint(x)
  back(Δ) = (Δ',)
  back(Δ::NamedTuple{(:parent,)}) = (Δ.parent,)
  return x', back
end

@adjoint function LinearAlgebra.Adjoint(x)
  back(Δ) = (LinearAlgebra.Adjoint(Δ),)
  back(Δ::NamedTuple{(:parent,)}) = (Δ.parent,)
  return LinearAlgebra.Adjoint(x), back
end

@adjoint parent(x::LinearAlgebra.Adjoint) = parent(x), ȳ -> (LinearAlgebra.Adjoint(ȳ),)
@adjoint parent(x::LinearAlgebra.Transpose) = parent(x), ȳ -> (LinearAlgebra.Transpose(ȳ),)

function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end

@adjoint kron(a::AbstractMatrix, b::AbstractMatrix) = pullback(_kron, a, b)

@adjoint logabsdet(xs::AbstractMatrix) = logabsdet(xs), Δ -> (Δ[1] * inv(xs)',)

@adjoint function inv(A::Union{Number, AbstractMatrix})
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
  return Y, function(Ȳ)
    B̄ = A' \ Ȳ
    return (-B̄ * Y', B̄)
  end
end

@adjoint function /(A::AbstractMatrix, B::Union{Diagonal, AbstractTriangular})
  Y = A / B
  return Y, function(Ȳ)
    Ā = Ȳ / B'
    return (Ā, -Y' * Ā)
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

# LinAlg Matrix Types
# ===================

@adjoint LinearAlgebra.LowerTriangular(A) = LowerTriangular(A), Δ->(LowerTriangular(Δ),)
@adjoint LinearAlgebra.UpperTriangular(A) = UpperTriangular(A), Δ->(UpperTriangular(Δ),)
@adjoint LinearAlgebra.UnitLowerTriangular(A) = UnitLowerTriangular(A), Δ->(UnitLowerTriangular(Δ)-I,)
@adjoint LinearAlgebra.UnitUpperTriangular(A) = UnitUpperTriangular(A), Δ->(UnitUpperTriangular(Δ)-I,)

# This is basically a hack while we don't have a working `ldiv!`.
@adjoint function \(A::Cholesky, B::AbstractVecOrMat)
  Y, back = Zygote.pullback((U, B)->U \ (U' \ B), A.U, B)
  return Y, function(Ȳ)
    Ā_factors, B̄ = back(Ȳ)
    return ((uplo=nothing, info=nothing, factors=Ā_factors), B̄)
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
  ŪL̄ = Δ .- Diagonal(_extract_imag(diag(Δ)))
  if istriu(Δ)
    return collect(uplo == 'U' ? ŪL̄ : ŪL̄')
  else
    return collect(uplo == 'U' ? ŪL̄' : ŪL̄)
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

@adjoint function cholesky(Σ::Diagonal; check = true)
  C = cholesky(Σ, check = check)
  return C, Δ::NamedTuple -> begin
    issuccess(C) || throw(PosDefException(C.info))
    return Diagonal(diag(Δ.factors) .* inv.(2 .* C.factors.diag)), nothing
  end
end

# Implementation due to Seeger, Matthias, et al. "Auto-differentiating linear algebra."
@adjoint function cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}}; check = true)
  C = cholesky(Σ, check = check)
  return C, function(Δ::NamedTuple)
    issuccess(C) || throw(PosDefException(C.info))
    U, Ū = C.U, Δ.factors
    Σ̄ = similar(U.data)
    Σ̄ = mul!(Σ̄, Ū, U')
    Σ̄ = copytri!(Σ̄, 'U')
    Σ̄ = ldiv!(U, Σ̄)
    Σ̄ = BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)
    Σ̄[diagind(Σ̄)] ./= 2
    return (UpperTriangular(Σ̄),)
  end
end

@adjoint function lyap(A::AbstractMatrix, C::AbstractMatrix)
  X = lyap(A, C)
  return X, function (X̄)
    C̄ = lyap(collect(A'), X̄)
    Ā = C̄*X' + C̄'*X
    return (Ā, C̄)
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

_hasrealdomain(::typeof(^), x) = all(x -> x ≥ 0, x)

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
function _pullback_series_func_scalar(f::typeof(^), λ, p)
  compλ = _process_series_eigvals(f, λ)
  r, powλ = isinteger(p) ? (Integer(p), λ) : (p, compλ)
  fλ = powλ .^ r
  return (fλ,
          ()->conj.(r .* powλ .^ (r - 1)),
          ()->conj.((r * (r - 1)) .* powλ .^ (r - 2)),
          f̄λ -> (dot(fλ .* log.(compλ), f̄λ),))
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
    ārgs = hasargs ? argsback(diag(f̄Λ)) : ()
    P = _pairdiffquotmat(f, n, λ, conj(fλ), dfthunk(), d²fthunk())
    Ā = U * (P .* f̄Λ) * U'
    return (nothing, Ā, ārgs...)
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
    Ā = back(B̄)[1]
    return (Ā, nothing)
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

# ChainRules has this also but does not use FillArrays, so we have our own definition
# for improved performance. See https://github.com/JuliaDiff/ChainRules.jl/issues/46
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

@adjoint function Matrix(S::UniformScaling, i::Integer, j::Integer)
  return Matrix(S, i, j), Δ -> ((λ=tr(Δ),), nothing, nothing)
end
@adjoint function Matrix(S::UniformScaling, ij::NTuple{2, Integer})
  return Matrix(S, ij), Δ -> ((λ=tr(Δ),), nothing)
end
@adjoint function Matrix{T}(S::UniformScaling, i::Integer, j::Integer) where {T}
  return Matrix{T}(S, i, j), Δ -> ((λ=tr(Δ),), nothing, nothing)
end
@adjoint function Matrix{T}(S::UniformScaling, ij::NTuple{2, Integer}) where {T}
  return Matrix{T}(S, ij), Δ -> ((λ=tr(Δ),), nothing)
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

# Abstract FFT
# ===================

# AbstractFFTs functions do not work with FillArrays, which are needed
# for some functionality of Zygote. To make it work with FillArrays
# as well, overload the relevant functions
AbstractFFTs.fft(x::Fill, dims...) = AbstractFFTs.fft(collect(x), dims...)
AbstractFFTs.bfft(x::Fill, dims...) = AbstractFFTs.bfft(collect(x), dims...)
AbstractFFTs.ifft(x::Fill, dims...) = AbstractFFTs.ifft(collect(x), dims...)
AbstractFFTs.rfft(x::Fill, dims...) = AbstractFFTs.rfft(collect(x), dims...)
AbstractFFTs.irfft(x::Fill, d, dims...) = AbstractFFTs.irfft(collect(x), d, dims...)
AbstractFFTs.brfft(x::Fill, d, dims...) = AbstractFFTs.brfft(collect(x), d, dims...)

# the adjoint jacobian of an FFT with respect to its input is the reverse FFT of the
# gradient of its inputs, but with different normalization factor
@adjoint function fft(xs)
  return AbstractFFTs.fft(xs), function(Δ)
    return (AbstractFFTs.bfft(Δ),)
  end
end

@adjoint function *(P::AbstractFFTs.Plan, xs)
  return P * xs, function(Δ)
    N = prod(size(xs)[[P.region...]])
    return (nothing, N * (P \ Δ))
  end
end

@adjoint function \(P::AbstractFFTs.Plan, xs)
  return P \ xs, function(Δ)
    N = prod(size(Δ)[[P.region...]])
    return (nothing, (P * Δ)/N)
  end
end

# all of the plans normalize their inverse, while we need the unnormalized one.
@adjoint function ifft(xs)
  return AbstractFFTs.ifft(xs), function(Δ)
    N = length(xs)
    return (AbstractFFTs.fft(Δ)/N,)
  end
end

@adjoint function bfft(xs)
  return AbstractFFTs.bfft(xs), function(Δ)
    return (AbstractFFTs.fft(Δ),)
  end
end

@adjoint function fftshift(x)
    return fftshift(x), function(Δ)
        return (ifftshift(Δ),)
    end
end

@adjoint function ifftshift(x)
    return ifftshift(x), function(Δ)
        return (fftshift(Δ),)
    end
end


# to actually use rfft, one needs to insure that everything
# that happens in the Fourier domain could've been done in
# the space domain with real numbers. This means enforcing
# conjugate symmetry along all transformed dimensions besides
# the first. Otherwise this is going to result in *very* weird
# behavior.
@adjoint function rfft(xs::AbstractArray{<:Real})
  return AbstractFFTs.rfft(xs), function(Δ)
    N = length(Δ)
    originalSize = size(xs,1)
    return (AbstractFFTs.brfft(Δ, originalSize),)
  end
end

@adjoint function irfft(xs, d)
  return AbstractFFTs.irfft(xs, d), function(Δ)
    total = length(Δ)
    fullTransform = AbstractFFTs.rfft(real.(Δ))/total
    return (fullTransform, nothing)
  end
end

@adjoint function brfft(xs, d)
  return AbstractFFTs.brfft(xs, d), function(Δ)
    fullTransform = AbstractFFTs.rfft(real.(Δ))
    return (fullTransform, nothing)
  end
end


# if we're specifying the dimensions
@adjoint function fft(xs, dims)
  return AbstractFFTs.fft(xs, dims), function(Δ)
    # dims can be int, array or tuple,
    # convert to collection for use as index
    dims = collect(dims)
    return (AbstractFFTs.bfft(Δ, dims), nothing)
  end
end

@adjoint function bfft(xs, dims)
  return AbstractFFTs.ifft(xs, dims), function(Δ)
    dims = collect(dims)
    return (AbstractFFTs.fft(Δ, dims),nothing)
  end
end

@adjoint function ifft(xs, dims)
  return AbstractFFTs.ifft(xs, dims), function(Δ)
    dims = collect(dims)
    N = prod(collect(size(xs))[dims])
    return (AbstractFFTs.fft(Δ, dims)/N,nothing)
  end
end

@adjoint function rfft(xs, dims)
  return AbstractFFTs.rfft(xs, dims), function(Δ)
    dims = collect(dims)
    N = prod(collect(size(xs))[dims])
    return (N * AbstractFFTs.irfft(Δ, size(xs,dims[1]), dims), nothing)
  end
end

@adjoint function irfft(xs, d, dims)
  return AbstractFFTs.irfft(xs, d, dims), function(Δ)
    dims = collect(dims)
    N = prod(collect(size(xs))[dims])
    return (AbstractFFTs.rfft(real.(Δ), dims)/N, nothing, nothing)
  end
end
@adjoint function brfft(xs, d, dims)
  return AbstractFFTs.brfft(xs, d, dims), function(Δ)
    dims = collect(dims)
    return (AbstractFFTs.rfft(real.(Δ), dims), nothing, nothing)
  end
end


@adjoint function fftshift(x, dims)
    return fftshift(x), function(Δ)
        return (ifftshift(Δ, dims), nothing)
    end
end

@adjoint function ifftshift(x, dims)
    return ifftshift(x), function(Δ)
        return (fftshift(Δ, dims), nothing)
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
