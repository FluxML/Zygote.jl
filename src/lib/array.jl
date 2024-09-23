using Random, FillArrays, AbstractFFTs
using FillArrays: AbstractFill, getindex_value
using Base.Broadcast: broadcasted, broadcast_shape
using Distributed: pmap, AbstractWorkerPool
using LinearAlgebra: Diagonal, Hermitian, LowerTriangular, UpperTriangular
using LinearAlgebra: UnitLowerTriangular, UnitUpperTriangular

@adjoint Array(xs::AbstractArray) = Array(xs), ȳ -> (ȳ,)
@adjoint Array(xs::Array) = Array(xs), ȳ -> (ȳ,)

@adjoint copy(x::AbstractArray) = copy(x), ȳ -> (ȳ,)

@adjoint function collect(x::Tuple)
  collect_tuple_pullback(dy) = (Tuple(dy),)
  collect(x), collect_tuple_pullback
end

@adjoint function collect(x::NamedTuple{names}) where names
  collect_namedtuple_pullback(dy) = (NamedTuple{names}(Tuple(dy)),)
  collect(x), collect_namedtuple_pullback
end

@adjoint function collect(x::AbstractArray)
  collect_array_pullback(dy) = (dy,)
  collect(x), collect_array_pullback
end

@adjoint function collect(d::Dict)
  _keys = collect(keys(d))
  collect_dict_pullback(Δ) = (reconstruct_if_dict(Δ, _keys),)
  collect(d), collect_dict_pullback
end

# Array Constructors
@adjoint function (::Type{T})(x::Number, sz) where {T <: Fill}
    back(Δ::AbstractArray) = (sum(Δ), nothing)
    back(Δ::NamedTuple) = (Δ.value, nothing)
    return Fill(x, sz), back
end

@adjoint (::Type{T})(sz) where {T<:Zeros} = T(sz), Δ->(nothing,)
@adjoint (::Type{T})(sz) where {T<:Ones} = T(sz), Δ->(nothing,)

"""
    OneElement(val, ind, axes) <: AbstractArray

Extremely simple `struct` used for the gradient of scalar `getindex`.
"""
struct OneElement{T,N,I,A} <: AbstractArray{T,N}
  val::T
  ind::I
  axes::A
  OneElement(val::T, ind::I, axes::A) where {T<:Number, I<:NTuple{N,Int}, A<:NTuple{N,AbstractUnitRange}} where {N} = new{T,N,I,A}(val, ind, axes)
end
Base.size(A::OneElement) = map(length, A.axes)
Base.axes(A::OneElement) = A.axes
Base.getindex(A::OneElement{T,N}, i::Vararg{Int,N}) where {T,N} = ifelse(i==A.ind, A.val, zero(T))


_zero(xs::AbstractArray{<:Number}, T::Type{Nothing}) = fill!(similar(xs), zero(eltype(xs)))
_zero(xs::AbstractArray{<:Number}, T) = fill!(similar(xs, T), false)
_zero(xs::AbstractArray, T) = fill!(similar(xs, Union{Nothing, T}), nothing)

_droplike(dy, dxv) = dy
_droplike(dy::Union{LinearAlgebra.Adjoint, LinearAlgebra.Transpose}, dxv::AbstractVector) =
  dropdims(dy; dims=2)

@adjoint getindex(::Type{T}, xs...) where {T} = T[xs...], dy -> (nothing, dy...)

_throw_mutation_error(f, args...) = error("""
Mutating arrays is not supported -- called $f($(join(map(typeof, args), ", ")), ...)
This error occurs when you ask Zygote to differentiate operations that change
the elements of arrays in place (e.g. setting values with x .= ...)

Possible fixes:
- avoid mutating operations (preferred)
- or read the documentation and solutions for this error
  https://fluxml.ai/Zygote.jl/latest/limitations
""")

@adjoint! setindex!(xs::AbstractArray, x...) = setindex!(xs, x...),
  _ -> _throw_mutation_error(setindex!, xs)

@adjoint! copyto!(xs, args...) = copyto!(xs, args...),
  _ -> _throw_mutation_error(copyto!, xs)

for f in [push!, pop!, pushfirst!, popfirst!]
  @eval @adjoint! $f(x::AbstractVector, ys...) = $f(x, ys...),
    _ -> _throw_mutation_error($f, x)
end

# General

@adjoint collect(x::Array) = collect(x), Δ -> (Δ,)

@adjoint permutedims(xs) = permutedims(xs), Δ -> (permutedims(Δ),)

@adjoint permutedims(xs::AbstractVector) = permutedims(xs), Δ -> (vec(permutedims(Δ)),)

@adjoint permutedims(xs, dims) = permutedims(xs, dims),
  Δ -> (permutedims(Δ, invperm(dims)), nothing)

@adjoint PermutedDimsArray(xs, dims) = PermutedDimsArray(xs, dims),
  Δ -> (PermutedDimsArray(Δ, invperm(dims)), nothing)

@adjoint reshape(xs, dims...) = reshape(xs, dims...),
  Δ -> (reshape(Δ, size(xs)),map(_->nothing,dims)...)

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
(::StaticGetter{i})(::Nothing) where {i} = nothing
function _unzip(tuples, ::Val{N}) where {N}
  getters = ntuple(n -> StaticGetter{n}(), N)
  map(g -> map(g, tuples), getters)
end
function unzip(tuples)
  N = length(first(tuples))
  _unzip(tuples, Val(N))
end

# Reverse iteration order in ∇map, for stateful functions.
# This is also used by comprehensions, which do guarantee iteration order.
# Not done for pmap, presumably because all is lost if you are relying on its order.
_tryreverse(m, backs, Δ) = backs, Δ
_tryreverse(m::typeof(map), backs, Δ) = _reverse(backs), _reverse(Δ)

_tryreverse(m, x) = x
_tryreverse(m::typeof(map), x) = _reverse(x)

# Fallback
_reverse(x) = reverse(x)

# Known cases in the standard library on which `reverse` errors (issue #1393)
_reverse(x::LowerTriangular) = UpperTriangular(_reverse(parent(x)))
_reverse(x::UpperTriangular) = LowerTriangular(_reverse(parent(x)))
_reverse(x::UnitLowerTriangular) = UnitUpperTriangular(_reverse(parent(x)))
_reverse(x::UnitUpperTriangular) = UnitLowerTriangular(_reverse(parent(x)))
_reverse(x::Hermitian) = Hermitian(_reverse(x.data), x.uplo == 'U' ? :L : :U)
_reverse(x::Symmetric) = Symmetric(_reverse(x.data), x.uplo == 'U' ? :L : :U)

# With mismatched lengths, map stops early. With mismatched shapes, it makes a vector.
# So we keep axes(x) to restore gradient dx to its full length & correct shape.
_tryaxes(x) = (s = Base.IteratorSize(x); s isa Base.HasShape ? axes(x) : s isa Base.HasLength ? (Base.OneTo(length(x)),) : throw(ArgumentError("iterator size must be finite")))
_tryaxes(x::AbstractArray) = axes(x)
_tryaxes(x::Tuple) = Val(length(x))
_tryaxes(x::Number) = x
_restore(dx::AbstractArray{Nothing}, ax::Tuple) = similar(dx, ax)
_restore(dx, ax::Tuple) = axes(dx) == ax ? dx : reshape(vcat(dx, falses(prod(map(length, ax)) - length(dx))), ax)
_restore(dx, ::Val{N}) where {N} = ntuple(i -> get(dx,i,nothing), N)
_restore(dx, ::Number) = only(dx)

# Sometimes a pullback doesn't return a Tuple, but rather returns only a
# single nothing to say "all arguments have zero cotangent". This function is needed to
# account for that inside the pullback for map.
last_or_nothing(::Nothing) = nothing
last_or_nothing(x) = last(x)

for (mapfunc,∇mapfunc) in [(:map,:∇map),(:pmap,:∇pmap)]
  @eval function $∇mapfunc(cx, f::F, args::Vararg{Any, N}) where {F, N}
    ys_and_backs = $mapfunc((args...) -> _pullback(cx, f, args...), args...)
    ys = map(first, ys_and_backs)
    arg_ax = map(_tryaxes, args)
    function map_back(Δ)
      if Base.issingletontype(F) && length(args) == 1
        Δarg = $mapfunc(((_,pb), δ) -> last_or_nothing(pb(δ)), ys_and_backs, Δ) # No unzip needed
        (nothing, Δarg)
      elseif Base.issingletontype(F)
        # Ensures `f` is pure: nothing captured & no state.
        unzipped = _unzip($mapfunc(((_,pb), δ) -> tailmemaybe(pb(δ)), ys_and_backs, Δ), Val(N))
        Δargs = map(_restore, unzipped, arg_ax)
        (nothing, Δargs...)
      else
        # Apply pullbacks in reverse order. Needed for correctness if `f` is stateful.
        Δf_and_args_zipped = $mapfunc(((_,pb), δ) -> pb(δ), _tryreverse($mapfunc, ys_and_backs, Δ)...)
        Δf_and_args = _unzip(_tryreverse($mapfunc, Δf_and_args_zipped), Val(N + 1))
        Δf = reduce(accum, Δf_and_args[1]; init=nothing)
        Δargs = map(_restore, Δf_and_args[2:end], arg_ax)
        (Δf, Δargs...)
      end
    end
    map_back(::Nothing) = nothing
    return ys, map_back
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

function _pullback(cx::AContext, ::typeof(collect), g::Base.Generator)
  giter, _keys = collect_if_dict(g.iter) # map is not defined for dictionaries
  y, map_pullback = ∇map(cx, g.f, giter)

  collect_pullback(::Nothing) = nothing

  function collect_pullback(ȳ)
    f̄, x̄ = map_pullback(ȳ)
    x̄ = reconstruct_if_dict(x̄, _keys) # return a dictionary if needed
    (nothing, (f = f̄, iter = x̄),)
  end
  y, collect_pullback
end

collect_if_dict(x::Dict) = collect(x), collect(keys(x))
collect_if_dict(x) = x, nothing

reconstruct_if_dict(x̄, _keys::Nothing) = x̄

function reconstruct_if_dict(x̄, _keys)
  # This reverses `collect_if_dict`, which returns `_keys::Nothing` if x is not a Dict
  @assert x̄ isa AbstractVector{<:Union{Nothing, NamedTuple{(:first,:second)}}}
  # we don't compute gradients with respect to keys
  # @assert all(x -> x === nothing || x[1] == 0 || x[1] === nothing, x̄)
  d̄ = Dict(k => isnothing(x) ? nothing : x[2] for (x, k) in zip(x̄, _keys))
  return d̄
end

@adjoint iterate(r::UnitRange, i...) = iterate(r, i...), _ -> nothing

# Iterators

@adjoint function enumerate(xs)
  back(::AbstractArray{Nothing}) = nothing
  back(dy::NamedTuple{(:itr,)}) = tuple(dy.itr)
  back(diys::AbstractArray{Union{Nothing, T}}) where T = (map(x -> x === nothing ? x : last(x), diys),)
  back(diys) = (map(last, diys),)
  enumerate(xs), back
end

function _pullback(cx::AContext, ::Type{<:Iterators.Filter}, f, x)
  res, back = _pullback(cx, filter, f, collect(x))
  return res, back ∘ unthunk_tangent
end

_ndims(::Base.HasShape{d}) where {d} = d
_ndims(x) = Base.IteratorSize(x) isa Base.HasShape ? _ndims(Base.IteratorSize(x)) : 1

function productfunc(xs, dy)
  @assert length(first(dy)) == length(xs)
  ndim = map(Zygote._ndims, xs)
  cdim = cumsum((1, ndim[begin:end-1]...))
  getters = ntuple(n -> StaticGetter{n}(), length(xs))
  map(first(dy), xs, cdim, getters) do dyn, x, cd, getter
    dyn === nothing && return nothing
    nd = _ndims(x)
    dims = nd == 0 ? (:) : ntuple(i -> i<cd ? i : i+nd, Val(ndims(dy)-nd))
    init = map(zero, dyn) # allows for tuples, which accum can add:
    red = mapreduce(getter, accum, dy; dims, init)
    return _project(x, nd == 0 ? red : reshape(red, axes(x)))
  end
end

@adjoint function Iterators.product(xs...)
  product_pullback(::AbstractArray{Nothing}) = nothing
  product_pullback(dy::NamedTuple{(:iterators,)}) = dy.iterators
  product_pullback(dy::AbstractArray) = productfunc(xs, dy)
  Iterators.product(xs...), product_pullback
end

@adjoint function Base.collect(p::Base.Iterators.ProductIterator)
  collect_product_pullback(dy) = ((iterators=productfunc(p.iterators, dy),),)
  return collect(p), collect_product_pullback
end

function zipfunc(xs, dy)
  getters = ntuple(n -> StaticGetter{n}(), length(xs))
  map(xs, getters) do x, getter
    dx = map(getter, dy)
    _project(x, _restore(dx, _tryaxes(x)))
  end
end

@adjoint function Iterators.zip(xs...)
  zip_pullback(::AbstractArray{Nothing}) = nothing
  zip_pullback(dy::NamedTuple{(:is,)}) = dy.is
  zip_pullback(dy::AbstractArray) = zipfunc(xs, dy)
  Iterators.zip(xs...), zip_pullback
end

@adjoint function Base.collect(z::Base.Iterators.Zip)
  collect_zip_pullback(dy::AbstractArray) = ((is=zipfunc(z.is, dy),),)
  collect(z), collect_zip_pullback
end

takefunc(itr, dy) = _restore(dy, _tryaxes(itr))

@adjoint function Iterators.take(itr, n)
  take_pullback(::AbstractArray{Nothing}) = nothing
  take_pullback(dy::NamedTuple{(:xs,:n)}) = (dy.xs, dy.n)
  take_pullback(dy::NamedTuple{(:n,:xs)}) = (dy.xs, dy.n)
  take_pullback(dy::AbstractArray) = (takefunc(itr, dy), nothing)
  Iterators.take(itr, n), take_pullback
end

@adjoint function Base.collect(t::Iterators.Take)
    collect_take_pullback(dy) = ((xs=takefunc(t.xs, dy), n=nothing),)
    collect(t), collect_take_pullback
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

function _pullback(cx::AContext, ::typeof(prod), f, xs::AbstractArray)
  return _pullback(cx, (f, xs) -> prod(f.(xs)), f, xs)
end

@adjoint real(x::AbstractArray) = real(x), r̄ -> (real(r̄),)
@adjoint conj(x::AbstractArray) = conj(x), r̄ -> (conj(r̄),)
@adjoint imag(x::AbstractArray) = imag(x), ī -> (complex.(0, real.(ī)),)


# LinearAlgebra
# =============

@adjoint parent(x::LinearAlgebra.Adjoint) = parent(x), ȳ -> (LinearAlgebra.Adjoint(ȳ),)
@adjoint parent(x::LinearAlgebra.Transpose) = parent(x), ȳ -> (LinearAlgebra.Transpose(ȳ),)
@adjoint parent(x::LinearAlgebra.UpperTriangular) = parent(x), ȳ -> (LinearAlgebra.UpperTriangular(ȳ),)
@adjoint parent(x::LinearAlgebra.LowerTriangular) = parent(x), ȳ -> (LinearAlgebra.LowerTriangular(ȳ),)

function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end
_kron(a::AbstractVector, b::AbstractVector) = vec(_kron(reshape(a, :, 1), reshape(b, :, 1)))
_kron(a::AbstractVector, b::AbstractMatrix) = _kron(reshape(a, :, 1), b)
_kron(a::AbstractMatrix, b::AbstractVector) = _kron(a, reshape(b, :, 1))

function _pullback(cx::AContext, ::typeof(kron), a::AbstractVecOrMat, b::AbstractVecOrMat)
  res, back = _pullback(cx, _kron, a, b)
  return res, back ∘ unthunk_tangent
end

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
    B, back = _pullback(__context__, A -> Base.power_by_squaring(inv(A), -p), A)
  else
    B, back = _pullback(__context__, A -> Base.power_by_squaring(A, p), A)
  end
  Ω = Hermitian(_realifydiag!(B))
  return Ω, function (Ω̄)
    B̄ = _hermitian_back(Ω̄, 'U')
    Ā = last(back(B̄))
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
  # x is a square matrix checked by tr,
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

# FillArray functionality
# =======================

@adjoint function broadcasted(op, r::AbstractFill{<:Real})
  y, _back = _pullback(__context__, op, getindex_value(r))
  back(Δ::AbstractFill) = (nothing, Fill(last(_back(getindex_value(Δ))), size(r)))
  back(Δ::AbstractArray) = (nothing, last.(_back.(Δ)))
  return Fill(y, size(r)), back
end
