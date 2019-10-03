using FillArrays, FFTW
using FillArrays: AbstractFill, getindex_value
using Base.Broadcast: broadcasted, broadcast_shape

@adjoint (::Type{T})(::UndefInitializer, args...) where T<:Array = T(undef, args...), Δ -> nothing

@adjoint Array(xs::AbstractArray) = Array(xs), ȳ -> (ȳ,)
@adjoint Array(xs::Array) = Array(xs), ȳ -> (ȳ,)

@nograd size, length, eachindex, Colon(), findfirst, findlast, findall, randn, ones, zeros, one, zero,
  print, println, any, all

@adjoint rand(dims::Integer...) = rand(dims...), _ -> nothing

@adjoint Base.vect(xs...) = Base.vect(xs...), Δ -> (Δ...,)

@adjoint copy(x::AbstractArray) = copy(x), ȳ -> (ȳ,)

# Array Constructors
@adjoint (::Type{T})(x::T) where T<:Array = T(x), ȳ -> (ȳ,)
@adjoint (::Type{T})(x::Number, sz) where {T <: Fill} = Fill(x, sz), Δ -> (sum(Δ), nothing)
@adjoint (::Type{T})(sz) where {T<:Zeros} = Zeros(sz), Δ->(nothing,)
@adjoint (::Type{T})(sz) where {T<:Ones} = Ones(sz), Δ->(nothing,)

_zero(xs::AbstractArray{<:Integer}) = fill!(similar(xs, float(eltype(xs))), false)
_zero(xs::AbstractArray{<:Number}) = zero(xs)
_zero(xs::AbstractArray) = Any[nothing for x in xs]

@adjoint function getindex(xs::AbstractArray, i...)
  xs[i...], function (Δ)
    Δ′ = _zero(xs)
    Δ′[i...] = Δ
    (Δ′, map(_ -> nothing, i)...)
  end
end

@adjoint! setindex!(xs::AbstractArray, x...) = setindex!(xs, x...),
  _ -> error("Mutating arrays is not supported")

for f in [push!, pop!, pushfirst!, popfirst!]
  @eval @adjoint! $f(xs::Vector, x...) =
    push!(xs, x...), _ -> error("Mutating arrays is not supported")
end

@adjoint function view(x::AbstractArray, inds...; kw...)
  view(x, inds...; kw...), dy -> begin
    dx = _zero(x)
    copyto!(view(dx, inds...; kw...), dy)
    (dx, map(_->nothing, inds)...)
  end
end

# General

@adjoint collect(x::Array) = collect(x), Δ -> (Δ,)

@adjoint fill(x::Real, dims...) = fill(x, dims...), Δ->(sum(Δ), map(_->nothing, dims)...)

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

@adjoint function prod(xs::AbstractArray{<:Number}; dims = :)
  if dims === (:)
    prod(xs), Δ -> (prod(xs) ./ xs .* Δ,)
  else
    prod(xs, dims = dims),
      Δ -> (reshape(.*(circshift.([reshape(xs, length(xs))], 1:length(xs)-1)...), size(xs)) .* Δ,)
  end
end

function _pullback(cx::AContext, ::typeof(prod), f, xs::AbstractArray)
  y, back = pullback(cx, (xs -> prod(f.(xs))), xs)
  y, ȳ -> (nothing, nothing, back(ȳ)...)
end

@adjoint function maximum(xs; dims = :)
  max, i = findmax(xs, dims = dims)
  max, function (Δ)
    Δ isa Real && abs(Δ) <= sqrt(eps(float(Δ))) && return nothing
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
  return inv(A), function (Δ)
    Ainv = inv(A)
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
  if abs(Δx) ≤ sqrt(eps(T))
    return d²fx === nothing ? dfx[i] : dfx[i] - Δx / 2 * d²fx[i]
  end
  Δfx = fx[i] - fx[j]
  return Δfx / Δx
end

# Adjoint based on the Theano implementation, which uses the differential as described
# in Brančík, "Matlab programs for matrix exponential function derivative evaluation"
@adjoint exp(A::AbstractMatrix) = exp(A), function(F̄)
  n = size(A, 1)
  E = eigen(A)
  w = E.values
  ew = exp.(w)
  Δeij = (i, j)->_pairdiffquot(exp, i, j, w, ew, ew, ew)
  X = Δeij.(Base.OneTo(n), Base.OneTo(n)')
  V = E.vectors
  VF = factorize(V)
  Ā = (V * ((VF \ F̄' * V) .* X) / VF)'
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

for func in (:exp, :cos, :sin, :tan, :cosh, :sinh, :tanh, :atan, :asinh, :atanh)
  @eval begin
    @adjoint function ($func)(A::LinearAlgebra.RealHermSymComplexHerm)
      λ,U = eigen(A)
      n = size(λ)[1]
      fλ, fback = Zygote.pullback(x->($func).(x), λ)
      B = U * Diagonal(fλ) * U'
      issym = isreal(B)
      if issym
        Ω = Symmetric(B)
      else
        for i in 1:n
            B[i,i] = real(B[i,i])
        end
        Ω = Hermitian(B)
      end
      return Ω, function (Ω̄)
        B̄ = issym ? _symmetric_back(Ω̄, 'U') : _hermitian_back(Ω̄, 'U')
        dfλ = fback(ones(n))[1]
        Δfij = (i, j)->_pairdiffquot($func, i, j, λ, conj(fλ), dfλ) # TODO: add 2nd deriv
        P = @inbounds Δfij.(Base.OneTo(n), Base.OneTo(n)')
        J = U' * B̄ * U
        return (U * (P .* J) * U',)
      end
    end
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

@adjoint function +(A::AbstractMatrix, S::UniformScaling)
  return A + S, Δ->(Δ, (λ=sum(view(Δ, diagind(Δ))),))
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
