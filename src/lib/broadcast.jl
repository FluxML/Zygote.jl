#                        .-'''-.                               _..._
#                       '   _    \         _______          .-'_..._''.
#  /|                 /   /` '.   \        \  ___ `'.     .' .'      '.\
#  ||                .   |     \  '         ' |--.\  \   / .'
#  ||        .-,.--. |   '      |  '        | |    \  ' . '                                     .|
#  ||  __    |  .-. |\    \     / /  __     | |     |  '| |                 __                .' |_
#  ||/'__ '. | |  | | `.   ` ..' /.:--.'.   | |     |  || |              .:--.'.         _  .'     |
#  |:/`  '. '| |  | |    '-...-'`/ |   \ |  | |     ' .'. '             / |   \ |      .' |'--.  .-'
#  ||     | || |  '-             `" __ | |  | |___.' /'  \ '.          .`" __ | |     .   | / |  |
#  ||\    / '| |                  .'.''| | /_______.'/    '. `._____.-'/ .'.''| |   .'.'| |// |  |
#  |/\'..' / | |                 / /   | |_\_______|/       `-.______ / / /   | |_.'.'.-'  /  |  '.'
#  '  `'-'`  |_|                 \ \._,\ '/                          `  \ \._,\ '/.'   \_.'   |   /
#                                 `--'  `"                               `--'  `"             `'-'

using Base.Broadcast
using Base.Broadcast: Broadcasted, AbstractArrayStyle, broadcasted, materialize

# There's a saying that debugging code is about twice as hard as writing it in
# the first place. So if you're as clever as you can be when writing code, how
# will you ever debug it?

# AD faces a similar dilemma: if you write code that's as clever as the compiler
# can handle, how will you ever differentiate it? Differentiating makes clever
# code that bit more complex and the compiler gives up, usually resulting in
# 100x worse performance.

# Base's broadcasting is very cleverly written, and this makes differentiating
# it... somewhat tricky.

# Utilities
# =========

# ChainRules already marks this non-differentiable,# But inference can still give up because of the Zygote -> CR wrapper layer.
# This has been desugared from the (deprecated) @nograd macro.
@inline function Zygote._pullback(::AContext, ::typeof(Broadcast.combine_styles), args...)
  dargs = ntuple(_ -> nothing, length(args) + 1)
  combine_styles_pullback(_) = dargs
  return Broadcast.combine_styles(args...), combine_styles_pullback
end

accum_sum(xs; dims = :) = reduce(accum, xs, dims = dims)

# Work around reducedim_init issue
# https://github.com/JuliaLang/julia/issues/31427
accum_sum(xs::Nothing; dims = :) = nothing
accum_sum(xs::AbstractArray{Nothing}; dims = :) = nothing
accum_sum(xs::AbstractArray{<:Number}; dims = :) = sum(xs, dims = dims)
accum_sum(xs::AbstractArray{<:AbstractArray{<:Number}}; dims = :) = sum(xs, dims = dims)
accum_sum(xs::Number; dims = :) = xs

# https://github.com/FluxML/Zygote.jl/issues/594
function Base.reducedim_init(::typeof(identity), ::typeof(accum), A::AbstractArray, region)
  Base.reducedim_initarray(A, region, nothing, Union{Nothing,eltype(A)})
end

function unbroadcast(x::AbstractArray, maybethunked_x̄)
  x̄ = unthunk_tangent(maybethunked_x̄)
  N = ndims(x̄)
  if length(x) == length(x̄)
    _project(x, x̄)  # ProjectTo handles reshape, offsets, structured matrices, row vectors
  else
    dims = ntuple(d -> size(x, d) == 1 ? d : ndims(x̄)+1, ndims(x̄))
    _project(x, accum_sum(x̄; dims = dims))
  end
end
unbroadcast(x::Number, x̄) = accum_sum(x̄)
unbroadcast(x::Tuple{<:Any}, x̄) = (accum_sum(x̄),)
unbroadcast(x::Base.RefValue, x̄) = (x=accum_sum(x̄),)
unbroadcast(x::Tuple, x̄) =  NTuple{length(x)}(length(x) == length(x̄) ? x̄ : accum_sum(x̄; dims=2:ndims(x̄))) # case length(x) > 1
unbroadcast(x::Tuple, x̄::Nothing) = nothing
# fixing issue #1184, not duplicate method, since the above allows for an empty tuple
unbroadcast(x::Tuple{<:Any}, x̄::Nothing) = nothing

unbroadcast(x::AbstractArray, x̄::Nothing) = nothing

# Split Reverse Mode
# ==================

# TODO: use DiffRules here. It's complicated a little by the fact that we need
# to do CSE, then broadcast-ify the expression so that the closure captures the
# right arrays.

@adjoint broadcasted(::typeof(+), xs::Numeric...) =
  broadcast(+, xs...), ȳ -> (nothing, map(x -> unbroadcast(x, ȳ), xs)...)

@adjoint broadcasted(::typeof(-), x::Numeric, y::Numeric) = x .- y,
  Δ -> (nothing, unbroadcast(x, Δ), _minus(unbroadcast(y, Δ)))
@adjoint broadcasted(::typeof(-), x::Numeric) = .-x,
  Δ -> (nothing, _minus(Δ))
_minus(Δ) = -Δ
_minus(::Nothing) = nothing

@adjoint broadcasted(::typeof(*), x::Numeric, y::Numeric) = x.*y,
   Δ -> (nothing, unbroadcast(x, Δ .* conj.(y)), unbroadcast(y, Δ .* conj.(x)))
@adjoint broadcasted(::typeof(*), x::Number, y::AbstractArray{<:Number}) =
  _pullback(__context__, *, x, y)  # this uses dot(y,Δ) instead of sum(Δ .* conj.(y))
@adjoint broadcasted(::typeof(*), x::AbstractArray{<:Number}, y::Number) =
  _pullback(__context__, *, x, y)

@adjoint function broadcasted(::typeof(/), x::Numeric, y::Numeric)
  res = x ./ y
  res, Δ -> (nothing, unbroadcast(x, Δ ./ conj.(y)), unbroadcast(y, .-Δ .* conj.(res ./ y)))
end
@adjoint broadcasted(::typeof(/), x::AbstractArray{<:Number}, y::Number) =
  _pullback(__context__, /, x, y)

# `_pow_grad` guards against spurious `NaN`s from `0 * Inf`: when the local
# derivative `df` is exactly zero (e.g. `x^2` at `x == 0`), the contribution is
# zero even if the incoming sensitivity `ȳ` is infinite or `NaN`, as happens for
# `sqrt.(x.^2)` at the cusp `x == 0` (see FluxML/Zygote.jl#1598).
_pow_grad(ȳ, df) = iszero(df) ? zero(ȳ * df) : ȳ * df

@adjoint function broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::Numeric, exp::Val{p}) where p
  y = Base.literal_pow.(^, x, exp)
  y, ȳ -> (nothing, nothing, _pow_grad.(ȳ, p .* conj.(x .^ (p - 1))), nothing)
end

@adjoint broadcasted(::typeof(identity), x::Numeric) = x, Δ -> (nothing, Δ)

@adjoint function broadcasted(::typeof(tanh), x::Numeric)
  y = tanh.(x)
  y, ȳ -> (nothing, ȳ .* conj.(1 .- y.^2))
end

@adjoint broadcasted(::typeof(conj), x::Numeric) =
  conj(x), z̄ -> (nothing, conj(z̄))

@adjoint broadcasted(::typeof(real), x::Numeric) =
  real(x), z̄ -> (nothing, real(z̄))

@adjoint broadcasted(::typeof(imag), x::Numeric) =
  imag.(x), z̄ -> (nothing, im .* real.(z̄))

@adjoint broadcasted(::typeof(abs2), x::Numeric) =
  abs2.(x), z̄ -> (nothing, 2 .* real.(z̄) .* x)

@adjoint function broadcasted(::typeof(+), a::AbstractArray{<:Number}, b::Bool)
  y = b === false ? a : a .+ b
  y, Δ -> (nothing, Δ, nothing)
end
@adjoint function broadcasted(::typeof(+), b::Bool, a::AbstractArray{<:Number})
  y = b === false ? a : b .+ a
  y, Δ -> (nothing, nothing, Δ)
end

@adjoint function broadcasted(::typeof(-), a::AbstractArray{<:Number}, b::Bool)
  y = b === false ? a : a .- b
  y, Δ -> (nothing, Δ, nothing)
end
@adjoint function broadcasted(::typeof(-), b::Bool, a::AbstractArray{<:Number})
  b .- a, Δ -> (nothing, nothing, .-Δ)
end

@adjoint function broadcasted(::typeof(*), a::AbstractArray{<:Number}, b::Bool)
  if b === false
    zero(a), Δ -> (nothing, zero(Δ), nothing)
  else
    a, Δ -> (nothing, Δ, nothing)
  end
end
@adjoint function broadcasted(::typeof(*), b::Bool, a::AbstractArray{<:Number})
  if b === false
    zero(a), Δ -> (nothing, nothing, zero(Δ))
  else
    a, Δ -> (nothing, nothing, Δ)
  end
end

@adjoint broadcasted(::Type{T}, x::Numeric) where {T<:Number} =
  T.(x), ȳ -> (nothing, _project(x, ȳ),)


# Fix https://github.com/FluxML/Zygote.jl/issues/1399 by ensuring we avoid a lazier CR rule
# https://github.com/JuliaDiff/ChainRules.jl/blob/5855c10bdbe691fc07822752f5b5865b9cea44d3/src/rulesets/Base/broadcast.jl#L199
@adjoint function broadcasted(::typeof(*), x::Numeric, y::Numeric, zs::Numeric...)
  y, back = _broadcast_generic(__context__, *, x, y, zs...)
  return y, Base.tail∘back
end

# General Fallback
# ================

# The fused reverse mode implementation is the most general but currently has
# poor performance. It works by flattening the broadcast and mapping the call to
# `_pullback` over the input.

# However, the core call
# broadcast(_pullback, (cx,), f, args...)
# is already 10x slower than a simple broadcast (presumably due to inlining
# issues, or something similar) and the other operations needed take it to about
# 100x overhead.

@generated inclen(::NTuple{N,Any}) where N = Val(N+1)

# Avoid hitting special cases for `Adjoint` etc.
_broadcast(f::F, x...) where F = materialize(broadcasted(f, x...))

collapse_nothings(xs::AbstractArray{Nothing}) = nothing
collapse_nothings(xs) = xs

_dual_purefun(::Type{F}) where {F<:Function} = Base.issingletontype(F)
_dual_purefun(::Type) = false
_dual_purefun(::Type{typeof(^)}) = false  # avoid DomainError from negative powers

_dual_safearg(x::Numeric{<:Real}) = true
_dual_safearg(x::Numeric{<:Complex}) = true
_dual_safearg(x::Ref{<:Numeric{<:Real}}) = true
_dual_safearg(x::Ref{<:Numeric{<:Complex}}) = true
_dual_safearg(x::Union{Type,Val,Symbol}) = true  # non-differentiable types
_dual_safearg(x) = false

@adjoint broadcasted(::AbstractArrayStyle, f::F, args...) where {F} = _broadcast_generic(__context__, f, args...)
@inline function _broadcast_generic(__context__, f::F, args...) where {F}
  T = Broadcast.combine_eltypes(f, args)
  # Avoid generic broadcasting in two easy cases:
  if T == Bool
    return (f.(args...), _ -> nothing)
  elseif T <: Union{Real, Complex} && isconcretetype(T) && _dual_purefun(F) && all(_dual_safearg, args) && !isderiving()
    return broadcast_forward(f, args...)
  end
  len = inclen(args)
  y∂b = _broadcast((x...) -> _pullback(__context__, f, x...), args...)
  y = broadcast(first, y∂b)
  function ∇broadcasted(ȳ)
    dxs_zip = map(((_, pb), ȳ₁) -> pb(ȳ₁), y∂b, ȳ)
    getters = ntuple(i -> StaticGetter{i}(), len)
    dxs = map(g -> collapse_nothings(map(g, dxs_zip)), getters)
    (nothing, accum_sum(dxs[1]), map(unbroadcast, args, Base.tail(dxs))...)
  end
  return y, ∇broadcasted
end

@adjoint function broadcasted(::AbstractArrayStyle{0}, f, args...)
  y, ∂b = _broadcast((x...) -> _pullback(__context__, f, x...), args...)
  function ∇broadcasted0(ȳ)
    dxs = ∂b(ȳ)
    dxs === nothing && return nothing
    (nothing, dxs...)
  end
  y, ∇broadcasted0
end

# Use the `map` adjoint in this special case, which is the same but applies
# pullbacks in reverse order.
# This leaves regular `broadcast` technically incorrect when the broadcasted
# function is stateful.
# Look, I'm not proud of it, but this is extremely rare in practice.
# @adjoint function broadcasted(f, x)
#   ∇map(__context__, f, x)
# end

@adjoint! (b::typeof(broadcast))(f, args...) = _pullback(__context__, broadcasted, f, args...)

# Forward Mode -- necessary for CUDA, also used as a fast path above

import ForwardDiff
using ForwardDiff: Dual, Partials, value, partials


# We do this because it ensures type stability so it compiles nicely on the gpu
# The val is needed for some type stability
# The Val{C} flag indicates whether any sibling argument is complex, so that
# all duals in a mixed real/complex broadcast get 2N partials.
# See https://github.com/FluxML/Zygote.jl/issues/1601
#     https://github.com/FluxML/Zygote.jl/issues/1461
@inline dual(x, i, ::Val{N}, ::Val{C}) where {N,C} = x
@inline dual(x::Bool, i, ::Val{N}, ::Val{false}) where {N} = x
@inline dual(x::Bool, i, ::Val{N}, ::Val{true}) where {N} = x
@inline dual(x::Real, i, ::Val{N}, ::Val{false}) where {N} = Dual(x, ntuple(==(i), N))
@inline dual(x::Real, i, ::Val{N}, ::Val{true}) where {N} = Dual(x, ntuple(==(i), 2N))
# For complex since ForwardDiff.jl doesn't play nicely with complex numbers we
# construct a Complex dual number and tag the real and imaginary parts separately
@inline function dual(x::Complex{T}, i, ::Val{N}, ::Val{C}) where {T,N,C}
    re_dual = Dual(real(x), ntuple(==(i), 2N))
    im_dual = Dual(imag(x), ntuple(==(N+i), 2N))
    return Complex(re_dual, im_dual)
end

_iscomplex(::Complex) = true
_iscomplex(_) = false

function dualize(args::Vararg{Any, N}) where {N}
    has_complex = Val(any(_iscomplex, args))
    ds = map(args, ntuple(identity, N)) do x, i
        return dual(x, i, Val(N), has_complex)
    end
    return ds
end

@inline function dual_function(f::F) where F
    function (args::Vararg{Any,N}) where N
      ds = dualize(args...)
      return f(ds...)
    end
  end

# `abs` is non-differentiable at the origin. ForwardDiff/DiffRules pick the
# subgradient `1` there for real inputs, and produce `NaN` for complex inputs
# (`abs(z) == hypot(re, im)`, whose derivative is `0/0` at the origin). Both
# disagree with the ChainRules convention used everywhere else in Zygote, which
# is `0`. Route the forward-mode broadcast of `abs` (the GPU `sum(abs, x)` path,
# and `abs.(x)` on the CPU fast path) through a NaN-safe definition that matches.
# See https://github.com/FluxML/Zygote.jl/issues/1529
@inline dual_function(::typeof(abs)) = _abs_dualsafe ∘ only ∘ dualize

@inline function _abs_dualsafe(d::Dual{T}) where {T}
    v = value(d)
    return Dual{T}(abs(v), sign(v) * partials(d))
end
@inline function _abs_dualsafe(z::Complex{<:Dual{T}}) where {T}
    dr, di = reim(z)
    vr, vi = value(dr), value(di)
    a = hypot(vr, vi)
    s = iszero(a) ? zero(a) : inv(a)  # subgradient 0 at the origin
    return Dual{T}(a, (vr * s) * partials(dr) + (vi * s) * partials(di))
end


# Does this (possibly nested) type contain a `ForwardDiff.Dual` anywhere?
_has_dual(::Type{<:Dual}) = true
_has_dual(::Type{<:Complex{<:Dual}}) = true
_has_dual(::Type{T}) where {T<:Tuple} = any(_has_dual, fieldtypes(T))
_has_dual(::Type) = false

# Recursively strip `ForwardDiff.Dual` values from a broadcast output element,
# descending into `Tuple`s (e.g. the result of `broadcast(tuple, x, y)`).
@inline _strip_dual(x::Dual) = value(x)
@inline _strip_dual(x::Complex{<:Dual}) = Complex(value(real(x)), value(imag(x)))
@inline _strip_dual(x::Tuple) = map(_strip_dual, x)

# Contract an incoming cotangent element `ȳ1` with the partials of the output
# element `o1` w.r.t. the i-th broadcast argument, descending into `Tuple`s.
@inline _dual_partial(ȳ1, o1::Dual, i) = ȳ1 * partials(o1, i)
@inline _dual_partial(ȳ1::Tuple, o1::Tuple, i) = mapreduce((y1, p1) -> _dual_partial(y1, p1, i), +, ȳ1, o1)
@inline _dual_partial(::Nothing, o1, i) = false  # missing cotangent component → zero contribution

@inline function broadcast_forward(f, args::Vararg{Any,N}) where N
  out = dual_function(f).(args...)
  T = eltype(out)
  if T <: Union{Dual, Complex{<:Dual}}
    if any(eltype(a) <: Complex for a in args)
      return _broadcast_forward_complex(T, out, args...)
    else
      return _broadcast_forward(T, out, args...)
    end
  elseif _has_dual(T)
    # The output element is a `Tuple` (possibly nested) of `Dual`s, as produced by
    # e.g. `broadcast(tuple, x, y)`. Strip the `Dual`s from the returned value and
    # contract the partials in the pullback, rather than leaking `Dual`s to the
    # caller and returning a `nothing` gradient. See FluxML/Zygote.jl#1424.
    return _broadcast_forward_tuple(out, args...)
  else
    return (out, _ -> nothing)
  end
end

# Real input, `Tuple`-of-`Dual` output (e.g. `broadcast(tuple, x, y)`).
@inline function _broadcast_forward_tuple(out, args::Vararg{Any, N}) where {N}
  valN = Val(N)
  y = broadcast(_strip_dual, out)
  function bc_fwd_back(ȳ)
    dargs = ntuple(valN) do i
      unbroadcast(args[i], broadcast((ȳ1, o1) -> _dual_partial(ȳ1, o1, i), ȳ, out))
    end
    (nothing, nothing, dargs...) # nothings for broadcasted & f
  end
  return y, bc_fwd_back
end

# Real input and real output pullback
@inline function _broadcast_forward(::Type{<:Dual}, out, args::Vararg{Any, N}) where {N}
  valN = Val(N)
  y = broadcast(x -> value(x), out)
  function bc_fwd_back(ȳ)
    dargs = ntuple(valN) do i
      unbroadcast(args[i], broadcast((y1, o1) -> y1 * partials(o1,i), ȳ, out))
    end
    (nothing, nothing, dargs...) # nothings for broadcasted & f
  end
  return y, bc_fwd_back
end

# This handles the complex output and real input pullback
@inline function _broadcast_forward(::Type{<:Complex}, out, args::Vararg{Any, N}) where {N}
    valN = Val(N)
    y = broadcast(x -> Complex(value(real(x)), value(imag(x))), out)
    function bc_fwd_back(ȳ)
      dargs = ntuple(valN) do i
        unbroadcast(args[i], broadcast((y1, o1) -> (real(y1)*partials(real(o1),i) + imag(y1)*partials(imag(o1), i)), ȳ, out))
      end
      (nothing, nothing, dargs...) # nothings for broadcasted & f
    end
    return y, bc_fwd_back
  end

# This handles complex input and real output. We use the gradient definition from ChainRules here
# since it agrees with what Zygote did for real(x).
@inline function _broadcast_forward_complex(::Type{<:Dual}, out, args::Vararg{Any, N}) where {N}
    valN = Val(N)
    y = broadcast(x -> value(x), out)
    function bc_fwd_back(ȳ)
      dargs = ntuple(valN) do i
        unbroadcast(args[i], broadcast((y1, o1) -> y1 * Complex(partials(o1, i), partials(o1, i+N)), ȳ, out))
      end
      (nothing, nothing, dargs...) # nothings for broadcasted & f
    end
    return y, bc_fwd_back
end

# # # This is for complex input and complex output
# If we assume that
# f(x + iy) = u(x,y) + iv(x,y)
# then we do the following for the adjoint
# Δu ∂u/∂x + Δv∂v/∂x + i(Δu∂u/∂y + Δv ∂v/∂y )
# this follows https://juliadiff.org/ChainRulesCore.jl/stable/maths/complex.html
function _adjoint_complex(N, Δz, df, i)
    Δu, Δv = reim(Δz)
    du, dv = reim(df)
    return Complex(Δu*partials(du, i) + Δv*partials(dv, i), Δu*partials(du, i+N) + Δv*partials(dv, i+N))
end

@inline function _broadcast_forward_complex(::Type{<:Complex}, out, args::Vararg{Any, N}) where {N}
    valN = Val(N)
    y = broadcast(x -> Complex(value(real(x)), value(imag(x))), out)
    function bc_fwd_back(ȳ)
      dargs = ntuple(valN) do i
        unbroadcast(args[i], broadcast((y1, o1) -> _adjoint_complex(N, y1, o1, i), ȳ, out))
      end
      (nothing, nothing, dargs...) # nothings for broadcasted & f
    end
    return y, bc_fwd_back
end

using GPUArraysCore  # replaces @require CUDA block, weird indenting to preserve git blame

       # Ordinary broadcasting calls broadcast_forward anyway when certain its' safe,
       # so perhaps this can be deleted? Possible edge case here:
       # https://github.com/FluxML/Zygote.jl/pull/1018#issuecomment-873629415
  @adjoint function broadcasted(::AbstractGPUArrayStyle, f, args...)
    # Bool-valued broadcasts (e.g. `x .> 3`) are non-differentiable and must be
    # evaluated on the plain values: running predicates like `>` through ForwardDiff
    # `Dual`s gives wrong results, because ForwardDiff (>= v1) compares Duals with a
    # total order that breaks value-ties using the partials. See
    # https://github.com/FluxML/Zygote.jl/issues/1597
    # This mirrors the `T == Bool` short-circuit in `_broadcast_generic` above.
    if Broadcast.combine_eltypes(f, args) == Bool
      return (f.(args...), _ -> nothing)
    end
    broadcast_forward(f, args...)
  end

  @adjoint (::Type{T})(xs::Array) where {T <: AbstractGPUArray} =
    T(xs), Δ -> (convert(Array, Δ), )

  # `Array`/`collect` move a GPU array to the host; their pullback must move the
  # cotangent back to the device. Otherwise it stays a CPU array and a later
  # operation that broadcasts it against the GPU primal (e.g. `A*x` in the
  # original report) hits scalar indexing or fails to compile. See #1305.
  # `copyto!(similar(xs, ...), Δ)` relocates without naming a specific GPU
  # backend and without scalar indexing.
  _gpu_cotangent(xs, Δ::AbstractGPUArray) = Δ
  _gpu_cotangent(xs, Δ::AbstractArray) = copyto!(similar(xs, eltype(Δ), size(Δ)), Δ)

  @adjoint Array(xs::AbstractGPUArray) = Array(xs), Δ -> (_gpu_cotangent(xs, Δ),)
  @adjoint collect(xs::AbstractGPUArray) = collect(xs), Δ -> (_gpu_cotangent(xs, Δ),)

  # Make sure sum(f, ::CuArray) uses broadcast through forward-mode defined above
  # Not the ChainRules.rrule which will use the Zygote.Context and thus not be GPU compatible
  # `AnyGPUArray` (rather than `AbstractGPUArray`) also covers GPU array wrappers
  # such as `view`/`SubArray`, `Adjoint` and `Transpose`, whose generic
  # `sum(f, x)` rrule otherwise scalar-indexes on the GPU. See #1498.
  function _pullback(cx::AContext, ::typeof(sum), f, xs::AnyGPUArray)
    res, back = _pullback(cx, (f, xs) -> sum(f.(xs)), f, xs)
    return res, back ∘ unthunk_tangent
  end
  function _pullback(cx::AContext, ::Core.kwftype(typeof(sum)), kws, ::typeof(sum), f,
                     xs::AnyGPUArray)
    @assert !haskey(kws, :init) # TODO add init support (julia 1.6)
    res, back = _pullback(cx, (f, xs) -> sum(f.(xs); kws...), f, xs)
    sum_gpuarray_kw_pullback(Δ) = (nothing, nothing, back(unthunk_tangent(Δ))...)
    return res, sum_gpuarray_kw_pullback
  end

  @adjoint function Base.convert(::Type{T}, xs::Array)  where {T<:AbstractGPUArray}
    Base.convert(T, xs), Δ -> (nothing, Base.convert(Array, Δ),)
  end

  pull_block_vert(sz, Δ::AbstractGPUArray, A::Number) = @allowscalar Δ[sz]

  # Accumulating cotangents `accum`ulates by broadcasting `+`. When one cotangent
  # is a host-backed structured array (e.g. the `Diagonal{<:Fill}` returned by
  # the `tr` adjoint) and the other is a GPU array, that broadcast runs on the
  # host and scalar-indexes the GPU array (`gradient(x -> sum(abs2,x) - tr(x), cu)`
  # in #1512). Move the host operand onto the device first, then accumulate there.
  _gpu_like(ref::AbstractGPUArray, y::AbstractGPUArray) = y
  _gpu_like(ref::AbstractGPUArray, y::AbstractArray) = copyto!(similar(ref, eltype(y), size(y)), collect(y))
  accum(x::AbstractGPUArray, y::AbstractGPUArray) = Base.broadcast_preserving_zero_d(accum, x, y)
  accum(x::AbstractGPUArray, y::AbstractArray) = accum(x, _gpu_like(x, y))
  accum(x::AbstractArray, y::AbstractGPUArray) = accum(_gpu_like(y, x), y)
