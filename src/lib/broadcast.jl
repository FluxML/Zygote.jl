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
using NNlib

@nograd Broadcast.combine_styles, Broadcast.result_style

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

accum_sum(xs; dims = :) = reduce(accum, xs, dims = dims)
accum_sum(xs::Tuple; dims = :) = reduce(accum, xs)

# Work around reducedim_init issue
accum_sum(xs::Nothing; dims = :) = nothing
accum_sum(xs::AbstractArray{Nothing}; dims = :) = nothing
accum_sum(xs::AbstractArray{<:Number}; dims = :) = sum(xs, dims = dims)
accum_sum(xs::Number; dims = :) = xs

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, x̄) =
  size(x) == size(x̄) ? x̄ :
  length(x) == length(x̄) ? trim(x, x̄) :
    trim(x, accum_sum(x̄, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(x̄)+1, Val(ndims(x̄)))))

unbroadcast(x::Union{Number,Ref}, x̄) = accum_sum(x̄)

unbroadcast(x::Union{AbstractArray, Tuple}, x̄::Nothing) = nothing

unbroadcast(x::NTuple{N}, x̄::NTuple{N}) where N = x̄

# Split Reverse Mode
# ==================

# TODO: use DiffRules here. It's complicated a little by the fact that we need
# to do CSE, then broadcast-ify the expression so that the closure captures the
# right arrays.

Numeric{T<:Number} = Union{T,AbstractArray{<:T}}

@adjoint broadcasted(::typeof(+), xs::Numeric...) =
  broadcast(+, xs...), ȳ -> (nothing, map(x -> unbroadcast(x, ȳ), xs)...)

@adjoint broadcasted(::typeof(*), x::Numeric, y::Numeric) = x.*y,
  z̄ -> (nothing, unbroadcast(x, z̄ .* conj.(y)), unbroadcast(y, z̄ .* conj.(x)))

@adjoint function broadcasted(::typeof(/), x::Numeric, y::Numeric)
  res = x ./ y
  res, Δ -> (nothing, unbroadcast(x, Δ ./ y), unbroadcast(y, -Δ .* res ./ y))
end

@adjoint function broadcasted(::typeof(σ), x::Numeric)
  y = σ.(x)
  y, ȳ -> (nothing, ȳ .* conj.(y .* (1 .- y)))
end

@adjoint function broadcasted(::typeof(tanh), x::Numeric)
  y = tanh.(x)
  y, ȳ -> (nothing, ȳ .* conj.(1 .- y.^2))
end

@adjoint broadcasted(::typeof(conj), x::Numeric) =
  conj.(x), z̄ -> (nothing, conj.(z̄))

@adjoint broadcasted(::typeof(real), x::Numeric) =
  real.(x), z̄ -> (nothing, real.(z̄))

@adjoint broadcasted(::typeof(imag), x::Numeric) =
  imag.(x), z̄ -> (nothing, im .* real.(z̄))

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

abstract type ArgPlaceholder end
struct FDiffable <: ArgPlaceholder end
struct NonFDiffable <: ArgPlaceholder end
struct ConstantArg <: ArgPlaceholder end

struct DualArg{P} <: ArgPlaceholder
  partials::P
end

struct FlatFunction{F, Args <: Tuple{Vararg{ArgPlaceholder}}} <: ArgPlaceholder
  f::F
  args::Args

  # This is to match the layout of `FlatFunction` to `Broadcasted` so
  # that the pullback would create the named tuple usable for
  # `Broadcasted`:
  axes::Nothing
end

FlatFunction(f, args) = FlatFunction(f, args, nothing)

@inline foldlargs(op, x) = x
@inline foldlargs(op, x1, x2, xs...) = foldlargs(op, op(x1, x2), xs...)

consume(::ArgPlaceholder, x, rest...) = x, rest
consume(d::DualArg, x, rest...) = Dual(x, d.partials), rest
@inline function consume(ff::FlatFunction, args0...)
  args, rest = foldlargs(((), args0), ff.args...) do (args, rest), g
    a, r = consume(g, rest...)
    ((args..., a), r)
  end
  return (ff.f(args...), rest)
end

@inline function (ff::FlatFunction)(args...)
  y, rest = consume(ff, args...)
  @assert length(rest) == 0
  return y
end

@inline isconstant(x) = isconstanttype(typeof(x))
@inline isconstanttype(::Type) = false
@inline isconstanttype(::Type{<:Union{
  # From `Broadcast.broadcastable`:
  Symbol,
  AbstractString,
  UndefInitializer,
  Nothing,
  RoundingMode,
  Missing,
  Val,
  Ptr,
  Regex,
  Type,
}}) = true

@inline ArgPlaceholder(x) =
  isconstant(x) ? ConstantArg() : NonFDiffable()
@inline ArgPlaceholder(::Union{Real, AbstractArray{<:Real}}) = FDiffable()
@inline ArgPlaceholder(bc::Broadcasted) =
  FlatFunction(bc.f, map(ArgPlaceholder, bc.args))

fdiffables(_) = 0
fdiffables(::FDiffable) = 1
@inline fdiffables(ff::FlatFunction) = sum(fdiffables, ff.args)

@inline fillpartials(ff::FlatFunction) = _fillpartials(Val(fdiffables(ff)), 1, ff)[1]

@inline function _fillpartials(npartials, i, ff::FlatFunction)
  args, i = foldlargs(((), i), ff.args...) do (args, i), x
    if x isa FlatFunction
      a, i = fillpartials(npartials, i, x)
      (args..., a), i
    elseif x isa FDiffable
      (args..., DualArg(ntuple(j -> i == j, npartials))), i + 1
    else
      (args..., x), i
    end
  end
  return FlatFunction(ff.f, args), i
end

isfdiffable(_) = false
isfdiffable(::FDiffable) = true
@inline isfdiffable(ff::FlatFunction) =
  all(isfdiffable, ff.args) && Base.issingletontype(typeof(ff.f))

@generated inclen(::NTuple{N,Any}) where N = Val(N+1)

bcstyle(::Broadcasted{Style}) where Style = Style

flatargs(x) = (x,)
flatargs(bc::Broadcasted) = foldlargs((), bc.args...) do args, x
  (args..., flatargs(x)...)
end

@inline function back_materialize(bc::Broadcasted, ȳ, out, i=1)
  args, i = foldlargs(((), i), bc.args...) do (args, i), x
    if x isa Broadcasted
      a, i = back_materialize(x, ȳ, out, i)
      (args..., a), i
    elseif ArgPlaceholder(x) isa FDiffable
      (args..., unbroadcast(x, ((a, b) -> a*b.partials[i]).(ȳ, out))), i + 1
    else
      (args..., nothing), i
    end
  end
  return (f=nothing, args=args, axes=nothing), i
end

function unflatten(bc, dargs)
  dbc, rest = _unflatten(bc, dargs...)
  @assert length(rest) == 0
  return dbc
end
@inline _unflatten(_, d, rest...) = d, rest
@inline function _unflatten(bc::Broadcasted, dargs0...)
  dargs, rest = foldlargs(((), dargs0), bc.args...) do (dargs, rest), x
    a, r = _unflatten(x, rest...)
    ((dargs..., a), r)
  end
  return (f=nothing, args=dargs, axes=nothing), rest
end

# Avoid hitting special cases for `Adjoint` etc.
_broadcast(f::F, x...) where F = materialize(broadcasted(f, x...))

_get(x::Tuple, i) = x[i]
_get(::Nothing, i) = nothing
collapse_nothings(xs::Vector{Nothing}) = nothing
collapse_nothings(xs) = xs

@adjoint function copy(bc::Broadcasted)
  ff0 = ArgPlaceholder(bc)
  if isfdiffable(ff0)
    ff = fillpartials(ff0)
    out = copy(Broadcasted{bcstyle(bc)}(ff, flatargs(bc), bc.axes))
    eltype(out) <: Dual || out <: Dual || return (out, _ -> nothing)
    y = map(x -> x.value, out)
    back(ȳ) = back_materialize(bc, ȳ, out)
    return y, back
  else
    y∂b = _broadcast((x...) -> _pullback(__context__, ff0, x...), flatargs(bc)...)
    y = map(x -> x[1], y∂b)
    ∂b = map(x -> x[2], y∂b)
    y, function (ȳ)
      dxs_zip = map((∂b, ȳ) -> ∂b(ȳ), ∂b, ȳ)
      fargs = flatargs(bc)
      len = inclen(fargs)
      dxs = collapse_nothings.(ntuple(i -> map(x -> _get(x, i), dxs_zip), len))
      dargs = map(unbroadcast, fargs, Base.tail(dxs))
      (accum(accum_sum(dxs[1]), unflatten(bc, dargs)),)
      # Note: relying on that `dxs[1]`, `ff0`, and `bc0` have same shape
    end
  end
end

@adjoint function broadcasted(::AbstractArrayStyle{0}, f, args...)
  len = inclen(args)
  y, ∂b = _broadcast((x...) -> _pullback(__context__, f, x...), args...)
  y, function (ȳ)
    dxs = ∂b(ȳ)
    (nothing, dxs...)
  end
end

@adjoint! (b::typeof(broadcast))(f, args...) = _pullback(__context__, broadcasted, f, args...)

# Forward Mode (mainly necessary for CUDA)

import ForwardDiff
using ForwardDiff: Dual

dual(x, p) = x
dual(x::Real, p) = Dual(x, p)

dualtype(::Type{Dual{G,T,P}}) where {G,T,P} = T
dualtype(T) = T

function dual_function(f::F) where F
  function (args::Vararg{Any,N}) where N
    ds = map(args, ntuple(identity,Val(N))) do x, i
      dual(x, ntuple(j -> i==j, Val(N)))
    end
    return f(ds...)
  end
end

@inline function broadcast_forward(f, args::Vararg{Any,N}) where N
  T = Broadcast.combine_eltypes(f, args)
  out = dual_function(f).(args...)
  eltype(out) <: Dual || return (out, _ -> nothing)
  y = map(x -> x.value, out)
  _back(ȳ, i) = unbroadcast(args[i], ((a, b) -> a*b.partials[i]).(ȳ, out))
  back(ȳ) = ntuple(i -> _back(ȳ, i), N)
  return y, back
end

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  @adjoint function broadcasted(::Broadcast.ArrayStyle{CuArrays.CuArray}, f, args...)
    y, back = broadcast_forward(f, args...)
    y, ȳ -> (nothing, nothing, back(ȳ)...)
  end
end
