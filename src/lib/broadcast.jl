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

# Work around reducedim_init issue
accum_sum(xs::AbstractArray{Nothing}; dims = :) = nothing
accum_sum(xs::AbstractArray{<:Number}; dims = :) = sum(xs, dims = dims)
accum_sum(xs::Number; dims = :) = xs

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

_unbroadcast(x, x̄) = unbroadcast(x, x̄)
_unbroadcast(x, ::Nothing) = nothing

unbroadcast(x::AbstractArray, x̄) =
  size(x) == size(x̄) ? x̄ :
  length(x) == length(x̄) ? trim(x, x̄) :
    trim(x, accum_sum(x̄, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(x̄)+1, Val(ndims(x̄)))))

unbroadcast(x::Union{Number,Ref}, x̄) = accum_sum(x̄)

"""
    prefernothing.(Δ, broadcasting_expression)

Return `nothing` if `Δ` is `nothing` or array of `nothing`s without evaluating
`broadcasting_expression`.
"""
function prefernothing end
Broadcast.broadcasted(::typeof(prefernothing), Δ, bc) =
  if Δ === nothing || Δ isa AbstractArray{Nothing}
    nothing
  else
    # Otherwise, replace the entries of type `nothing` with `false` lazily:
    _zeronothing(Δ, bc)
  end
_zeronothing(Δ, bc::Broadcasted) =
  broadcasted(bc.f, map(x -> _zeronothing(Δ, x), bc.args)...)
_zeronothing(Δ, x) = Δ === x ? broadcasted(__zeronothing, Δ) : x
__zeronothing(::Nothing) = false
__zeronothing(x) = x

# Split Reverse Mode
# ==================

# TODO: use DiffRules here. It's complicated a little by the fact that we need
# to do CSE, then broadcast-ify the expression so that the closure captures the
# right arrays.

Numeric{T<:Number} = Union{T,AbstractArray{<:T}}

@adjoint broadcasted(::typeof(+), xs::Numeric...) =
  broadcast(+, xs...), ȳ -> (nothing, map(x -> _unbroadcast(x, ȳ), xs)...)

@adjoint broadcasted(::typeof(*), x::Numeric, y::Numeric) = x.*y,
  z̄ -> (nothing,
        _unbroadcast(x, prefernothing.(z̄, z̄ .* conj.(y))),
        _unbroadcast(y, prefernothing.(z̄, z̄ .* conj.(x))))

@adjoint function broadcasted(::typeof(/), x::Numeric, y::Numeric)
  res = x ./ y
  res, Δ -> (nothing,
             _unbroadcast(x, prefernothing.(Δ, Δ ./ y)),
             _unbroadcast(y, prefernothing.(Δ, .-Δ .* res ./ y)))
end

@adjoint function broadcasted(::typeof(σ), x::Numeric)
  y = σ.(x)
  y, ȳ -> (nothing, prefernothing.(ȳ, ȳ .* conj.(y .* (1 .- y))))
end

@adjoint function broadcasted(::typeof(tanh), x::Numeric)
  y = tanh.(x)
  y, ȳ -> (nothing, prefernothing.(ȳ, ȳ .* conj.(1 .- y.^2)))
end

@adjoint broadcasted(::typeof(conj), x::Numeric) =
  conj.(x), z̄ -> (nothing, prefernothing.(z̄, conj.(z̄)))

@adjoint broadcasted(::typeof(real), x::Numeric) =
  real.(x), z̄ -> (nothing, prefernothing.(z̄, real.(z̄)))

@adjoint broadcasted(::typeof(imag), x::Numeric) =
  imag.(x), z̄ -> (nothing, prefernothing.(z̄, im .* real.(z̄)))

# General Fallback
# ================

# The fused reverse mode implementation is the most general but currently has
# poor performance. It works by flattening the broadcast and mapping the call to
# `_forward` over the input.

# However, the core call
# broadcast(_forward, (cx,), f, args...)
# is already 10x slower than a simple broadcast (presumably due to inlining
# issues, or something similar) and the other operations needed take it to about
# 100x overhead.

@generated inclen(::NTuple{N,Any}) where N = Val(N+1)

# Avoid hitting special cases for `Adjoint` etc.
_broadcast(f::F, x...) where F = materialize(broadcasted(f, x...))

@adjoint function broadcasted(::AbstractArrayStyle, f, args...)
  len = inclen(args)
  y∂b = _broadcast((x...) -> _forward(__context__, f, x...), args...)
  y = map(x -> x[1], y∂b)
  ∂b = map(x -> x[2], y∂b)
  y, function (ȳ)
    dxs_zip = map((∂b, ȳ) -> ∂b(ȳ), ∂b, ȳ)
    dxs = ntuple(i -> map(x -> x[i], dxs_zip), len)
    (nothing, accum_sum(dxs[1]), map(_unbroadcast, args, Base.tail(dxs))...)
  end
end

@adjoint function broadcasted(::AbstractArrayStyle{0}, f, args...)
  len = inclen(args)
  y, ∂b = _broadcast((x...) -> _forward(__context__, f, x...), args...)
  y, function (ȳ)
    dxs = ∂b(ȳ)
    (nothing, dxs...)
  end
end

@adjoint! (b::typeof(broadcast))(f, args...) = _forward(__context__, broadcasted, f, args...)

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
  _back(ȳ, i) = _unbroadcast(args[i], ((a, b) -> a*b.partials[i]).(ȳ, out))
  back(ȳ) = ntuple(i -> _back(ȳ, i), N)
  return y, back
end

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  @adjoint function broadcasted(::Broadcast.ArrayStyle{CuArrays.CuArray}, f, args...)
    y, back = broadcast_forward(f, args...)
    y, ȳ -> (nothing, nothing, back(ȳ)...)
  end
end
