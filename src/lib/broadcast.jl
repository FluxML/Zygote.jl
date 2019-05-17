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
using Base.Broadcast: AbstractArrayStyle, broadcasted, materialize
using NNlib

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

unbroadcast(x::AbstractArray, x̄) =
  size(x) == size(x̄) ? x̄ :
  length(x) == length(x̄) ? trim(x, x̄) :
    trim(x, accum_sum(x̄, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(x̄)+1, Val(ndims(x̄)))))

unbroadcast(x::Union{Number,Ref}, x̄) = accum_sum(x̄)

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
    (nothing, accum_sum(dxs[1]), map(unbroadcast, args, Base.tail(dxs))...)
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
