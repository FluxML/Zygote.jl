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
using Base.Broadcast: Broadcasted, AbstractArrayStyle, broadcasted, materialize,
  instantiate, flatten, combine_eltypes

# There's a saying that debugging code is about twice as hard as writing it in
# the first place. So if you're as clever as you can be when writing code, how
# will you ever debug it?

# AD faces a similar dilemma: if you write code that's as clever as the compiler
# can handle, how will you ever differentiate it? Differentiating makes clever
# code that bit more complex and the compiler gives up, usually resulting in
# 100x worse performance.

# Base's broadcasting is very cleverly written, and this makes differentiating
# it... somewhat tricky.

# Structural utilities
# ====================

using Base: tail

tcat(x) = x
tcat(x, y, z...) = tcat((x..., y...), z...)

broadcast_args(x) = (x,)
broadcast_args(bc::Broadcasted) = tcat(map(broadcast_args, bc.args)...)

_unflatten(x, xs) = first(xs), tail(xs)

_unflatten(x::Tuple{}, xs) = (), xs

function _unflatten(x::Tuple, xs)
  t1, xs1 = _unflatten(first(x), xs)
  t2, xs2 = _unflatten(tail(x), xs1)
  (t1, t2...), xs2
end

function _unflatten(bc::Broadcasted, xs)
  t, xs′ = _unflatten(bc.args, xs)
  (args=t,f=nothing,axes=nothing), xs′
end

unflatten(x, xs) = _unflatten(x, xs)[1]

unflatten(x, xs::Nothing) = nothing

accum_sum(xs; dims = :) = reduce(accum, xs, dims = dims)

# Work around reducedim_init issue
accum_sum(xs::AbstractArray{Nothing}; dims = :) = nothing
accum_sum(xs::AbstractArray{<:Real}; dims = :) = sum(xs, dims = dims)

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, accum_sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Union{Number,Ref}, Δ) = accum_sum(Δ)

# Trivial Mode
# ============

# In some cases, such as `exp.(a .+ b)`, we can see the the gradient only depends
# on the output. Handling these specially is great for performance and memory
# usage, though of course relatively limited. It happens that the set of cases
# lines up nicely with activation functions commonly used in neural nets, though.

# TODO fix this up and use it

Jtrivial(f, a...) = nothing
Jtrivial(::typeof(+), a...) = a
Jtrivial(::typeof(-), a, b) = (a..., .-b...)

trivia(_) = (1,)
function trivia(bc::Broadcasted)
  t = map(trivia, bc.args)
  any(t -> t === nothing, t) && return
  Jtrivial(bc.f, t...)
end

Joutput(f, a...) = nothing
Joutput(::typeof(exp), x) = map(t -> y -> y*t, x)

function Jbroadcast(bc::Broadcasted)
  t = map(trivia, bc.args)
  any(t -> t === nothing, t) && return
  Joutput(bc.f, t...)
end

@inline function unbroadcast_t(x, y, ȳ, j::J) where J
  trim(x, j.(y).*ȳ)
end

@inline function unbroadcast_t(x::Number, y, ȳ, j::J) where J
  x̄ = zero(float(x))
  @simd for I in eachindex(y)
    @inbounds x̄ += j(y[I])*ȳ[I]
  end
  return x̄
end

function ∇broadcast_t(bc::Broadcasted, J)
  y = copy(instantiate(bc))
  back(ȳ) = map(unbroadcast_t, broadcast_args(bc), map(_ -> y, J), map(_ -> ȳ, J), J)
  return y, back
end

# Forward Mode
# ============

# Forward mode has many limitations – mainly in that it only supports reals /
# arrays of reals and small numbers of inputs – but in those cases it works very
# generally across broadcasted functions, and handles loops particularly well.
# Most importantly it's easy on the compiler, so until we figure out reverse
# mode we're maintaining this implementation for common cases.

import ForwardDiff
using ForwardDiff: Dual

dualtype(::Type{Dual{G,T,P}}) where {G,T,P} = T
dualtype(T) = T

function dual_function(f::F) where F
  function (args::Vararg{Any,N}) where N
    ds = map(args, ntuple(identity,Val(N))) do x, i
      Dual(x, ntuple(j -> i==j, Val(N)))
    end
    return f(ds...)
  end
end

dualify(bc::Broadcasted{S}) where S = Broadcasted{S}(dual_function(bc.f), bc.args, bc.axes)

@inline function broadcast_gradient!(bc::Broadcasted, dest::AbstractArray, grads...)
  @simd for I in eachindex(bc)
    @inbounds begin
      out = bc[I]
      dest[I] = ForwardDiff.value(out)
      Δs = out isa Dual ? out.partials.values : map(_ -> false, grads)
      map((g, p) -> g[I] = p, grads, Δs)
    end
  end
end

function broadcast_gradient(bc::Broadcasted, ::Type{T}) where T
  dest = similar(bc, T)
  grads = map(_ -> similar(bc, promote_type(T,Bool)), bc.args)
  broadcast_gradient!(bc, dest, grads...)
  return dest, grads
end

@inline function ∇broadcast_f(bc′::Broadcasted)
  bc = dualify(instantiate(flatten(bc′)))
  T = combine_eltypes(bc.f, bc.args)
  T <: Bool && return copy(bc′), _ -> nothing
  y, gs = broadcast_gradient(bc, dualtype(T))
  back(Δ) = (unflatten(bc′, map((x, d) -> unbroadcast(x, Δ.*d), bc.args, gs)),)
  return y, back
end

function ∇broadcast_f(bc::Broadcasted{<:AbstractArrayStyle{0}})
  out = dualify(instantiate(flatten(bc)))[]
  return out.value, Δ -> (unflatten(bc, map(x -> x*Δ, out.partials.values)),)
end

# Compatibility test

isrealinput(x) = x isa Union{Real,AbstractArray{<:Real}}
isrealinput(bc::Broadcasted) = all(isrealinput, bc.args)

# Reverse Mode
# ============

# The fused reverse mode implementation is the most general but currently has
# poor performance. It works by flattening the broadcast and mapping the call to
# `_forward` over the input.

# However, the core call
# broadcast(_forward, (cx,), bc′.f, bc′.args...)
# is already 10x slower than a simple broadcast (presumably due to inlining
# issues, or something similar) and the other operations needed take it to about
# 100x overhead.

# One thing to experiment with would be a non-fused reverse mode, which is the
# more typical option for this kind of AD. While less efficient than fusing
# in principle, it's likely that this can be much easier on the compiler.

@generated inclen(::NTuple{N,Any}) where N = Val(N+1)

function ∇broadcast_r(cx, bc::Broadcasted)
  bc′, unflatten = _forward(cx, Broadcast.flatten, bc)
  len = inclen(bc′.args)
  y∂b = broadcast(_forward, (cx,), bc′.f, bc′.args...)
  y = map(x -> x[1], y∂b)
  ∂b = map(x -> x[2], y∂b)
  y, function (ȳ)
    dxs_zip = map((∂b, ȳ) -> ∂b(ȳ), ∂b, ȳ)
    dxs = ntuple(i -> map(x -> x[i], dxs_zip), len)
    (f = accum_sum(dxs[1]),
     args = map(unbroadcast, bc′.args, Base.tail(dxs)),
     axes = nothing) |> unflatten |> Base.tail
  end
end

function ∇broadcast_r(bc::Broadcasted{<:AbstractArrayStyle{0}})
  bc′, unflatten = _forward(Broadcast.flatten, bc)
  len = Val(length(bc′.args)+1)
  y, ∂b = broadcast(_forward, bc′.f, bc′.args...)
  y, function (ȳ)
    dxs = ∂b(ȳ)
    (f = dxs[1],
     args = Base.tail(dxs),
     axes = nothing) |> unflatten |> Base.tail
  end
end

∇broadcast(cx, bc::Broadcasted, J) = ∇broadcast_t(bc, J)

∇broadcast(cx, bc::Broadcasted, ::Nothing) =
  isrealinput(bc) ? ∇broadcast_f(bc) : ∇broadcast_r(cx, bc)

∇broadcast(cx, bc::Broadcasted) = ∇broadcast(cx, bc, Jbroadcast(bc))

@adjoint function broadcasted(f, args...)
  broadcasted(f, args...), Δ -> (nothing, Δ.args...)
end

@adjoint materialize(bc::Broadcasted{<:AbstractArrayStyle}) =
  ∇broadcast(__context__, bc, nothing)
