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
using Base.Broadcast: Broadcasted, AbstractArrayStyle, DefaultArrayStyle, broadcasted,
  instantiate, materialize, flatten, combine_eltypes, _broadcast_getindex

# Structural utilities

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

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

# Compute which dimensions were broadcast
@Base.pure broadcasted_dims(sz, sΔ) = tuple(filter(i->i > length(sz) || sz[i] != sΔ[i], 1:length(sΔ))...)

function unbroadcast(x::AbstractArray, Δ)
  res = size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, Base.sum(Δ, dims = broadcasted_dims(size(x), size(Δ))))
  res
end

unbroadcast(x::Union{Number,Ref}, Δ) = sum(Δ)

# Reverse Mode

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

@inline function unbroadcast_r(x, y, ȳ, j::J) where J
  trim(x, j.(y).*ȳ)
end

@inline function unbroadcast_r(x::Number, y, ȳ, j::J) where J
  x̄ = zero(float(x))
  @simd for I in eachindex(y)
    @inbounds x̄ += j(y[I])*ȳ[I]
  end
  return x̄
end

function ∇broadcast_r(bc::Broadcasted, J)
  y = copy(instantiate(bc))
  back(ȳ) = map(unbroadcast_r, broadcast_args(bc), map(_ -> y, J), map(_ -> ȳ, J), J)
  return y, back
end

# Mixed Mode

import ForwardDiff
using ForwardDiff: Dual

dualtype(::Type{Dual{G,T,P}}) where {G,T,P} = T
dualtype(T) = T

function dual_function(f::F) where F
  function (args::Vararg{Any,N}) where N
    ds = map(args, ntuple(identity,Val(N))) do x, i
      Dual(x, ntuple(j -> i==j ? one(typeof(x)) : zero(typeof(x)), Val(N)))
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
  y, gs = broadcast_gradient(bc, dualtype(T))
  back(Δ) = map((x, d) -> unbroadcast(x, Δ.*d), bc.args, gs)
  return y, back
end

function ∇broadcast_f(bc::Broadcasted{<:AbstractArrayStyle{0}})
  out = dualify(instantiate(flatten(bc)))[]
  return out.value, Δ -> map(x -> x*Δ, out.partials.values)
end

# ∇broadcast(bc::Broadcasted, ::Nothing) = ∇broadcast_f(bc)
# ∇broadcast(bc::Broadcasted, J) = ∇broadcast_r(bc, J)
# ∇broadcast(bc::Broadcasted) = ∇broadcast(bc, Jbroadcast(bc))

∇broadcast(bc::Broadcasted) = ∇broadcast_f(bc)

@adjoint function broadcasted(f, args...)
  broadcasted(f, args...), Δ -> (nothing, Δ.args...)
end

@adjoint function copy(bc::Broadcasted{<:AbstractArrayStyle})
  y, back = ∇broadcast(bc)
  y, Δ -> (unflatten(bc, back(Δ)),)
end

@adjoint function materialize(bc::Broadcasted{<:AbstractArrayStyle})
  y, back = ∇broadcast(bc)
  y, Δ -> (unflatten(bc, back(Δ)),)
end
