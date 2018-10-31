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

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Number, Δ) = sum(Δ)

# Reverse Mode

Jtrivial(f, a...) = nothing
Jtrivial(::typeof(+), a...) = a
Jtrivial(::typeof(-), a, b) = (a..., .-b...)

trivia(_) = (1,)
function trivia(bc::Broadcasted)
  t = trivia.(bc.args)
  any(t -> t === nothing, t) && return
  Jtrivial(bc.f, t...)
end

Joutput(f, a...) = nothing
Joutput(::typeof(exp), x) = map(t -> y -> y*t, x)

function Jbroadcast(bc::Broadcasted)
  t = trivia.(bc.args)
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

using ForwardDiff: Dual

dual(x, p) = x
dual(x::Real, p) = Dual(x, p)

dualtype(::Type{Dual{G,T,P}}) where {G,T,P} = T

function dual_function(f::F) where F
  function (args::Vararg{Any,N}) where N
    ds = map(args, ntuple(identity,Val(N))) do x, i
      dual(x, ntuple(j -> i==j, Val(N)))
    end
    return f(ds...)
  end
end

dualify(bc::Broadcasted{S}) where S = Broadcasted{S}(dual_function(bc.f), bc.args, bc.axes)

function broadcast_gradient!(bc::Broadcasted, dest::AbstractArray, grads::Vararg{Any})
  @simd for I in eachindex(bc)
    @inbounds begin
      out = bc[I]
      dest[I] = out.value
      map((g, p) -> g[I] = p, grads, out.partials.values)
    end
  end
end

function broadcast_gradient(bc::Broadcasted, ::Type{T}) where T
  dest = similar(bc, T)
  grads = map(_ -> similar(bc, T), bc.args)
  broadcast_gradient!(bc, dest, grads...)
  return dest, grads
end

@inline function ∇broadcast(bc′::Broadcasted)
  bc = dualify(instantiate(flatten(bc′)))
  T = combine_eltypes(bc.f, bc.args)
  y, gs = broadcast_gradient(bc, dualtype(T))
  back(Δ) = map((x, d) -> unbroadcast(x, Δ.*d), bc.args, gs)
  return y, back
end

function ∇broadcast(bc::Broadcasted{<:AbstractArrayStyle{0}})
  out = dualify(instantiate(flatten(bc)))[]
  return out.value, Δ -> map(x -> x*Δ, out.partials.values)
end

@grad function broadcasted(f, args...)
  broadcasted(f, args...), Δ -> (nothing, Δ.args...)
end

@grad function materialize(bc::Broadcasted{<:DefaultArrayStyle})
  let (y, back) = ∇broadcast(bc)
    y, Δ -> (unflatten(bc, back(Δ)),)
  end
end
