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
using Base.Broadcast: Broadcasted, DefaultArrayStyle, instantiate

# Structural utilities

using Base: tail

tcat(x) = x
tcat(x, y, z...) = tcat((x..., y...), z...)

broadcast_args(x) = (x,)
broadcast_args(bc::Broadcasted) = tcat(map(broadcast_args, bc.args)...)

accum_sum(xs; dims = :) = reduce(accum, xs, dims = dims)

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, accum_sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Union{Number,Ref}, Δ) = accum_sum(Δ)

# Trivial Special Cases
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

# Reverse Mode

# TODO: forward context appropriately
# multi-output map for better performance
function ∇broadcast_r(bc::Broadcasted)
  bc′, unflatten = _forward(Broadcast.flatten, bc)
  len = Val(length(bc′.args)+1)
  y∂b = broadcast(_forward, bc′.f, bc′.args...)
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

function ∇broadcast_r(bc::Broadcasted{<:DefaultArrayStyle{0}})
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

∇broadcast(bc::Broadcasted, ::Nothing) = ∇broadcast_r(bc)
∇broadcast(bc::Broadcasted, J) = ∇broadcast_t(bc, J)
∇broadcast(bc::Broadcasted) = ∇broadcast(bc, Jbroadcast(bc))

@adjoint Broadcast.materialize(bc::Broadcasted{<:DefaultArrayStyle}) = ∇broadcast_r(bc)
