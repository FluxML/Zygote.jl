grad(xs::Array) = grad.(xs)

@inline _forward(::Context, ::Type{T}, args...) where T<:Array = T(args...), Δ -> nothing

@grad Base.vect(xs...) = Base.vect(xs...), Δ -> (Δ...,)

@grad function getindex(xs::Array, i...)
  xs[i...], function (Δ)
    Δ′ = zero(xs)
    Δ′[i...] = Δ
    (Δ′, map(_ -> nothing, i)...)
  end
end

@grad a::AbstractVecOrMat * b::AbstractVecOrMat =
  a * b, Δ -> (Δ * transpose(b), transpose(a) * Δ)

@grad sum(xs::AbstractArray, dim...) =
  sum(xs, dim...), Δ -> (similar(xs) .= Δ, map(_->nothing, dim)...)

@grad prod(xs) = prod(xs), Δ -> (prod(xs) ./ xs .* Δ,)

@grad prod(xs, dim) = prod(xs, dim),
  Δ -> (reshape(.*(circshift.([reshape(xs, length(xs))], 1:length(xs)-1)...), size(xs)) .* Δ,
        nothing)

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

using Base: tail
using Base.Broadcast
using Base.Broadcast: Broadcasted, DefaultArrayStyle, broadcasted, materialize, instantiate
using ForwardDiff: Dual, partials

dualify(x, ps) = x
dualify(x::Real, ps) = Dual(x, ps)
dualify(xs::AbstractArray{<:Real}, ps) = dualify.(xs, (ps,))

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val{ndims(x)}))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
    trim(x, sum(Δ, filter(n -> size(x, n) == 1, 1:ndims(Δ))))

unbroadcast(x::Number, Δ) = sum(Δ)

function getpartial(Δ, x, i)
  @inbounds p = getindex(partials(x), i)
  return Δ * p
end

function ∇broadcast(f, args::NTuple{N,Any}) where N
  dargs = map((x,i) -> dualify(x, ntuple(j -> i==j, Val(N))),
              args, ntuple(identity, Val(N)))
  out = broadcast(f, dargs...)
  (x -> x.value).(out), function (Δ)
    Δxs = ntuple(i -> getpartial.(Δ, out, i), Val(N))
    unbroadcast.(args, Δxs)
  end
end

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

@grad function broadcasted(f, args...)
  broadcasted(f, args...), Δ -> (nothing, Δ.args...)
end

@grad function materialize(bc::Broadcasted{<:DefaultArrayStyle})
  bc′ = instantiate(Broadcast.flatten(bc))
  y, back = ∇broadcast(bc′.f, bc′.args)
  y, Δ -> (unflatten(bc, back(Δ)),)
end
