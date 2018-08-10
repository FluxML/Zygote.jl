@nograd size, length, eachindex

@inline _forward(::Context, ::Type{T}, args...) where T<:Array = T(args...), Δ -> nothing

@grad Base.vect(xs...) = Base.vect(xs...), Δ -> (Δ...,)

Base.zero(xs::AbstractArray{Any}) = fill!(similar(xs), nothing)

@grad function getindex(xs::Array, i...)
  xs[i...], function (Δ)
    Δ′ = zero(xs)
    Δ′[i...] = Δ
    (Δ′, map(_ -> nothing, i)...)
  end
end

@grad permutedims(xs, dims) = permutedims(xs, dims),
  Δ -> (permutedims(Δ, invperm(dims)),nothing)

@grad a::AbstractVecOrMat * b::AbstractVecOrMat =
  a * b, Δ -> (Δ * transpose(b), transpose(a) * Δ)

@grad transpose(x) = transpose(x), Δ -> (transpose(Δ),)
@grad adjoint(x) = adjoint(x), Δ -> (adjoint(Δ),)

@grad sum(xs::AbstractArray) = sum(xs), Δ -> (similar(xs) .= Δ,)

# @grad sum(xs::AbstractArray, dim...) =
#   sum(xs, dim...), Δ -> (similar(xs) .= Δ, map(_->nothing, dim)...)

@grad prod(xs) = prod(xs), Δ -> (prod(xs) ./ xs .* Δ,)

# @grad prod(xs, dim) = prod(xs, dim),
#   Δ -> (reshape(.*(circshift.([reshape(xs, length(xs))], 1:length(xs)-1)...), size(xs)) .* Δ,
#         nothing)

# NNlib

using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv, maxpool, meanpool

@grad softmax(xs) = softmax(xs), Δ -> (∇softmax(Δ, xs),)

@grad logsoftmax(xs) = logsoftmax(xs), Δ -> (∇logsoftmax(Δ, xs),)

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

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Number, Δ) = sum(Δ)

dual(x, p) = x
dual(x::Real, p) = Dual(x, p)

function partial(f::F, Δ, i, args::Vararg{Any,N}) where {F,N}
  dargs = ntuple(j -> dual(args[j], i==j), Val(N))
  return Δ * f(dargs...).partials[1]
end

@inline function ∇broadcast(f::F, args::Vararg{Any,N}) where {F,N}
  y = broadcast(f, args...)
  function back(Δ)
    Δargs = ntuple(i -> partial.(f, Δ, i, args...), Val(N))
    return unbroadcast.(args, Δargs)
  end
  return y, back
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
  let (y, back) = ∇broadcast(bc′.f, bc′.args...)
    y, Δ -> (unflatten(bc, back(Δ)),)
  end
end
