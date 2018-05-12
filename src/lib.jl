# TODO: DiffRules
_forward(::typeof(sin), x) = (sin(x), Δ -> (cos(x)*Δ,))
_forward(::typeof(cos), x) = (cos(x), Δ -> (-sin(x)*Δ,))

_forward(::typeof(+), a, b) = (a+b, Δ -> (Δ, Δ))
_forward(::typeof(*), a, b) = (a*b, Δ -> (Δ*b', a'*Δ))

_forward(::typeof(getindex), xs::NTuple{N}, i::Integer) where N =
  (xs[i], Δ -> (ntuple(j -> i == j ? Δ : nothing, Val{N}), nothing))

# Non-numeric

@generated nt_nothing(x) = Expr(:tuple, [:($f=nothing) for f in fieldnames(x)]...)

@generated pair(::Val{k}, v) where k = :($k = v,)

@inline _forward(::typeof(Base.getfield), x, f::Symbol) =
  getfield(x, f), Δ -> ((;nt_nothing(x)...,pair(Val{f}(), Δ)...), nothing)

@generated function __new__(T, args...)
  quote
    Base.@_inline_meta
    $(Expr(:new, :T, [:(args[$i]) for i = 1:length(args)]...))
  end
end

struct Jnew{T} end

_forward(::typeof(__new__), T, args...) = __new__(T, args...), Jnew{T}()

@generated function (::Jnew{T})(Δ) where T
  Expr(:tuple, nothing, map(f -> :(Δ.$f), fieldnames(T))...)
end

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

function _forward(::typeof(broadcast), f, args::Vararg{Any,N}) where N
  dargs = map((x,i) -> dualify(x, ntuple(j -> i==j, Val{N})),
              args, ntuple(identity, Val{N}))
  out = broadcast(f, dargs...)
  (x -> x.value).(out), function (Δ)
    Δxs = ntuple(i -> getpartial.(Δ, out, i), Val{N})
    (nothing, unbroadcast.(args, Δxs)...)
  end
end
