# Interfaces

grad(::Type) = nothing

@generated function grad(x)
  (x.mutable || fieldcount(x) == 0) && return
  Expr(:tuple, [:($f = grad(x.$f)) for f in fieldnames(x)]...)
end

grad(x::Tuple) = grad.(x)

accum(x, y) = x + y
accum(x, _::Nothing) = x
accum(x::Nothing, _) = x
accum(x::Nothing, _::Nothing) = x
accum(x::Tuple, y::Tuple) = accum.(x, y)

@generated function accum(x::NamedTuple, y::NamedTuple)
  grad(x) = x in fieldnames(y) ? :(y.$x) : :nothing
  Expr(:tuple, [:($f=accum(x.$f, $(grad(f)))) for f in fieldnames(x)]...)
end

using MacroTools: combinedef

_gradtuple(::Nothing) = nothing
_gradtuple(x::Tuple) = (nothing, x...)
_gradtuple(x) = error("Gradient $x should be a tuple")

macro grad(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {T__} = body_)) || error("Need a function definition")
  T == nothing && (T = [])
  args = esc.(args)
  T = esc.(T)
  pushfirst!(args, :(::Context), :(::typeof($(esc(name)))))
  body = quote
    Base.@_inline_meta
    y, back = $(esc(body))
    back2(::Nothing) = nothing
    # return needed for type inference
    back2(Δ) = return _gradtuple(back(Δ))
    y, back2
  end
  :(_forward($(args...)) where $(T...) = $body)
end

macro nograd(ex)
  isexpr(ex, :tuple) || (ex = Expr(:tuple, ex))
  blk = :(;)
  for f in ex.args
    push!(blk.args, :(@inline Zygote._forward(::Context, ::typeof($(esc(f))), args...) = $(esc(f))(args...), Δ -> nothing))
  end
  return blk
end

# Core functions

@nograd Core.apply_type, Core.typeof, nfields,
  (==), (===), (>=), (<), (>)

@grad ifelse(cond::Bool, t, f) =
  Base.select_value(cond, t, f),
  Δ -> cond ? (Δ, zero(Δ)) : (zero(Δ), Δ)

@grad Base.typeassert(x, T) = Base.typeassert(x, T), Δ -> (Δ, nothing)

# Tuples

@grad tuple(xs...) = xs, identity

@grad getindex(xs::NTuple{N,Any}, i::Integer) where N =
  (xs[i], Δ -> (ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing))

# Needed for iteration lowering
@grad Core.getfield(xs::NTuple{N,Any}, i::Integer) where N =
  (xs[i], Δ -> (ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing))

# TODO faster version
function unapply(xs, Δs)
  Δs′ = []
  for x in xs
    push!(Δs′, Δs[1:length(x)])
    Δs = Δs[length(x)+1:end]
  end
  return (Δs′...,)
end

function _forward(ctx::Context, ::typeof(Core._apply), f, args...)
  y, J = Core._apply(_forward, (ctx, f), args...)
  y, function (Δ)
    Δ = J(Δ)
    (nothing, first(Δ), unapply(args, Base.tail(Δ))...)
 end
end

# Structs

@generated nt_nothing(x) = Expr(:tuple, [:($f=nothing) for f in fieldnames(x)]...)

@generated pair(::Val{k}, v) where k = :($k = v,)

# TODO make this inferrable
@grad Base.getfield(x, f::Symbol) =
  getfield(x, f), Δ -> ((;nt_nothing(x)...,pair(Val{f}(), Δ)...), nothing)

@generated function __new__(T, args...)
  quote
    Base.@_inline_meta
    $(Expr(:new, :T, [:(args[$i]) for i = 1:length(args)]...))
  end
end

struct Jnew{T} end

@grad __new__(T, args...) = __new__(T, args...), Jnew{T}()

@generated function (::Jnew{T})(Δ) where T
  Expr(:tuple, nothing, map(f -> :(Δ.$f), fieldnames(T))...)
end
