# Interfaces

accum() = nothing
accum(x) = x

accum(x, y) =
  x == nothing ? y :
  y == nothing ? x :
  x + y

accum(x, y, zs...) = accum(accum(x, y), zs...)

accum(x::Tuple, y::Tuple) = accum.(x, y)
accum(x::AbstractArray, y::AbstractArray) = accum.(x, y)

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
  pushfirst!(args, :($(esc(:__context__))::Context), :(::typeof($(esc(name)))))
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

@nograd Core.apply_type, Core.typeof, nfields, fieldtype,
  (==), (===), (>=), (<), (>)

@grad ifelse(cond::Bool, t, f) =
  Base.select_value(cond, t, f),
  Δ -> cond ? (Δ, zero(Δ)) : (zero(Δ), Δ)

@grad Base.typeassert(x, T) = Base.typeassert(x, T), Δ -> (Δ, nothing)

function accum_param(cx::Context, x, Δ)
  haskey(cache(cx), x) && (cache(cx)[x] = accum(cache(cx)[x],Δ))
  return
end

unwrap(x) = x

@grad unwrap(x) = unwrap(x), Δ -> accum_param(__context__, x, Δ)

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

@grad function Core._apply(f, args...)
  y, J = Core._apply(_forward, (__context__, f), args...)
  y, function (Δ)
    Δ = J(Δ)
    (first(Δ), unapply(args, Base.tail(Δ))...)
  end
end

# Structs

@generated nt_nothing(x) = Expr(:tuple, [:($f=nothing) for f in fieldnames(x)]...)

@generated pair(::Val{k}, v) where k = :($k = v,)

# TODO make this inferrable
@grad function getfield(x, f::Symbol)
  val = getfield(x, f)
  unwrap(val), function (Δ)
    accum_param(__context__, val, Δ)
    if isimmutable(x)
      ((;nt_nothing(x)...,pair(Val{f}(), Δ)...), nothing)
    else
      accum!(getfield(grad_mut(__context__, x), f), Δ)
      return
    end
  end
end

@generated function grad_mut(x)
  Expr(:tuple, [:($f = Ref{Any}(nothing)) for f in fieldnames(x)]...)
end

function grad_mut(cx::Context, x)
  T = Core.Compiler.return_type(grad_mut, Tuple{typeof(x)})
  ch = cache(cx)
  if haskey(ch, x)
    ch[x]::T
  else
    ch[x] = grad_mut(x)
  end
end

function _forward(cx::Context, ::typeof(setfield!), x, f, val)
  y = setfield!(x, f, val)
  g = grad_mut(cx::Context, x)
  y, function (_)
    r = getfield(g, f)
    Δ = deref!(r)
    (nothing, nothing, nothing, Δ)
  end
end

@generated function __new__(T, args...)
  quote
    Base.@_inline_meta
    $(Expr(:new, :T, [:(args[$i]) for i = 1:length(args)]...))
  end
end

struct Jnew{T,G}
  g::G
end

Jnew{T}(g) where T = Jnew{T,typeof(g)}(g)

function _forward(cx::Context, ::typeof(__new__), T, args...)
  x = __new__(T, args...)
  g = !T.mutable || fieldcount(T) == 0 ? nothing : grad_mut(cx, x)
  x, Jnew{T}(g)
end

# TODO captured mutables + multiple calls to `back`
@generated function (back::Jnew{T,G})(Δ) where {T,G}
  !T.mutable && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ  : :(back.g)
  :(nothing, nothing, $(map(f -> :(deref!($Δ.$f)), fieldnames(T))...))
end
