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

# Core functions

@nograd Core.apply_type, Core.typeof, nfields, fieldtype,
  (==), (===), (>=), (<), (>), isempty

@adjoint (::Type{V})(x...) where V<:Val = V(x...), _ -> nothing

@adjoint ifelse(cond::Bool, t, f) =
  ifelse(cond, t, f),
  Δ -> cond ? (Δ, zero(Δ)) : (zero(Δ), Δ)

@adjoint Base.typeassert(x, T) = Base.typeassert(x, T), Δ -> (Δ, nothing)

@generated function accum_param(cx::Context, x, Δ)
  isbitstype(x) && return :Δ
  quote
    k = Key(x)
    haskey(cache(cx), k) && (cache(cx)[k] = accum(cache(cx)[k],Δ))
    return
  end
end

unwrap(x) = x

@adjoint unwrap(x) = unwrap(x), Δ ->(accum_param(__context__, x, Δ),)

# Tuples

@adjoint tuple(xs...) = xs, identity

@adjoint getindex(xs::NTuple{N,Any}, i::Integer) where N =
  (xs[i], Δ -> (ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing))

# Needed for iteration lowering
@adjoint Core.getfield(xs::NTuple{N,Any}, i::Integer) where N =
  (xs[i], Δ -> (ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing))

@adjoint function Base.first(xs::Tuple)
  drest = map(_->nothing, tail(xs))
  first(xs), Δ -> ((Δ, drest...),)
end

_empty(x) = length(x)
_empty(x::Tuple) = map(_->nothing, x)

_unapply(t::Integer, xs) = xs[1:t], xs[t+1:end]
_unapply(t, xs) = first(xs), tail(xs)
_unapply(t::Tuple{}, xs) = (), xs

function _unapply(t::Tuple, xs)
  t1, xs1 = _unapply(first(t), xs)
  t2, xs2 = _unapply(tail(t), xs1)
  (t1, t2...), xs2
end

unapply(t, xs) = _unapply(t, xs)[1]

@adjoint function Core._apply(f, args...)
  y, back = Core._apply(_forward, (__context__, f), args...)
  st = map(_empty, args)
  y, function (Δ)
    Δ = back(Δ)
    (first(Δ), unapply(st, Base.tail(Δ))...)
  end
end

# Structs

deref!(x) = x

function deref!(x::Ref)
  d = x[]
  x[] = nothing
  return d
end

@generated nt_nothing(x) = Expr(:tuple, [:($f=nothing) for f in fieldnames(x)]...)

@generated pair(::Val{k}, v) where k = :($k = v,)

# TODO make this inferrable
# Right now constant prop is too fragile ...
@adjoint function getfield(x, f::Symbol)
  val = getfield(x, f)
  unwrap(val), function (Δ)
    accum_param(__context__, val, Δ)
    if isimmutable(x)
      ((;nt_nothing(x)...,pair(Val(f), Δ)...), nothing)
    else
      dx = getfield(grad_mut(__context__, x), f)
      dx[] = accum(dx[], Δ)
      return
    end
  end
end

# ... so we have Zygote call this version where we can.
literal_getproperty(x, ::Val{f}) where f = getproperty(x, f)

@adjoint function literal_getproperty(x, ::Val{f}) where f
  val = getproperty(x, f)
  function back(Δ)
    accum_param(__context__, val, Δ)
    if isimmutable(x)
      ((;nt_nothing(x)...,pair(Val(f), Δ)...), nothing)
    else
      dx = getfield(grad_mut(__context__, x), f)
      dx[] = accum(dx[], Δ)
      return
    end
  end
  unwrap(val), back
end

@generated function grad_mut(x)
  Expr(:tuple, [:($f = Ref{Any}(nothing)) for f in fieldnames(x)]...)
end

function grad_mut(cx::Context, x)
  T = Core.Compiler.return_type(grad_mut, Tuple{typeof(x)})
  k = Key(x)
  ch = cache(cx)
  if haskey(ch, k)
    ch[k]::T
  else
    ch[k] = grad_mut(x)
  end
end

@adjoint! function setfield!(x, f, val)
  y = setfield!(x, f, val)
  g = grad_mut(__context__, x)
  y, function (_)
    r = getfield(g, f)
    Δ = deref!(r)
    (nothing, nothing, Δ)
  end
end

@generated function __new__(T, args...)
  quote
    Base.@_inline_meta
    $(Expr(:new, :T, [:(args[$i]) for i = 1:length(args)]...))
  end
end

@generated function __splatnew__(T, args)
  quote
    Base.@_inline_meta
    $(Expr(:splatnew, :T, :args))
  end
end

struct Jnew{T,G,splat}
  g::G
end

Jnew{T}(g) where T = Jnew{T,typeof(g)}(g)

@adjoint! function __new__(T, args...)
  x = __new__(T, args...)
  g = !T.mutable || fieldcount(T) == 0 ? nothing : grad_mut(__context__, x)
  x, Jnew{T,typeof(g),false}(g)
end

@adjoint! function __splatnew__(T, args)
  x = __splatnew__(T, args)
  g = !T.mutable || fieldcount(T) == 0 ? nothing : grad_mut(__context__, x)
  x, Jnew{T,typeof(g),true}(g)
end

# TODO captured mutables + multiple calls to `back`
@generated function (back::Jnew{T,G,false})(Δ::Union{NamedTuple,Nothing}) where {T,G}
  !T.mutable && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ : :(back.g)
  :(nothing, $(map(f -> :(deref!($Δ.$f)), fieldnames(T))...))
end


@generated function (back::Jnew{T,G,true})(Δ::Union{NamedTuple,Nothing}) where {T,G}
  !T.mutable && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ : :(back.g)
  :(nothing, ($(map(f -> :(deref!($Δ.$f)), fieldnames(T))...),))
end

# Mutable Primitives (e.g. arrays)

ismutvalue(x) = false

mutkey(x) = ismutvalue(x) ? Key(x) : nothing
mutkeys(xs...) = map(mutkey, xs)

function out_grad_mut(cx, k::Key, default)
  Δ = get(cache(cx), k, nothing)
  cache(cx)[k] = nothing
  return accum(Δ, default)
end

out_grad_mut(cx, ::Nothing, default) = default

out_grad_mut(cx, xs::Tuple, dxs) = map((x, dx) -> out_grad_mut(cx, x, dx), xs, dxs)
out_grad_mut(cx, xs::Tuple, ::Nothing) = nothing

function in_grad_mut(cx, k::Key, Δ)
  cache(cx)[k] = accum(get(cache(cx), k, nothing), Δ)
  return
end

in_grad_mut(cx, ::Nothing, Δ) = Δ

in_grad_mut(cx, ::Tuple, ::Nothing) = nothing

in_grad_mut(cx, xs::Tuple, dxs) = map((x, dx) -> in_grad_mut(cx, x, dx), xs, dxs)

mutback(cache, ks::NTuple{<:Any,Nothing}, ::Nothing, back) = back

function mutback(cx, xs, y, back::F) where F
  return function (dy)
    dxs = back(out_grad_mut(cx, y, dy))
    in_grad_mut(cx, xs, dxs)
  end
end
