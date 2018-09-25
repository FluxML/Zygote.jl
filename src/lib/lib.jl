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

@grad (::Type{V})(x...) where V<:Val = V(x...), _ -> nothing

@grad ifelse(cond::Bool, t, f) =
  ifelse(cond, t, f),
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

@grad function Base.first(xs::Tuple)
  let drest = map(_->nothing, tail(xs))
    first(xs), Δ -> ((Δ, drest...),)
  end
end

_empty(x) = nothing
_empty(x::Tuple) = map(_empty, x)

_unapply(t, xs) = first(xs), tail(xs)
_unapply(t::Tuple{}, xs) = (), xs

function _unapply(t::Tuple, xs)
  t1, xs1 = _unapply(first(t), xs)
  t2, xs2 = _unapply(tail(t), xs1)
  (t1, t2...), xs2
end

unapply(t, xs) = _unapply(t, xs)[1]

@grad function Core._apply(f, args...)
  y, J = Core._apply(_forward, (__context__, f), args...)
  let st = _empty(args), J = J
    y, function (Δ)
      Δ = J(Δ)
      (first(Δ), unapply(st, Base.tail(Δ))...)
    end
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
@grad function getfield(x, f::Symbol)
  val = getfield(x, f)
  w = unwrap(val)
  let val=val
    w, function (Δ)
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
end

# ... so we have Zygote call this version where we can.
literal_getproperty(x, ::Val{f}) where f = getproperty(x, f)

@grad function literal_getproperty(x, ::Val{f}) where f
  let val = getproperty(x, f)
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

@grad! function setfield!(x, f, val)
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

struct Jnew{T,G}
  g::G
end

Jnew{T}(g) where T = Jnew{T,typeof(g)}(g)

@grad! function __new__(T, args...)
  x = __new__(T, args...)
  g = !T.mutable || fieldcount(T) == 0 ? nothing : grad_mut(__context__, x)
  x, Jnew{T}(g)
end

# TODO captured mutables + multiple calls to `back`
@generated function (back::Jnew{T,G})(Δ) where {T,G}
  !T.mutable && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ : :(back.g)
  :(nothing, $(map(f -> :(deref!($Δ.$f)), fieldnames(T))...))
end
