using Base: RefValue

# Interfaces

accum() = nothing
accum(x) = x

accum(x, y) =
  x == nothing ? y :
  y == nothing ? x :
  x + y

accum(x, y, zs...) = accum(accum(x, y), zs...)

accum(x::Tuple, y::Tuple) = accum.(x, y)

@generated function accum(x::NamedTuple, y::NamedTuple)
  grad(x) = x in fieldnames(y) ? :(y.$x) : :nothing
  Expr(:tuple, [:($f=accum(x.$f, $(grad(f)))) for f in fieldnames(x)]...)
end

function accum(x::RefValue, y::RefValue)
  @assert x === y
  return x
end

# Core functions

@nograd Core.apply_type, Core.typeof, nfields, fieldtype,
  (==), (===), (>=), (<), (>), isempty, supertype, Base.typename,
  Base.parameter_upper_bound, eps

@adjoint deepcopy(x) = deepcopy(x), ȳ -> (ȳ,)

@adjoint (::Type{V})(x...) where V<:Val = V(x...), _ -> nothing

@adjoint ifelse(cond::Bool, t, f) =
  ifelse(cond, t, f),
  Δ -> cond ? (nothing, Δ, zero(Δ)) : (nothing, zero(Δ), Δ)

@adjoint Base.typeassert(x, T) = Base.typeassert(x, T), Δ -> (Δ, nothing)

@generated function accum_param(cx::Context, x, Δ)
  isbitstype(x) && return :Δ
  quote
    ismutvalue(x) && return accum!(grad_mut(cx, x), Δ)
    k = Key(x)
    haskey(cache(cx), k) && (cache(cx)[k] = accum(cache(cx)[k],Δ))
    return
  end
end

function accum_global(cx::Context, ref, x̄)
  gs = globals(cx)
  gs[ref] = accum(get(gs, ref, nothing), x̄)
  return
end

unwrap(x) = x

@adjoint unwrap(x) = unwrap(x), x̄ -> (accum_param(__context__, x, x̄),)

unwrap(ref, x) = x

# Right now we accumulate twice, for both implicit params and the `globals`
# API. Eventually we'll deprecate implicit params.
@adjoint unwrap(ref, x) = unwrap(x), function (x̄)
  accum_global(__context__, ref, x̄)
  accum_param(__context__, x, x̄)
  return
end

# Tuples

using Base: tail

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
      dx = grad_mut(__context__, x)
      dx[] = (;dx[]...,pair(Val(f),accum(getfield(dx[], f), Δ))...)
      return (dx,nothing)
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
      dx = grad_mut(__context__, x)
      dx[] = (;dx[]...,pair(Val(f),accum(getfield(dx[], f), Δ))...)
      return (dx,nothing)
    end
  end
  unwrap(val), back
end

grad_mut(x) = Ref{Any}(nt_nothing(x))

function grad_mut(cx::Context, x)
  k = Key(x)
  ch = cache(cx)
  if haskey(ch, k)
    ch[k]
  else
    ch[k] = grad_mut(x)
  end
end

@adjoint! function setfield!(x, f, val)
  y = setfield!(x, f, val)
  g = grad_mut(__context__, x)
  y, function (_)
    Δ = getfield(g[], f)
    g[] = (;g[]...,pair(Val(f),nothing)...)
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
@generated function (back::Jnew{T,G,false})(Δ::Union{NamedTuple,Nothing,RefValue}) where {T,G}
  !T.mutable && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ : :(back.g[])
  quote
    x̄ = $Δ
    $(G == Nothing || :($Δ = nt_nothing($Δ)))
    (nothing, $(map(f -> :(x̄.$f), fieldnames(T))...))
  end
end

@generated function (back::Jnew{T,G,true})(Δ::Union{NamedTuple,Nothing,RefValue}) where {T,G}
  !T.mutable && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ : :(back.g)
  quote
    x̄ = $Δ
    $(G == Nothing || :($Δ = nt_nothing($Δ)))
    (nothing, ($(map(f -> :(x̄.$f), fieldnames(T))...),))
  end
end

# Mutable Primitives (e.g. arrays)

ismutvalue(x) = false

mutkey(x) = ismutvalue(x) ? Key(x) : nothing
mutkeys(xs...) = map(mutkey, xs)

function out_grad_mut(cx, x, x̄)
  ismutvalue(x) || return x̄
  Δ = grad_mut(cx, x)
  accum!(Δ, x̄)
  return Δ
end

out_grad_mut(cx, xs::Tuple, dxs) = map((x, dx) -> out_grad_mut(cx, x, dx), xs, dxs)
out_grad_mut(cx, xs::Tuple, ::Nothing) = nothing

function in_grad_mut(cx, x, x̄)
  ismutvalue(x) || return x̄
  return accum!(grad_mut(cx, x), x̄)
end

in_grad_mut(cx, xs::Tuple, dxs) = map((x, dx) -> in_grad_mut(cx, x, dx), xs, dxs)
in_grad_mut(cx, ::Tuple, ::Nothing) = nothing

mutback(cache, ks::NTuple{<:Any,Nothing}, ::Nothing, back) = back

function mutback(cx, xs, y, back::F) where F
  return function (ȳ)
    dxs = back(out_grad_mut(cx, y, ȳ))
    in_grad_mut(cx, xs, dxs)
  end
end
