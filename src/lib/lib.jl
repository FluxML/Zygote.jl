using Base: RefValue

# Interfaces

accum() = nothing
accum(x) = x

accum(x, y) =
  x === nothing ? y :
  y === nothing ? x :
  x + y

accum(x, y, zs...) = accum(accum(x, y), zs...)

accum(x::Tuple, y::Tuple) = accum.(x, y)
accum(x::AbstractArray, y::AbstractArray) = accum.(x, y)

@generated function accum(x::NamedTuple, y::NamedTuple)
  grad(fx) = fx in fieldnames(y) ? :(y.$fx) : :(Zero())
  Expr(:tuple, [:($f=accum(x.$f, $(grad(f)))) for f in fieldnames(x)]...)
end

function accum(x::RefValue, y::RefValue)
  @assert x === y
  return x
end

# Core functions
@nograd eps, Base.eval, Core.TypeVar, Core.UnionAll

@adjoint deepcopy(x) = deepcopy(x), ȳ -> (ȳ,)

@adjoint (::Type{V})(x...) where V<:Val = V(x...), _ -> nothing

@adjoint ifelse(cond::Bool, t, f) =
  ifelse(cond, t, f),
  Δ -> cond ? (nothing, Δ, zero(Δ)) : (nothing, zero(Δ), Δ)

@adjoint Base.typeassert(x, T) = Base.typeassert(x, T), Δ -> (Δ, nothing)

@generated function accum_param(cx::Context, x, Δ)
  isbitstype(x) && return :(Δ)
  quote
    if haskey(cache(cx), x)
      cache(cx)[x] = accum(cache(cx)[x],Δ)
      return
    else
      return Δ
    end
  end
end

function accum_global(cx::Context, ref, x̄)
  (x̄ === nothing || isconst(ref.mod, ref.name)) && return
  gs = cache(cx)
  gs[ref] = accum(get(gs, ref, Zero()), x̄)
  return
end

unwrap(x) = x

@adjoint unwrap(x) = unwrap(x), x̄ -> (accum_param(__context__, x, x̄),)

unwrap(ref, x) = x

@adjoint unwrap(ref, x) = unwrap(x), function (x̄)
  accum_global(__context__, ref, x̄)
  (accum_param(__context__, x, x̄),)
end

function global_set(ref, val)
  ccall(:jl_set_global, Cvoid, (Any, Any, Any),
        ref.mod, ref.name, val)
end

@adjoint! function global_set(ref, x)
  global_set(ref, x), function (x̄)
    gs = cache(__context__)
    x̄ = accum(get(gs, ref, nothing), x̄)
    gs[ref] = Zero() # this is a side effect so escapes legacy2differential transform
    return (nothing, x̄)
  end
end

# Tuples

using Base: tail

@adjoint tuple(xs...) = xs, identity

@adjoint function literal_getindex(xs::NTuple{N,Any}, ::Val{i}) where {N,i}
  val = xs[i]
  function back(Δ)
    accum_param(__context__, val, Δ) === nothing && return
    return ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing
  end
  val, back
end

@adjoint function getindex(xs::NTuple{N,Any}, i::Integer) where N
  val = xs[i]
  function back(Δ)
    accum_param(__context__, val, Δ) === nothing && return
    return ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing
  end
  return val, back
end

@adjoint getindex(xs::NTuple{N,Any}, r::AbstractUnitRange) where N =
  (xs[r], Δ -> (ntuple(j -> j in r ? Δ[findfirst(isequal(j), r)] : nothing, Val(N)), nothing))

function _pullback(cx::Context, ::typeof(literal_indexed_iterate), xs::Tuple, ::Val{i}) where i
  y, b = _pullback(cx, literal_getindex, xs, Val(i))
  back(::Nothing) = (legacytype_warn(Nothing); return Zero())
  back(x::AbstractZero) = x
  back(ȳ) = b(ȳ[1])
  (y, i+1), back
end

function _pullback(cx::Context, ::typeof(literal_indexed_iterate), xs::Tuple, ::Val{i}, st) where i
  y, b = _pullback(cx, literal_getindex, xs, Val(i))
  back(::Nothing) = (legacytype_warn(Nothing); return Zero())
  back(x::AbstractZero) = x
  back(ȳ) = (b(ȳ[1])..., Zero())
  (y, i+1), back
end

# Needed for iteration lowering
@adjoint Core.getfield(xs::NTuple{N,Any}, i::Integer) where N =
  (xs[i], Δ -> (ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing))

@adjoint Core.getfield(xs::NamedTuple{K,<:NTuple{N,Any}}, i::Integer) where {K,N} =
  (xs[i], Δ -> (NamedTuple{K}(ntuple(j -> i == j ? Δ : nothing, Val(N))), nothing))

@adjoint function Base.first(xs::Tuple)
  drest = map(_->nothing, tail(xs))
  first(xs), Δ -> ((Δ, drest...),)
end

@adjoint Base.tail(xs::Tuple) = tail(xs), x̄s -> ((nothing, x̄s...),)

_empty(x) = length(x)
_empty(x::Union{Tuple,NamedTuple}) = map(_->nothing, x)

_unapply(t::Integer, xs) = xs[1:t], xs[t+1:end]
_unapply(t, xs) = first(xs), tail(xs)
_unapply(t::Tuple{}, xs) = (), xs

function _unapply(t::Tuple, xs)
  t1, xs1 = _unapply(first(t), xs)
  t2, xs2 = _unapply(tail(t), xs1)
  (t1, t2...), xs2
end

function _unapply(t::NamedTuple{K}, xs) where K
  t, rst = _unapply(Tuple(t), xs)
  NamedTuple{K}(t), rst
end

unapply(t, xs) = _unapply(t, xs)[1]

@adjoint! function Core._apply(f, args...)
  y, back = Core._apply(_pullback, (__context__, f), args...)
  st = map(_empty, args)
  y, function (Δ)
    Δ = differential2legacy(back(legacy2differential(Δ)))
    if Δ === nothing
      return nothing
    else
      (first(Δ), unapply(st, Base.tail(Δ))...)
    end
  end
end

if VERSION >= v"1.4.0-DEV.304"
  @adjoint! function Core._apply_iterate(::typeof(iterate), f, args...)
    y, back = Core._apply(_pullback, (__context__, f), args...)
    st = map(_empty, args)
    y, function (Δ)
      Δ = differential2legacy(back(legacy2differential(Δ)))
      if Δ === nothing
        return nothing
      else
        (nothing, first(Δ), unapply(st, Base.tail(Δ))...)
      end
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

@generated nt_zero(x) = Expr(:tuple, [:($f=Zero()) for f in fieldnames(x)]...)

@generated pair(::Val{k}, v) where k = :($k = v,)

@adjoint function literal_getproperty(x, ::Val{f}) where f # TODO rewrite as explicit pullback
  val = getproperty(x, f)
  function back(Δ)
    accum_param(__context__, val, Δ) === nothing && return
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

_pullback(cx::Context, ::typeof(getproperty), x, f::Symbol) =
  _pullback(cx, literal_getproperty, x, Val(f))

_pullback(cx::Context, ::typeof(getfield), x, f::Symbol) =
  _pullback(cx, literal_getproperty, x, Val(f))

_pullback(cx::Context, ::typeof(literal_getindex), x::NamedTuple, ::Val{f}) where f =
  _pullback(cx, literal_getproperty, x, Val(f))

_pullback(cx::Context, ::typeof(literal_getproperty), x::Tuple, ::Val{f}) where f =
  _pullback(cx, literal_getindex, x, Val(f))

#grad_mut(x) = Ref{Any}(nt_zero(x)) # TODO
grad_mut(x::T) where T = Ref{Any}(Composite{T}())

function grad_mut(cx::Context, x)
  ch = cache(cx)
  if haskey(ch, x)
    ch[x]
  else
    ch[x] = grad_mut(x)
  end
end

@adjoint! function setfield!(x, f, val)
  y = setfield!(x, f, val)
  g = grad_mut(__context__, x)
  y, function (_)
    Δ = differential2legacy(getfield(g[], f))
    g[] = (;g[]...,pair(Val(f),Zero())...)
    (nothing, nothing, Δ)
  end
end

struct Jnew{T,G,splat} # T is the primal type, G is the gradient type
  g::G
end

Jnew{T}(g) where T = Jnew{T,typeof(g)}(g)

function _pullback(__context__::AContext, ::typeof(__new__), ::Type{T}, args...) where T
  x = __new__(T, args...)
  g = !T.mutable || fieldcount(T) == 0 ? Zero() : grad_mut(__context__, x)
  return x, Δ -> gradtuple1(Jnew{T,typeof(g),false}(g)(Δ))
end

function _pullback(__context__::AContext, ::typeof(__splatnew__), ::Type{T}, args) where T
  x = __splatnew__(T, args)
  g = !T.mutable || fieldcount(T) == 0 ? Zero() : grad_mut(__context__, x)
  return x,  Δ -> gradtuple1(Jnew{T,typeof(g),true}(g)(Δ))
end

const allowed_gradient_T = Union{
  NamedTuple,
  Nothing,
  AbstractZero,
  RefValue,
  ChainRules.Composite#{Any, T} where T<:Union{Tuple, NamedTuple} #TODO
}

# TODO captured mutables + multiple calls to `back`
@generated function (back::Jnew{T,G,false})(Δ::allowed_gradient_T) where {T,G}
  Δ <: Union{Nothing, NamedTuple} && legacytype_warn(Δ)
  if !T.mutable
    Δ <: AbstractZero && return :Δ
  end
  Δ_expr = if G <: AbstractZero
    :Δ
  elseif Δ <: RefValue
    :(back.g[]) # TODO: is this right? Why don't we need to accum? 
  else
    :(accum(back.g[], Δ))
  end
  quote
    x̄ = $Δ_expr
    $(G <: AbstractZero || :(back.g[] = nt_zero($Δ_expr)))
    return (DoesNotExist(), $(map(fn -> :(x̄.$fn), fieldnames(T))...))
  end
end

@generated function (back::Jnew{T,G,true})(Δ::allowed_gradient_T) where {T,G}
  Δ == Union{Nothing, NamedTuple} && legacytype_warn(Δ)
  if !T.mutable
    Δ <: AbstractZero && return :Δ
  end
  if G <: AbstractZero
    quote
      return (DoesNotExist(), ($(map(fn -> :(Δ.$fn), fieldnames(T))...),))
    end
  else # TODO is this dead code? back is an (immutable) struct
    quote
      x̄ = back.g
      back.g = nt_zero(back.g)
      return (DoesNotExist(), ($(map(fn -> :(x̄.$fn), fieldnames(T))...),))
    end
  end
end

(back::Jnew{T})(Δ) where T = error("Need an adjoint for constructor $T. Gradient is of type $(typeof(Δ))")
