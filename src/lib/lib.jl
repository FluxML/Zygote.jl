using Base: RefValue

if VERSION > v"1.7.0-DEV.204"
  using Base: ismutabletype
else
  function ismutabletype(@nospecialize(t::Type))
    t = Base.unwrap_unionall(t)
    return isa(t, DataType) && t.mutable
  end
end

# Interfaces

accum() = nothing
accum(x) = x

accum(x, y) =
  x === nothing ? y :
  y === nothing ? x :
  x + y

accum(x, y, zs...) = accum(accum(x, y), zs...)

accum(x::Tuple, ys::Tuple...) = map(accum, x, ys...)
accum(x::AbstractArray, ys::AbstractArray...) = accum.(x, ys...)

@generated function accum(x::NamedTuple, y::NamedTuple)
  # assumes that y has no keys apart from those also in x
  fieldnames(y) ⊆ fieldnames(x) || throw(ArgumentError("$y keys must be a subset of $x keys"))

  grad(field) = field in fieldnames(y) ? :(y.$field) : :nothing
  Expr(:tuple, [:($f=accum(x.$f, $(grad(f)))) for f in fieldnames(x)]...)
end

function accum(x::RefValue, y::RefValue)
  @assert x === y
  return x
end

# Core functions
@_adjoint_keepthunks deepcopy(x) = deepcopy(x), ȳ -> (ȳ,)

@_adjoint_keepthunks (::Type{V})(x...) where V<:Val = V(x...), _ -> nothing

@_adjoint_keepthunks ifelse(cond::Bool, t, f) =
  ifelse(cond, t, f),
  Δ -> cond ? (nothing, Δ, zero(Δ)) : (nothing, zero(Δ), Δ)

@_adjoint_keepthunks Base.typeassert(x, T) = Base.typeassert(x, T), Δ -> (Δ, nothing)

accum_param(::Context{false}, _, Δ) = Δ
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
  gs[ref] = accum(get(gs, ref, nothing), x̄)
  return
end

unwrap(x) = x

@_adjoint_keepthunks unwrap(x) = unwrap(x), x̄ -> (accum_param(__context__, x, x̄),)

unwrap(ref, x) = x

@_adjoint_keepthunks unwrap(ref, x) = unwrap(x), function (x̄)
  accum_global(__context__, ref, x̄)
  (accum_param(__context__, x, x̄),)
end

function global_set(ref, val)
  @static if VERSION < v"1.9.0-DEV.265"
    ccall(:jl_set_global, Cvoid, (Any, Any, Any),
          ref.mod, ref.name, val)
  else
    setglobal!(ref.mod, ref.name, val)
  end
end

@_adjoint_keepthunks! function global_set(ref, x)
  global_set(ref, x), function (x̄)
    gs = cache(__context__)
    x̄ = accum(get(gs, ref, nothing), x̄)
    gs[ref] = nothing
    return (nothing, x̄)
  end
end

# Tuples

using Base: tail

@_adjoint_keepthunks tuple(xs...) = xs, identity

@_adjoint_keepthunks function literal_getindex(xs::NTuple{N,Any}, ::Val{i}) where {N,i}
  val = xs[i]
  function back(Δ)
    accum_param(__context__, val, Δ) === nothing && return
    return ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing
  end
  val, back
end

@_adjoint_keepthunks function getindex(xs::NTuple{N,Any}, i::Integer) where N
  val = xs[i]
  function back(Δ)
    accum_param(__context__, val, Δ) === nothing && return
    return ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing
  end
  return val, back
end

@_adjoint_keepthunks getindex(xs::NTuple{N,Any}, r::AbstractUnitRange) where N =
  (xs[r], Δ -> (ntuple(j -> j in r ? Δ[findfirst(isequal(j), r)] : nothing, Val(N)), nothing))

@_adjoint_keepthunks function getindex(xs::NTuple{N,Any}, r::AbstractVector) where N
  val = xs[r]
  function back(Δ)
    dxs = ntuple(Val(length(xs))) do x
      total = nothing
      for r_i in eachindex(r)
        r[r_i] === x || continue
        total = accum(total, Δ[r_i])
      end
      return total
    end
    return (dxs, nothing)
  end
  val, back
end

function _pullback(cx::AContext, ::typeof(literal_indexed_iterate), xs::Tuple, ::Val{i}) where i
  y, b = _pullback(cx, literal_getindex, xs, Val(i))
  back(::Nothing) = nothing
  back(ȳ) = b(ȳ[1])
  (y, i+1), back
end

function _pullback(cx::AContext, ::typeof(literal_indexed_iterate), xs::Tuple, ::Val{i}, st) where i
  y, b = _pullback(cx, literal_getindex, xs, Val(i))
  back(::Nothing) = nothing
  back(ȳ) = (b(ȳ[1])..., nothing)
  (y, i+1), back
end

# Needed for iteration lowering
@_adjoint_keepthunks Core.getfield(xs::NTuple{N,Any}, i::Int) where N =
  (xs[i], Δ -> (ntuple(j -> i == j ? Δ : nothing, Val(N)), nothing))

@_adjoint_keepthunks Core.getfield(xs::NamedTuple{K,<:NTuple{N,Any}}, i::Int) where {K,N} =
  (xs[i], Δ -> (NamedTuple{K}(ntuple(j -> i == j ? Δ : nothing, Val(N))), nothing))

@_adjoint_keepthunks function Base.first(xs::Tuple)
  drest = map(_->nothing, tail(xs))
  first(xs), Δ -> ((Δ, drest...),)
end

@_adjoint_keepthunks Base.tail(xs::Tuple) = tail(xs), x̄s -> ((nothing, x̄s...),)

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

@_adjoint_keepthunks! function Core._apply(f, args...)
  y, back = Core._apply(_pullback, (__context__, f), args...)
  st = map(_empty, args)
  y, function (Δ)
    Δ = back(Δ)
    Δ === nothing ? nothing :
      (first(Δ), unapply(st, Base.tail(Δ))...)
  end
end

if VERSION >= v"1.4.0-DEV.304"
  @_adjoint_keepthunks! function Core._apply_iterate(::typeof(iterate), f, args...)
    y, back = Core._apply(_pullback, (__context__, f), args...)
    st = map(_empty, args)
    y, function (Δ)
      Δ = back(Δ)
      Δ === nothing ? nothing :
        (nothing, first(Δ), unapply(st, Base.tail(Δ))...)
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

@generated pair(::Val{k}, v, _=nothing) where k = :($k = v,)
@generated pair(::Val{k}, v, ::NamedTuple{keys}) where {k,keys} = k isa Int ? :($(getfield(keys, k)) = v,) : :($k = v,)

@_adjoint_keepthunks function literal_getfield(x, ::Val{f}) where f
  val = getfield(x, f)
  function back(Δ)
    accum_param(__context__, val, Δ) === nothing && return
    if isimmutable(x)
      dx = (; nt_nothing(x)..., pair(Val(f), Δ, x)...)
      (_project(x, dx), nothing)
    else
      dx = grad_mut(__context__, x)
      dx[] = (; dx[]..., pair(Val(f), accum(getfield(dx[], f), Δ))...)
      return (dx,nothing)
    end
  end
  unwrap(val), back
end

_pullback(cx::AContext, ::typeof(getfield), x, field_name::Symbol) =
  _pullback(cx, literal_getfield, x, Val(field_name))

function _pullback(cx::AContext, ::typeof(literal_getproperty), x::NamedTuple,
                   ::Val{property_name}) where {property_name}
  return _pullback(cx, literal_getfield, x, Val(property_name))
end
function _pullback(cx::AContext, ::typeof(literal_getindex), x::NamedTuple,
                   ::Val{key}) where {key}
  return _pullback(cx, literal_getfield, x, Val(key))
end

function _pullback(cx::AContext, ::typeof(literal_getproperty), x::Tuple,
                   ::Val{index}) where {index}
  return _pullback(cx, literal_getindex, x, Val(index))
end
function _pullback(cx::AContext, ::typeof(literal_getfield), x::Tuple,
                   ::Val{index}) where {index}
  return _pullback(cx, literal_getindex, x, Val(index))
end

grad_mut(x) = Ref{Any}(nt_nothing(x))

function grad_mut(cx::Context, x)
  ch = cache(cx)
  if haskey(ch, x)
    ch[x]
  else
    ch[x] = grad_mut(x)
  end
end

@_adjoint_keepthunks! function setfield!(x, f, val)
  y = setfield!(x, f, val)
  g = grad_mut(__context__, x)
  y, function (_)
    Δ = getfield(g[], f)
    g[] = (;g[]...,pair(Val(f),nothing)...)
    (nothing, nothing, Δ)
  end
end

struct Jnew{T,G,splat}
  g::G
end

Jnew{T}(g) where T = Jnew{T,typeof(g)}(g)

@_adjoint_keepthunks! function __new__(T, args...)
  x = __new__(T, args...)
  g = !ismutabletype(T) || fieldcount(T) == 0 ? nothing : grad_mut(__context__, x)
  x, Jnew{T,typeof(g),false}(g)
end

@_adjoint_keepthunks! function __splatnew__(T, args)
  x = __splatnew__(T, args)
  g = !ismutabletype(T) || fieldcount(T) == 0 ? nothing : grad_mut(__context__, x)
  x, Jnew{T,typeof(g),true}(g)
end

# TODO captured mutables + multiple calls to `back`
@generated function (back::Jnew{T,G,false})(Δ::Union{NamedTuple,Nothing,RefValue}) where {T,G}
  !ismutabletype(T) && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ :
      Δ <: RefValue ? :(back.g[]) :
      :(accum(back.g[], Δ))
  quote
    x̄ = $Δ
    $(G == Nothing || :(back.g[] = nt_nothing($Δ)))
    (nothing, $(map(f -> :(x̄.$f), fieldnames(T))...))
  end
end

@generated function (back::Jnew{T,G,true})(Δ::Union{NamedTuple,Nothing,RefValue}) where {T,G}
  !ismutabletype(T) && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ : :(back.g)
  quote
    x̄ = $Δ
    $(G == Nothing || :($Δ = nt_nothing($Δ)))
    (nothing, ($(map(f -> :(x̄.$f), fieldnames(T))...),))
  end
end

(back::Jnew{T})(Δ) where T = error("Need an adjoint for constructor $T. Gradient is of type $(typeof(Δ))")
