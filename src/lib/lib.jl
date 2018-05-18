# Interfaces

@generated function grad(x)
  (x.mutable || nfields(x) == 0) && return
  Expr(:tuple, [:($f = grad(x.$f)) for f in fieldnames(x)]...)
end

grad(x::Tuple) = grad.(x)

accum(x, y) = x + y
accum(x, ::Void) = x
accum(::Void, _) = nothing
accum(x::Tuple, y::Tuple) = accum.(x, y)

@generated function accum(x::NamedTuple, y::NamedTuple)
  grad(x) = x in fieldnames(y) ? :(y.$x) : :nothing
  Expr(:tuple, [:($f=accum(x.$f, $(grad(f)))) for f in fieldnames(x)]...)
end

using MacroTools: combinedef

macro grad(ex)
  def = splitdef(ex)
  pushfirst!(def[:args], :(::typeof($(def[:name]))))
  def[:name] = :_forward
  def[:body] = quote
    Base.@_inline_meta
    y, back = $(def[:body])
    y, Δ -> (Base.@_inline_meta; (nothing, back(Δ)::Tuple...))
  end
  combinedef(def)
end

# Tuples

@grad tuple(xs...) = xs, Δ -> Δ == nothing ? map(_ -> nothing, xs) : Δ

@grad getindex(xs::NTuple{N}, i::Integer) where N =
  (xs[i], Δ -> (ntuple(j -> i == j ? Δ : nothing, Val{N}), nothing))

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
  y, J = Core._apply(_forward, (f,), args...)
  y, function (Δ)
    Δ = J(Δ)
    (first(Δ), unapply(args, Base.tail(Δ))...)
 end
end

# Structs

@generated nt_nothing(x) = Expr(:tuple, [:($f=nothing) for f in fieldnames(x)]...)

@generated pair(::Val{k}, v) where k = :($k = v,)

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
