zerolike(x::Number) = zero(x)
zerolike(x::Tuple) = zerolike.(x)

@generated function zerolike(x::T) where T
  length(fieldnames(T)) == 0 ? nothing :
  :(NamedTuple{$(fieldnames(T))}(($(map(f -> :(zerolike(x.$f)), fieldnames(T))...),)))
end

# TODO figure out why this made a test fail
zerolike(x::Union{Module,Type}) = nothing

# Required to not get an UndefRefError on 1.10
zerolike(x::GlobalRef) = nothing

# TODO: `@non_differentiable` and `@linear`

@tangent zerolike(x) = zerolike(x), _ -> zerolike(x)
@tangent one(x::Number) = one(x), _ -> zero(x)
@tangent one(T::Type) = one(T), _ -> zero(T)
@tangent Core.Typeof(x) = Core.Typeof(x), _ -> nothing
@tangent typeof(x) = typeof(x), _ -> nothing
@tangent Core.apply_type(args...) = Core.apply_type(args...), (_...) -> nothing
@tangent fieldtype(args...) = fieldtype(args...), (_...) -> nothing
@tangent isa(a, b) = isa(a, b), (_, _) -> false
@tangent repr(x) = repr(x), _ -> nothing
@tangent println(x...) = println(x...), (_...) -> nothing
@tangent typeassert(x, T) = typeassert(x, T), (ẋ, _) -> ẋ
@tangent fieldnames(T) = fieldnames(T), _ -> zerolike(fieldnames(T))
@tangent eltype(x) = eltype(x), ẋ -> zerolike(eltype(ẋ))

@tangent fieldcount(T) = fieldcount(T), _ -> zerolike(fieldcount(T))

@tangent tuple(t...) = t, (ṫ...) -> ṫ
@tangent tail(t) = tail(t), tail

@tangent setfield!(t, i, x) = setfield!(t, i, x), (ṫ, _, ẋ) -> setfield!(ṫ, i, ẋ)
@tangent getindex(t, i) = getindex(t, i), (ṫ, _) -> getindex(ṫ, i)
@tangent isdefined(t, i) = isdefined(t, i), (_, _) -> false

# TODO should be using a context for this
zerolike(x::Core.Box) = isdefined(x, :contents) ? Core.Box(zerolike(x.contents)) : Core.Box()
@tangent Core.Box() = Core.Box(), () -> Core.Box()
@tangent Core.Box(x) = Core.Box(x), ẋ -> Core.Box(x)
# TODO: this is too generic (e.g. broadcast).
# @tangent Base.copy(x) = copy(@show x), ẋ -> copy(ẋ)

@tangent Core.Compiler.return_type(args...) =
  Core.Compiler.return_type(args...), (_...) -> nothing

@tangent __new__(T, s...) =
  __new__(T, s...), (_, ṡ...) -> NamedTuple{fieldnames(T)}(ṡ)

@tangent __splatnew__(T, s) =
  __splatnew__(T, s), (_, ṡ) -> NamedTuple{fieldnames(T)}(ṡ)

function _pushforward(dargs, ::typeof(Core._apply), f, args...)
  dargs = tail(dargs) # drop self gradient
  df, dargs = first(dargs), tail(dargs)
  dargs = Core._apply(tuple, dargs...)
  Core._apply(_pushforward, ((df, dargs...), f), args...)
end

if VERSION >= v"1.4.0-DEV.304"
  _pushforward(dargs, ::typeof(Core._apply_iterate), ::typeof(iterate), f, args...) =
    _pushforward((first(args), tail(tail(dargs))...), Core._apply, f, args...)
end

using ..Zygote: literal_getproperty, literal_getfield, literal_getindex

function _pushforward(dargs, ::typeof(literal_getproperty), x::NamedTuple,
                      ::Val{property_name}) where {property_name}
  return _pushforward(dargs, literal_getfield, x, Val(property_name))
end
_pushforward(dargs, ::typeof(getproperty), x::NamedTuple, f) =
  _pushforward(dargs, literal_getfield, x, Val(f))

_pushforward(dargs, ::typeof(getfield), x, f) =
  _pushforward(dargs, literal_getfield, x, Val(f))

@tangent function literal_getfield(t, ::Val{i}) where i
  y = getfield(t, i)
  forw(ṫ, _) = getfield(ṫ, i)
  forw(ṫ::Nothing, _) = zerolike(y)
  return y, forw
end

@tangent literal_getindex(t, ::Val{i}) where i =
  getindex(t, i), (ṫ, _) -> getindex(ṫ, i)

@tangent getfield(t::Tuple, i::Integer) =
  getfield(t, i), (ṫ, _) -> getfield(ṫ, i)
