zerolike(x::Number) = zero(x)
zerolike(x::Tuple) = zerolike.(x)

@generated function zerolike(x::T) where T
  :(NamedTuple{$(fieldnames(T))}(($(map(f -> :(zerolike(x.$f)), fieldnames(T))...),)))
end

# TODO figure out why this made a test fail
zerolike(x::Union{Module,Type}) = false

# TODO: `@nograd` and `@linear`

@tangent zerolike(x) = zerolike(x), _ -> zerolike(x)
@tangent one(x) = one(x), _ -> zerolike(x)
@tangent typeof(x) = typeof(x), _ -> nothing
@tangent Core.apply_type(args...) = Core.apply_type(args...), (_...) -> nothing
@tangent fieldtype(args...) = fieldtype(args...), (_...) -> nothing
@tangent isa(a, b) = isa(a, b), (_, _) -> false
@tangent repr(x) = repr(x), _ -> nothing
@tangent println(x...) = println(x...), (_...) -> nothing
@tangent typeassert(x, T) = typeassert(x, T), (ẋ, _) -> ẋ
@tangent fieldnames(T) = fieldnames(T), _ -> zerolike(fieldnames(T))
@tangent fieldcount(T) = fieldcount(T), _ -> zerolike(fieldcount(T))

@tangent tuple(t...) = t, (ṫ...) -> ṫ
@tangent tail(t) = tail(t), ṫ -> tail(ṫ)

@tangent getfield(t, i) = getfield(t, i), (ṫ, _) -> getfield(ṫ, i)
@tangent setfield!(t, i, x) = setfield!(t, i, x), (ṫ, _, ẋ) -> setfield!(ṫ, i, ẋ)
@tangent getindex(t, i) = getindex(t, i), (ṫ, _) -> getindex(ṫ, i)
@tangent isdefined(t, i) = isdefined(t, i), (_, _) -> false

# TODO should be using a context for this
zerolike(x::Core.Box) = isdefined(x, :contents) ? Core.Box(zerolike(x.contents)) : Core.Box()
@tangent Core.Box() = Core.Box(), () -> Core.Box()
@tangent Core.Box(x) = Core.Box(x), ẋ -> Core.Box(x)

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
