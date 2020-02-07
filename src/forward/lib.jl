zerolike(x::Number) = zero(x)
zerolike(x::Tuple) = zerolike.(x)
zerolike(x::T) where T =
  NamedTuple{fieldnames(T)}(map(f -> zerolike(getfield(x, f)), fieldnames(T)))
zerolike(x::Union{Module,Type}) = nothing

# Julia internal definitions
_tangent(_, ::typeof(zerolike), x) = zerolike(x), zerolike(x)
_tangent(_, ::typeof(one), x) = one(x), zerolike(x)
_tangent(_, ::typeof(typeof), x) = typeof(x), nothing
_tangent(_, ::typeof(Core.apply_type), args...) = Core.apply_type(args...), nothing
_tangent(ṫ, ::typeof(tuple), t...) = t, tail(ṫ)
_tangent((_, ṫ, _), ::typeof(getfield), t, i) = getfield(t, i), getfield(ṫ, i)
_tangent((_, ṫ, _), ::typeof(getindex), t, i) = getindex(t, i), getindex(ṫ, i)
_tangent(ṡ, ::typeof(__new__), T, s...) = __new__(T, s...), NamedTuple{fieldnames(T)}(tail(tail(ṡ)))

# Mathematical definitions
_tangent((_, ȧ, ḃ), ::typeof(+), a, b) = a + b, ȧ + ḃ
_tangent((_, ȧ, ḃ), ::typeof(-), a, b) = a - b, ȧ - ḃ
_tangent((_, ȧ, ḃ), ::typeof(*), a, b) = a*b, ȧ*b+ḃ*a
_tangent((_, ȧ, ḃ), ::typeof(>), a, b) = a>b, false
_tangent((_, ȧ), ::typeof(-), a) = -a, -ȧ

_tangent((_, ẋ), ::typeof(sin), x) = sin(x), ẋ*cos(x)
_tangent((_, ẋ), ::typeof(cos), x) = cos(x), -ẋ*sin(x)
