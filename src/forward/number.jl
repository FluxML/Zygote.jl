for f in [>, <, ==, ===, !=, in]
  @eval @tangent $f(a, b) = $f(a, b), (_, _) -> false
end

@tangent convert(T::Type{<:Real}, x::Real) = convert(T, x), (_, ẋ) -> convert(T, ẋ)

@tangent function Colon()(xs...)
  c = Colon()(xs...)
  c, (_...) -> zerolike(c)
end

zerolike(x::AbstractRange) = invoke(zerolike, Tuple{Any}, x)
