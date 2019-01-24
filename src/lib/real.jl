using DiffRules, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    @adjoint $M.$f(x::Real) = $M.$f(x),
      Δ -> (Δ * $(DiffRules.diffrule(M, f, :x)),)
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  @eval begin
    @adjoint $M.$f(a::Real, b::Real) = $M.$f(a, b),
      Δ -> (Δ * $da, Δ * $db)
  end
end

@adjoint Base.convert(T::Type{<:Real}, x::Real) = convert(T, x), Δ -> (nothing, Δ)

@adjoint Base.:+(xs...) = +(xs...), Δ -> map(_ -> Δ, xs)

@adjoint function sincos(x)
  s, c = sincos(x)
  (s, c), ((s̄, c̄),) -> (s̄*c - c̄*s,)
end

@adjoint a // b = (a // b, c̄ -> (c̄ * 1//b, - c̄ * a // b // b))
