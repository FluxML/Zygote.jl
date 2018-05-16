using DiffRules, SpecialFunctions, NaNMath

grad(x::Real) = zero(x)
grad(x::Integer) = zero(float(x))

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    @grad $M.$f(x) = $M.$f(x),
      Δ -> (Δ * $(DiffRules.diffrule(M, f, :x)),)
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  @eval begin
    @grad $M.$f(a, b) = $M.$f(a, b),
      Δ -> (Δ * $da, Δ * $db)
  end
end
