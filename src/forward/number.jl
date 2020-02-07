using DiffRules, SpecialFunctions, NaNMath
using Base.FastMath: fast_op, make_fastmath

# TODO use CSE here

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  dx = DiffRules.diffrule(M, f, :x)
  @eval begin
    @tangent $M.$f(x::Number) = $M.$f(x), ẋ -> ẋ * $dx
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  @eval begin
    @tangent $M.$f(a::Number, b::Number) = $M.$f(a, b), (ȧ, ḃ) -> ȧ*$da + ḃ*$db
  end
end

for f in [>, <, ==, !=, in]
  @eval @tangent $f(a, b) = $f(a, b), (_, _) -> false
end
