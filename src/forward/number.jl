using DiffRules, SpecialFunctions, NaNMath, LogExpFunctions
using Base.FastMath: fast_op, make_fastmath

# TODO use CSE here

for (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
  if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
    @warn "$M.$f is not available and hence rule for it can not be defined"
    continue  # Skip rules for methods not defined in the current scope
  end
  if arity == 1
    dx = DiffRules.diffrule(M, f, :x)
    @eval begin
      @tangent $M.$f(x::Number) = $M.$f(x), ẋ -> ẋ * $dx
    end
  elseif arity == 2
    da, db = DiffRules.diffrule(M, f, :a, :b)
    @eval begin
      @tangent $M.$f(a::Number, b::Number) = $M.$f(a, b), (ȧ, ḃ) -> ȧ*$da + ḃ*$db
    end
  end
end

# Some specific overrides
# The DiffRules definitions are suboptimal due to repeated work in the tangent

@tangent function tanh(x)
  y = tanh(x)
  y, ẋ -> ẋ * (1 - y^2)
end

@tangent function exp(x)
  y = exp(x)
  y, ẋ -> ẋ * y
end

for f in [>, <, ==, ===, !=, in]
  @eval @tangent $f(a, b) = $f(a, b), (_, _) -> false
end

@tangent pyconvert(T::Type{<:Real}, x::Real) = pyconvert(T, x), (_, ẋ) -> pyconvert(T, ẋ)

@tangent function Colon()(xs...)
  c = Colon()(xs...)
  c, (_...) -> zerolike(c)
end

zerolike(x::AbstractRange) =
  invoke(zerolike, Tuple{Any}, x)

DiffRules._abs_deriv(x::Complex) = x/abs(x)
