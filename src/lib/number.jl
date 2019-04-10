using DiffRules, SpecialFunctions, NaNMath

@nograd isinf, isnan, isfinite

const non_holomorphic = :[abs2, abs].args

# TODO use CSE here

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  dx = DiffRules.diffrule(M, f, :x)
  f in non_holomorphic || (dx = :(conj($dx)))
  @eval begin
    @adjoint $M.$f(x::Number) = $M.$f(x),
      Δ -> (Δ * $dx,)
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  @eval begin
    @adjoint $M.$f(a::Number, b::Number) = $M.$f(a, b),
      Δ -> (Δ * conj($da), Δ * conj($db))
  end
end

@adjoint Base.convert(T::Type{<:Real}, x::Real) = convert(T, x), ȳ -> (nothing, ȳ)
@adjoint (T::Type{<:Real})(x::Real) = T(x), ȳ -> (nothing, ȳ)

for T in Base.uniontypes(Core.BuiltinInts)
    @adjoint (::Type{T})(x::Core.BuiltinInts) = T(x), Δ -> (Δ,)
end

@adjoint Base.:+(xs...) = +(xs...), Δ -> map(_ -> Δ, xs)

@adjoint function sincos(x)
  s, c = sincos(x)
  (s, c), ((s̄, c̄),) -> (s̄*c - c̄*s,)
end

@adjoint a // b = (a // b, c̄ -> (c̄ * 1//b, - c̄ * a // b // b))

@nograd floor, ceil, trunc, round, hash

# Complex Numbers

@adjoint (T::Type{<:Complex})(re, im) = T(re, im), c̄ -> (nothing, real(c̄), imag(c̄))

@adjoint real(x::Complex) = real(x), r̄ -> (real(r̄) + zero(r̄)*im,)
@adjoint imag(x::Complex) = imag(x), ī -> (zero(ī) + real(ī)*im,)

DiffRules._abs_deriv(x::Complex) = x/abs(x)
