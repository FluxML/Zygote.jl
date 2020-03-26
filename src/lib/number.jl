using DiffRules, SpecialFunctions, NaNMath
using Base.FastMath: fast_op, make_fastmath

@nograd isinf, isnan, isfinite, div

# TODO use CSE here

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  Δ = :Δ
  dx = DiffRules.diffrule(M, f, :x)
  if f in [:abs, :abs2]
    Δ = :(real($Δ))
  else
    dx = :(conj($dx))
  end
  @eval begin
    @adjoint $M.$f(x::Number) = $M.$f(x),
      Δ -> ($Δ * $dx,)
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  f == :^ && continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  @eval begin
    @adjoint $M.$f(a::Number, b::Number) = $M.$f(a, b),
      Δ -> (Δ * conj($da), Δ * conj($db))
  end
end

@adjoint Base.:^(x::Number, p::Number) = x^p,
  Δ -> (Δ * conj(p * x^(p-1)), Δ * conj(x^p * log(complex(x))))
@adjoint Base.literal_pow(::typeof(^), x::Number, ::Val{p}) where {p} =
  Base.literal_pow(^,x,Val(p)),
  Δ -> (nothing, Δ * conj(p * Base.literal_pow(^,x,Val(p-1))), nothing)

@adjoint Base.convert(T::Type{<:Real}, x::Real) = convert(T, x), ȳ -> (nothing, ȳ)
@adjoint (T::Type{<:Real})(x::Real) = T(x), ȳ -> (nothing, ȳ)

for T in Base.uniontypes(Core.BuiltinInts)
    @adjoint (::Type{T})(x::Core.BuiltinInts) = T(x), Δ -> (Δ,)
end

@adjoint Base.:+(xs::Number...) = +(xs...), Δ -> map(_ -> Δ, xs)

@adjoint Base.muladd(x::Number, y::Number, z::Number) =
  Base.muladd(x, y, z), ō -> (y'ō, x'ō, ō)

@adjoint Base.fma(x::Number, y::Number, z::Number) =
  Base.fma(x, y, z), ō -> (y'ō, x'ō, ō)

@adjoint function sincos(x)
  s, c = sincos(x)
  (s, c), ((s̄, c̄),) -> (s̄*c - c̄*s,)
end

@adjoint acosh(x::Complex) =
  acosh(x), Δ -> (Δ * conj(inv(sqrt(x - 1) * sqrt(x + 1))),)

@adjoint a // b = (a // b, c̄ -> (c̄ * 1//b, - c̄ * a // b // b))

@nograd floor, ceil, trunc, round, hash

# Complex Numbers

@adjoint (T::Type{<:Complex})(re, im) = T(re, im), c̄ -> (nothing, real(c̄), imag(c̄))

@adjoint real(x::Number) = real(x), r̄ -> (real(r̄),)
@adjoint conj(x::Number) = conj(x), r̄ -> (conj(r̄),)
@adjoint imag(x::Number) = imag(x), ī -> (real(ī)*im,)

DiffRules._abs_deriv(x::Complex) = x/abs(x)

 # adjoint for Fastmath operations
for (f, fastf) in fast_op
  if DiffRules.hasdiffrule(:Base, f, 1)
    dx = DiffRules.diffrule(:Base, f, :x)
    Δ = :Δ
    if f in [:abs, :abs2]
      Δ = :(real($Δ))
    else
      dx = :(conj($dx))
    end
    @eval begin
      @adjoint Base.FastMath.$fastf(x::Number) =
        Base.FastMath.$fastf(x), Δ -> ($Δ * make_fastmath($dx),)
    end
  elseif DiffRules.hasdiffrule(:Base, f, 2)
    dx, dy = DiffRules.diffrule(:Base, f, :x, :y)
    @eval begin
      @adjoint Base.FastMath.$fastf(x::Number, y::Number) =
        Base.FastMath.$fastf(x, y),
        Δ -> (Δ * make_fastmath(conj($dx)), Δ * make_fastmath(conj($dy)))
    end
  end
end
