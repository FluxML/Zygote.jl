using DiffRules, SpecialFunctions, NaNMath

@nograd isinf, isnan, isfinite

# TODO use CSE here

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

# Hack for conversions

using ForwardDiff: Dual

(T::Type{<:Real})(x::Dual) = Dual(T(x.value), map(T, x.partials.values))
(Dual{T,V,N})(x::Dual) where {T,V,N} = invoke(Dual{T,V,N}, Tuple{Number}, x)
