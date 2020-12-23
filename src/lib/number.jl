
@nograd floor, ceil, trunc, round, div

@adjoint Base.literal_pow(::typeof(^), x::Number, ::Val{p}) where {p} =
  Base.literal_pow(^,x,Val(p)),
  Δ -> (nothing, Δ * conj(p * Base.literal_pow(^,x,Val(p-1))), nothing)

@adjoint Base.convert(T::Type{<:Real}, x::Real) = convert(T, x), ȳ -> (nothing, ȳ)
@adjoint (T::Type{<:Real})(x::Real) = T(x), ȳ -> (nothing, ȳ)

for T in Base.uniontypes(Core.BuiltinInts)
    @adjoint (::Type{T})(x::Core.BuiltinInts) = T(x), Δ -> (Δ,)
end

@adjoint Base.:+(xs::Number...) = +(xs...), Δ -> map(_ -> Δ, xs)

@adjoint a // b = (a // b, c̄ -> (c̄ * 1//b, - c̄ * a // b // b))

# Complex Numbers

@adjoint (T::Type{<:Complex})(re, im) = T(re, im), c̄ -> (nothing, real(c̄), imag(c̄))

# we define these here because ChainRules.jl only defines them for x::Union{Real,Complex}

@adjoint abs2(x::Number) = abs2(x), Δ -> (real(Δ)*(x + x),)
@adjoint real(x::Number) = real(x), r̄ -> (real(r̄),)
@adjoint conj(x::Number) = conj(x), r̄ -> (conj(r̄),)
@adjoint imag(x::Number) = imag(x), ī -> (real(ī)*im,)

# for real x, ChainRules pulls back a zero real adjoint, whereas we treat x
# as embedded in the complex numbers and pull back a pure imaginary adjoint
@adjoint imag(x::Real) = zero(x), ī -> (real(ī)*im,)
