function ChainRulesCore.rrule(
    ::ZygoteRuleConfig, ::typeof(convert), T::Type{<:Real}, x::Real
)
    convert_pullback(Δ) = (NoTangent(), NoTangent(), Δ)
    return convert(T, x), convert_pullback
end

function ChainRulesCore.rrule(
    ::ZygoteRuleConfig, ::typeof(Base.literal_pow), ::typeof(^), x::Number, ::Val{p}
) where {p}
    function literal_pow_pullback(Δ)
        df = conj(p * Base.literal_pow(^,x,Val(p-1)))
        # When the local derivative `df` is exactly zero (e.g. `x^2` at `x == 0`),
        # the gradient contribution is zero even if `Δ` is infinite or `NaN`. This
        # avoids spurious `NaN`s from `0 * Inf` for functions like `sqrt(x^2)`
        # at the cusp `x == 0` (see FluxML/Zygote.jl#1598).
        dx = iszero(df) ? zero(Δ * df) : Δ * df
        return (NoTangent(), NoTangent(), dx, NoTangent())
    end
    return Base.literal_pow(^,x,Val(p)), literal_pow_pullback
end

function ChainRulesCore.rrule(::ZygoteRuleConfig, T::Type{<:Real}, x::Real)
    Real_pullback(Δ) = (NoTangent(), Δ)
    return T(x), Real_pullback
end

for T in Base.uniontypes(Core.BuiltinInts)
    @eval function ChainRulesCore.rrule(::ZygoteRuleConfig, ::Type{$T}, x::Core.BuiltinInts)
        IntX_pullback(Δ) = (NoTangent(), Δ)
        return $T(x), IntX_pullback
    end
end

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(+), xs::Number...)
    plus_pullback(Δ) = (NoTangent(), map(_ -> Δ, xs)...)
    return +(xs...), plus_pullback
end

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(//), a, b)
    divide_pullback(r̄) = (NoTangent(), r̄ * 1//b, - r̄ * a // b // b)
    return a // b, divide_pullback
end

# `flipsign`/`copysign` are implemented with bit-twiddling intrinsics (`bitcast`,
# `xor_int`, ...) that are not differentiable on their own, so the source transform
# bottoms out at an intrinsic and errors. The functions themselves are differentiable:
# they only copy `y`'s sign bit onto the magnitude, so d/dx is ±1 (`flipsign(1, y)`)
# and d/dy is 0 a.e. Needed for e.g. `Colors.colordiff` (issues #467, #619).
function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(flipsign), x::Real, y::Real)
    flipsign_pullback(Δ) = (NoTangent(), flipsign(Δ, y), ZeroTangent())
    return flipsign(x, y), flipsign_pullback
end

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(copysign), x::Real, y::Real)
    copysign_pullback(Δ) = (NoTangent(), flipsign(flipsign(Δ, x), y), ZeroTangent())
    return copysign(x, y), copysign_pullback
end

# Complex Numbers

function ChainRulesCore.rrule(::ZygoteRuleConfig, T::Type{<:Complex}, r, i)
    Complex_pullback(c̄) = (NoTangent(), real(c̄), imag(c̄))
    return T(r, i), Complex_pullback
end

# we define these here because ChainRules.jl only defines them for x::Union{Real,Complex}

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(abs2), x::Number)
    abs2_pullback(Δ) = (NoTangent(), real(Δ)*(x + x))
    return abs2(x), abs2_pullback
end

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(real), x::Number)
    real_pullback(r̄) = (NoTangent(), real(r̄))
    return real(x), real_pullback
end

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(conj), x::Number)
    conj_pullback(c̄) = (NoTangent(), conj(c̄))
    return conj(x), conj_pullback
end

# for real x, ChainRules pulls back a zero real adjoint, whereas we treat x
# as embedded in the complex numbers and pull back a pure imaginary adjoint
function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(imag), x::Number)
    imag_pullback(ī) = (NoTangent(), real(ī)*im)
    return imag(x), imag_pullback
end
