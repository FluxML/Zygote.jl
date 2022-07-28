function ChainRulesCore.rrule(
    ::ZygoteRuleConfig, ::typeof(Base.literal_pow), ::typeof(^), x::Number, ::Val{p}
) where {p}
    function literal_pow_pullback(Δ)
        dx = Δ * conj(p * Base.literal_pow(^,x,Val(p-1)))
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

function ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(imag), x::Number)
    imag_pullback(ī) = (NoTangent(), real(ī)*im)
    return imag(x), imag_pullback
end
