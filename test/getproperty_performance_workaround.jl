struct YourType
    x::Float64
end

# Copy + paste suggestion from ZygoteRules docstring.
using ZygoteRules
using ZygoteRules: AContext, literal_getproperty, pullback_for_default_literal_getproperty

function ZygoteRules._pullback(
  cx::AContext, ::typeof(literal_getproperty), x::YourType, ::Val{f}
) where {f}
    return pullback_for_default_literal_getproperty(cx, x, Val{f}())
end

@testset "getproperty_performance_workaround" begin
    x = YourType(5.0)
    @inferred (x -> x.x)(x)
    @inferred Zygote._pullback(Zygote.Context(), x -> x.x, x)
    out, pb = Zygote._pullback(Zygote.Context(), x -> x.x, x)
    @inferred pb(4.0)
end
