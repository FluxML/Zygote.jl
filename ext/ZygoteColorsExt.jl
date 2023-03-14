module ZygoteColorsExt

if isdefined(Base, :get_extension)
    using Zygote
    using Colors
else
    using ..Zygote
    using ..Colors
end

Zygote.@non_differentiable Colors.ColorTypes._parameter_upper_bound(::Any...)

end
