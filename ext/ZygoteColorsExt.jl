module ZygoteColorsExt

using Zygote
using Colors

Zygote.@non_differentiable Colors.ColorTypes._parameter_upper_bound(::Any...)

end
