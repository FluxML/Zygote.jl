"""
    dropgrad(x) -> x

Drop the gradient of `x`.

    julia> gradient(2, 3) do a, b
         dropgrad(a)*b
       end
    (nothing, 2)
"""
function dropgrad end

@adjoint dropgrad(x) = dropgrad(x), _ -> nothing

Base.@deprecate dropgrad(x) ChainRulesCore.ignore_derivatives(x)
