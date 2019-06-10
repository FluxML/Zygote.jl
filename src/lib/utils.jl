"""
    dropgrad(x) -> x

Drop the gradient of `x`.

    julia> gradient(2, 3) do a, b
         dropgrad(a)*b
       end
    (nothing, 2)
"""
dropgrad(x) = x
@adjoint dropgrad(x) = dropgrad(x), _ -> nothing

"""
    hook(x̄ -> ..., x) -> x

Gradient hooks. Allows you to apply an arbitrary function to the gradient for
`x`.

    julia> gradient(2, 3) do a, b
             hook(ā -> @show(ā), a)*b
           end
    ā = 3
    (3, 2)

    julia> gradient(2, 3) do a, b
             hook(-, a)*b
           end
    (-3, 2)
"""
hook(f, x) = x

@adjoint! hook(f, x) = x, x̄ -> (nothing, f(x̄),)

"""
    @showgrad(x) -> x

Much like `@show`, but shows the gradient about to accumulate to `x`. Useful for
debugging gradients.

    julia> gradient(2, 3) do a, b
             @showgrad(a)*b
           end
    ∂(a) = 3
    (3, 2)

Note that the gradient depends on how the output of `@showgrad` is *used*, and is
not the *overall* gradient of the variable `a`. For example:

    julia> gradient(2) do a
         @showgrad(a)*a
       end
    ∂(a) = 2
    (4,)

    julia> gradient(2, 3) do a, b
             @showgrad(a) # not used, so no gradient
             a*b
           end
    ∂(a) = nothing
    (3, 2)
"""
macro showgrad(x)
  :(hook($(esc(x))) do x̄
      println($"∂($x) = ", repr(x̄))
      x̄
    end)
end

"""
    hessian(f, x)

Construct the Hessian of `f`, where `x` is a real or real array and `f(x)` is
a real.

    julia> hessian(((a, b),) -> a*b, [2, 3])
    2×2 Array{Int64,2}:
     0  1
     1  0
"""
hessian(f, x::AbstractArray) = forward_jacobian(x -> gradient(f, x)[1], x)[2]
