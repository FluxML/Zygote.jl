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
    ignore() do
      ...
    end

Tell Zygote to ignore a block of code. Everything inside the `do` block will run
on the forward pass as normal, but Zygote won't try to differentiate it at all.
This can be useful for e.g. code that does logging of the forward pass.

Obviously, you run the risk of incorrect gradients if you use this incorrectly.
"""
ignore(f) = f()
@adjoint ignore(f) = ignore(f), _ -> nothing

"""
    @ignore (...)

Tell Zygote to ignore an expression. Equivalent to `ignore() do (...) end`.
Example:

```julia-repl
julia> f(x) = (y = Zygote.@ignore x; x * y);
julia> f'(1)
1
```
"""
macro ignore(ex)
    return :(Zygote.ignore() do
        $(esc(ex))
    end)
end

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
   isderiving()
   isderiving(x)

Check whether the current function call is happening while taking the derivative.


    julia> function f(x)
             @show isderiving()
           end

    f (generic function with 1 method)

    julia> f(3)
    isderiving() = false
    false

    julia> gradient(f, 4)
    isderiving() = true
    (nothing,)
"""
isderiving() = false
isderiving(x) = false

@adjoint isderiving() = true, _ -> nothing
@adjoint isderiving(x) = true, x -> (nothing,)
