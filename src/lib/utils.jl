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

"""
    jacobian(f, args...)

If `y = f(args...)` is a vector, then for each vector `a ∈ args`
this returns a matrix with `J_a[o,i] = ∂yₒ/∂aᵢ`.

For scalar `x ∈ args` it returns a vector with `J_x[o] = ∂yₒ/∂x`,
and for other shapes `size(J_arg) = (size(y)..., size(arg)...)`.

Keyword `matrix=true` reshapes to treat both output `y` and
every `arg` as a vector, hence every `J_arg` a matrix.
This will give an error on scalar `y`.
"""
function jacobian(f, args...; matrix::Bool = false)
    res, back = Zygote.forward(f, args...)
    if !(res isa AbstractArray)
        matrix && error("jacobian(f, args...; matrix=true) cannot " *
            "handle scalar output, try gradient(f, args...)")
        return gradient(f, args...)
    end
    out = map(args) do p
        T = Base.promote_type(eltype(p), eltype(res))
        similar(res, T, size(res)..., size(p)...)
    end
    delta = fill!(similar(res), 0)
    for k in CartesianIndices(res)
        delta[k] = 1
        grads = back(delta)
        for (g,o) in zip(grads, out)
            c = map(_->(:), size(g))
            o[k,c...] .= g
        end
        delta[k] = 0
    end
    matrix ? reshape.(out, length(res), :) : out
end

function jacobian(f, ps::Params) # Union{Tracker.Params, Zygote.Params}
    res, back = forward(f, ps)
    out = IdDict()
    for p in ps
        T = Base.promote_type(eltype(p), eltype(res))
        J = similar(res, T, size(res)..., size(p)...)
        out[p] = J
    end
    delta = fill!(similar(res), 0)
    for k in CartesianIndices(res)
        delta[k] = 1
        grads = back(delta)
        for p in ps
            g = grads[p]
            c = map(_->(:), size(g))
            o = out[p]
            o[k,c...] .= g
        end
        delta[k] = 0
    end
    out
end
