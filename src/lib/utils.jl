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

"""
    hessian(f, x)

Construct the Hessian `∂²f/∂x∂x`, where `x` is a real number or an array,
and `f(x)` is a real number.

Uses forward over reverse, ForwardDiff over Zygote, calling `hessian_dual(f, x)`.
See [`hessian_reverse`](@ref) for an all-Zygote version.

# Examples

```jldoctest; setup=:(using Zygote)
julia> Zygote.hessian(x -> x[1]*x[2], randn(2))
2×2 Array{Float64,2}:
 0.0  1.0
 1.0  0.0

julia> Zygote.hessian(x -> sum(x.^3), [1 2; 3 4])  # uses linear indexing of x
4×4 Array{$Int,2}:
 6   0   0   0
 0  18   0   0
 0   0  12   0
 0   0   0  24

julia> Zygote.hessian(sin, pi/2)
-1.0
```
"""
hessian(f, x::AbstractArray) = forward_jacobian(x -> gradient(f, x)[1], x)[2]

hessian(f, x) = hessian_dual(f, x)

hessian_dual(f, x::AbstractArray) = forward_jacobian(x -> gradient(f, x)[1], x)[2]

hessian_dual(f, x::Number) = ForwardDiff.derivative(x -> gradient(f, x)[1], x)

"""
    hessian_reverse(f, x)

This should be equivalent to [`hessian(f, x)`](@ref hessian),
but implemented using reverse over reverse mode, all Zygote.
(This is usually much slower, and more likely to find errors.)
"""
hessian_reverse(f, x::AbstractArray) = jacobian(x -> gradient(f, x)[1], x)[1]

hessian_reverse(f, x::Number) = gradient(x -> gradient(f, x)[1], x)[1]

"""
    jacobian(f, args...)

For each array `a ∈ args` this returns a matrix with `Ja[k,i] = ∂y[k]/∂a[i]`
where `y = f(args...)` is usually a vector.
For scalar `x::Number ∈ args`, the result `Jx[k,1] = ∂y[k]/∂x` is a vector,
while for scalar `y` all results have just one row.

For any other argument type, no result is produced, even if [`gradient`](@ref) would work.

This reverse-mode Jacobian needs to evaluate the pullback once for each element of `y`.
This is usually only efficient when `length(y)` is small compared to `length(a)`,
otherwise forward mode is likely to be better.

See also [`hessian`](@ref), [`hessian_reverse`](@ref).

# Examples

```jldoctest
julia> jacobian(a -> 100*a[1:3].^2, 1:7)[1]  # first index (rows) is output
3×7 Array{$Int,2}:
 200    0    0  0  0  0  0
   0  400    0  0  0  0  0
   0    0  600  0  0  0  0

julia> jacobian((a,x) -> a.^2 .* x, [1,2,3], 1)  # scalar argument has vector jacobian
([2 0 0; 0 4 0; 0 0 6], [1, 4, 9])

julia> jacobian((a,d) -> prod(a, dims=d), [1 2; 3 4; 5 6], 2)
([2 0 … 0 0; 0 4 … 3 0; 0 0 … 0 5], [0, 0, 0])
```

!!! warning
    For arguments of any type except `Number` & `AbstractArray`, the result is `nothing`.

```jldoctest
julia> jacobian((a,s) -> a.^length(s), [1,2,3], "str")
([3 0 0; 0 12 0; 0 0 27], nothing)

julia> jacobian((a,t) -> sum(a .* t[1]) + t[2], [1,2,3], (4,5))
([4 4 4], nothing)

julia> gradient((a,t) -> sum(a .* t[1]) + t[2], [1,2,3], (4,5))  # gradient undersands the tuple
([4, 4, 4], (6, 1))
```
"""
function jacobian(f, args...)
  y, back = pullback(_jvec∘f, args...)
  out = map(args) do x
    T = promote_type(eltype(x), eltype(y))
    dx = x isa AbstractArray ? similar(x, T, length(y), length(x)) :
      x isa Number ? similar(y, T, length(y)) :
      nothing
  end
  # delta = diagm(fill!(similar(y), 1))
  delta = fill!(similar(y, length(y), length(y)), 0)
  delta[LinearAlgebra.diagind(delta)] .= 1
  for k in LinearIndices(y)
    grads = back(delta[:,k])
    for (dx, grad) in zip(out, grads)
      dx isa AbstractArray || continue
      _gradcopy!(view(dx,k,:), grad)
    end
  end
  out
end

_jvec(x::AbstractArray) = vec(x)
_jvec(x::Number) = _jvec(vcat(x))
_jvec(x) = throw(ArgumentError("jacobian expected a function which returns an array, or a scalar, got $(typeof(x))"))
_jvec(x::AbstractArray{<:Complex}) = throw(ArgumentError("jacobian does not accept complex output"))

_gradcopy!(dst::AbstractArray, src::AbstractArray{<:Number}) = copyto!(dst, src)
_gradcopy!(dst::AbstractArray, src::Number) = copyto!(dst, src)
_gradcopy!(dst::AbstractArray, src::Nothing) = dst .= 0
_gradcopy!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, g isa Number ? g : 0 for g in src) # e.g. Union{Nothing,Float64}

"""
    jacobian(loss, ::Params)

Like `gradient` with implicit parameters, this method takes a zero-argument function
and returns an `IdDict`-like object, now containing the Jacobian for each parameter.

# Examples
```jldoctest
julia> xs = [1 2; 3 4]; ys = [5,7,9];

julia> Jxy = jacobian(() -> ys[1:2] .+ sum(xs.^2), Params([xs, ys]))
Grads(...)

julia> Jxy[ys]
2×3 Array{$Int,2}:
 1  0  0
 0  1  0

julia> Jxy[xs]
2×4 Array{$Int,2}:
 2  6  4  8
 2  6  4  8
```
"""
function jacobian(f, pars::Params)
  y, back = pullback(_jvec∘f, pars)
  out = IdDict()
  for p in pars
    T = Base.promote_type(eltype(p), eltype(y))
    J = similar(y, T, length(y), length(p))
    out[p] = J
  end
  delta = fill!(similar(y, length(y), length(y)), 0)
  delta[LinearAlgebra.diagind(delta)] .= 1
  for k in LinearIndices(y)
    grads = back(delta[:,k])
    for p in pars
      out[p] isa AbstractArray || continue
      _gradcopy!(view(out[p],k,:), grads[p])
    end
  end
  Grads(out, pars)
end

