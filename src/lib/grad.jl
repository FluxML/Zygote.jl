macro which(ex)
  @capture(ex, f_(args__)) || error("Zygote.@which f(args...)")
  :(InteractiveUtils.@which adjoint(Context(), $(esc(f)), $(esc.(args)...)))
end

"""

    checkpointed(f, xs...)

Use gradient checkpointing on the call `f(xs...)`. This means that
`checkpointed(f, xs...) === f(xs...)`, but when computing the derivative
intermediate results from the forward pass of `f` will not be stored. Instead the forward
pass will be repeated, when computing the derivative.
This saves memory at the cost of increasing execution time.

!!! warning
    If `f` is not a pure function, `checkpointed` will likely give wrong results.
"""
checkpointed(f, xs...) = f(xs...)

function Zygote._pullback(ctx::Zygote.AContext, ::typeof(checkpointed), f, xs...)
    y = f(xs...)
    function pullback_checkpointed(Δy)
        y, pb = Zygote._pullback(ctx, f, xs...)
        return (nothing, pb(Δy)...)
    end
    return y, pullback_checkpointed
end

"""
    hessian(f, x)

Construct the Hessian `∂²f/∂x²`, where `x` is a real number or an array,
and `f(x)` is a real number. When `x` is an array, the result is a matrix
`H[i,j] = ∂²f/∂x[i]∂x[j]`, using linear indexing `x[i]` even if the argument
is higher-dimensional.

This uses forward over reverse, ForwardDiff over Zygote, calling `hessian_dual(f, x)`.
See [`hessian_reverse`](@ref) for an all-Zygote alternative.

See also [`diaghessian`](@ref) to compute only the diagonal part.

# Examples

```jldoctest; setup=:(using Zygote)
julia> hessian(x -> x[1]*x[2], randn(2))
2×2 Matrix{Float64}:
 0.0  1.0
 1.0  0.0

julia> hessian(x -> sum(x.^3), [1 2; 3 4])  # uses linear indexing of x
4×4 Matrix{$Int}:
 6   0   0   0
 0  18   0   0
 0   0  12   0
 0   0   0  24

julia> hessian(sin, pi/2)
-1.0
```
"""
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
    jacobian(f, args...) -> Tuple

For each array `a ∈ args` this returns a matrix with `Ja[k,i] = ∂y[k]/∂a[i]`
where `y = f(args...)` is usually a vector.
Arrays of higher dimension are treated like `vec(a)`, or `vec(y)` for output.

For scalar `x::Number ∈ args`, the result is a vector `Jx[k] = ∂y[k]/∂x`,
while for scalar `y` all results have just one row.

With any other argument type, no result is produced, even if [`gradient`](@ref) would work.

This reverse-mode Jacobian needs to evaluate the pullback once for each element of `y`.
Doing so is usually only efficient when `length(y)` is small compared to `length(a)`,
otherwise forward mode is likely to be better.

See also [`withjacobian`](@ref), [`hessian`](@ref), [`hessian_reverse`](@ref).

# Examples

```jldoctest; setup=:(using Zygote)
julia> jacobian(a -> 100*a[1:3].^2, 1:7)[1]  # first index (rows) is output
3×7 Matrix{$Int}:
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

```
julia> jacobian((a,s) -> a.^length(s), [1,2,3], "str")
([3 0 0; 0 12 0; 0 0 27], nothing)

julia> jacobian((a,t) -> sum(a .* t[1]) + t[2], [1,2,3], (4,5))
([4 4 4], nothing)

julia> gradient((a,t) -> sum(a .* t[1]) + t[2], [1,2,3], (4,5))  # gradient undersands the tuple
([4 4 4], (6, 1))
```
"""
jacobian(f, args...) = withjacobian(f, args...).grad

"""
    withjacobian(f, args...)

Returns both the value `f(args...)` and the [`jacobian`](@ref) as a named tuple.

```jldoctest; setup=:(using Zygote)
julia> withjacobian(cumsum, [1,2,3])
(val = [1, 3, 6], grad = ([1 0 0; 1 1 0; 1 1 1],))
```
"""
function withjacobian(f, args...)
  y, back = pullback(_jvec∘f, args...)
  out = map(args) do x
    T = promote_type(eltype(x), eltype(y))
    dx = x isa AbstractArray ? similar(x, T, length(y), length(x)) :
      x isa Number ? similar(y, T, length(y)) :
      nothing
  end
  delta = _eyelike(y)
  for k in LinearIndices(y)
    grads = back(delta[:,k])
    for (dx, grad) in zip(out, grads)
      dx isa AbstractArray || continue
      _gradcopy!(view(dx,k,:), grad)
    end
  end
  (val=y, grad=out)
end

_jvec(x::AbstractArray) = vec(x)
_jvec(x::Number) = _jvec(vcat(x))
_jvec(x) = throw(ArgumentError("jacobian expected a function which returns an array, or a scalar, got $(typeof(x))"))
_jvec(x::AbstractArray{<:Complex}) = throw(ArgumentError("jacobian does not accept complex output"))

_eyelike(y::Vector) = Matrix{eltype(y)}(I, length(y), length(y))
function _eyelike(y::AbstractVector) # version which works on GPU
  out = fill!(similar(y, length(y), length(y)), 0)
  out[LinearAlgebra.diagind(out)] .= 1
  out
end

_gradcopy!(dst::AbstractArray, src::AbstractArray{<:Number}) = copyto!(dst, src)
_gradcopy!(dst::AbstractArray, src::Number) = copyto!(dst, src)
_gradcopy!(dst::AbstractArray, src::Nothing) = dst .= 0
_gradcopy!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, g isa Number ? g : 0 for g in src) # e.g. Union{Nothing,Float64}

"""
    jacobian(loss, ::Params)

Like [`gradient`](@ref) with implicit parameters, this method takes a zero-argument function
and returns an `IdDict`-like object, now containing the Jacobian for each parameter.

# Examples
```jldoctest; setup=:(using Zygote)
julia> xs = [1 2; 3 4]; ys = [5,7,9];

julia> Jxy = jacobian(() -> ys[1:2] .+ sum(xs.^2), Params([xs, ys]))
Grads(...)

julia> Jxy[ys]
2×3 Matrix{$Int}:
 1  0  0
 0  1  0

julia> Jxy[xs]
2×4 Matrix{$Int}:
 2  6  4  8
 2  6  4  8
```
"""
jacobian(f, pars::Params) = withjacobian(f, pars::Params).grad

function withjacobian(f, pars::Params)
  y, back = pullback(_jvec∘f, pars)
  out = IdDict()
  for p in pars
    T = Base.promote_type(eltype(p), eltype(y))
    J = similar(y, T, length(y), length(p))
    out[p] = J
  end
  delta = _eyelike(y)
  for k in LinearIndices(y)
    grads = back(delta[:,k])
    for p in pars
      out[p] isa AbstractArray || continue
      _gradcopy!(view(out[p],k,:), grads[p])
    end
  end
  (val=y, grad=Grads(out, pars))
end

"""
    diaghessian(f, args...) -> Tuple

Diagonal part of the Hessian. Returns a tuple containing, for each argument `x`,
`h` of the same shape with `h[i] = Hᵢᵢ = ∂²y/∂x[i]∂x[i]`. 
The original evaluation `y = f(args...)` must give a real number `y`.

For one vector argument `x`, this is equivalent to `(diag(hessian(f,x)),)`.
Like [`hessian`](@ref) it uses ForwardDiff over Zygote. 

!!! warning
    For arguments of any type except `Number` & `AbstractArray`, the result is `nothing`.

# Examples
```jldoctest; setup=:(using Zygote, LinearAlgebra)
julia> diaghessian(x -> sum(x.^3), [1 2; 3 4])[1]
2×2 Matrix{$Int}:
  6  12
 18  24

julia> Diagonal(vec(ans)) == hessian(x -> sum(x.^3), [1 2; 3 4])  # full Hessian is diagonal
true

julia> diaghessian((x,y) -> sum(x .* y .* y'), [1 22; 333 4], [0.5, 0.666])  # two array arguments
([0.0 0.0; 0.0 0.0], [2.0, 8.0])

julia> diaghessian(atan, 1, 2)  # two scalar arguments
(-0.16, 0.16)

julia> hessian(xy -> atan(xy[1], xy[2]), [1, 2])  # full Hessian is not diagonal
2×2 Matrix{Float64}:
 -0.16  -0.12
 -0.12   0.16
```
"""
function diaghessian(f, args...)
  ntuple(length(args)) do n
    let x = args[n], valn = Val(n)  # let Val improves speed, sometimes
      if x isa AbstractArray
        forward_diag(x -> gradient(f, _splice(x, args, valn)...)[n], x)[2]
      elseif x isa Number
        ForwardDiff.derivative(x -> gradient(f, _splice(x, args, valn)...)[n], x)
      end
    end
  end
end

_splice(x, args, ::Val{n}) where {n} = ntuple(i -> i==n ? x : args[i], length(args))
