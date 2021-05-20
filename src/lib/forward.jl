using ForwardDiff
using ForwardDiff: Dual

seed(x::Real, ::Val) = Dual(x, true)

function seed(x, ::Val{N}, offset = 0) where N
  map(x, reshape(1:length(x), size(x))) do x, i
    Dual(x, ntuple(j -> j+offset == i, Val(N)))
  end
end
function seed!(xplus, x, ::Val{N}, offset) where N
  @assert size(x) == size(xplus)
  map!(xplus, x, reshape(1:length(x), size(x))) do x, i
    Dual(x, ntuple(j -> j+offset == i, Val(N)))
  end
end

extract(x::ForwardDiff.Dual) = x.value, [x.partials...]

function extract(xs::AbstractArray{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
  J = similar(xs, V, N, length(xs))
  for i = 1:length(xs), j = 1:N
    J[j, i] = xs[i].partials.values[j]
  end
  return map(x -> x.value, xs), J
end

function forward_jacobian(f, x, ::Val{N}) where N
  y, _J = extract(f(seed(x, Val(N))))
  J = similar(_J, length(x), length(y))
  J[1:N,:] = _J
  offset = 0
  while offset + N < length(x)
    offset += N
    _, _J = extract(f(seed(x, Val(N), offset)))
    range = (1+offset):min(N+offset,length(x))
    J[range,:] = @view _J[range.-offset,:]
  end
  return y, J
end

function forward_jacobian(f, x)
  if length(x) < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    forward_jacobian(f, x, Val(length(x)))
  else
    forward_jacobian(f, x, Val(ForwardDiff.DEFAULT_CHUNK_THRESHOLD))
  end
end

vec_scalar(x) = vec(x)
vec_scalar(x::Real) = [x]
reshape_scalar(x, y) = reshape(y, size(x))
reshape_scalar(x::Real, y) = y[]

# very similar functions needed for diaghessian:

function extract_diag!(out, offset, xs::AbstractArray{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
  for j in 1:min(N, length(xs)-offset)
    out[offset+j] = xs[offset+j].partials.values[j]
  end
end

function forward_diag(f, x::AbstractArray{T}, ::Val{N}) where {N,T}
  xplus = seed(x, Val(N))
  fx = f(xplus)
  out = similar(x, ForwardDiff.valtype(eltype(fx)))
  extract_diag!(out, 0, fx)
  offset = 0
  while offset + N < length(x)
    offset += N
    extract_diag!(out, offset, f(seed!(xplus, x, Val(N), offset)))
  end
  return map(y -> y.value, fx), out
end

function forward_diag(f, x::AbstractArray)
  if length(x) < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    forward_diag(f, x, Val(length(x)))
  else
    forward_diag(f, x, Val(ForwardDiff.DEFAULT_CHUNK_THRESHOLD))
  end
end

"""
    forwarddiff(f, x) -> f(x)

Runs `f(x)` as usual, but instructs Zygote to differentiate `f` using forward
mode, rather than the usual reverse mode.

Forward mode takes time linear in `length(x)` but only has constant memory
overhead, and is very efficient for scalars, so in some cases this can be a
useful optimisation.

```julia
julia> function pow(x, n)
         r = one(x)
         for i = 1:n
           r *= x
         end
         return r
       end
pow (generic function with 1 method)

julia> gradient(5) do x
         forwarddiff(x) do x
           pow(x, 2)
         end
       end
(10,)
```

Note that the function `f` will *drop gradients* for any closed-over values.

```julia
julia> gradient(2, 3) do a, b
         forwarddiff(a) do a
           a*b
         end
       end
(3, nothing)
```

This can be rewritten by explicitly passing through `b`, i.e.

```julia
gradient(2, 3) do a, b
  forwarddiff([a, b]) do (a, b)
    a*b
  end
end
```
"""
forwarddiff(f, x) = f(x)

@adjoint function forwarddiff(f, x)
  y, J = forward_jacobian(f, x)
  return y, ȳ -> (nothing, reshape_scalar(x, J*vec_scalar(ȳ)))
end

# Use this to allow second derivatives -- this is forward-over-forward, 
# see  https://github.com/FluxML/Zygote.jl/issues/769  for a forward-over-reverse proposal
@adjoint ForwardDiff.gradient(f, x) = pullback(forwarddiff, x -> ForwardDiff.gradient(f, x), x)
@adjoint ForwardDiff.jacobian(f, x) = pullback(forwarddiff, x -> ForwardDiff.jacobian(f, x), x)

@adjoint ForwardDiff.derivative(f, x) = pullback(forwarddiff, x -> ForwardDiff.derivative(f, x), x)
@adjoint ForwardDiff.hessian(f, x) = pullback(forwarddiff, x -> ForwardDiff.hessian(f, x), x)
