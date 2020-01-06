"""
    Buffer(xs, ...)

`Buffer` is an array-like type which is mutable when taking gradients. You can
construct a `Buffer` with the same syntax as `similar` (e.g. `Buffer(xs, 5)`)
and then use normal indexing. Finally, use `copy` to get back a normal array.

For example:

```julia
julia> function vstack(xs)
           buf = Buffer(xs, length(xs), 5)
           for i = 1:5
             buf[:, i] = xs
           end
           return copy(buf)
         end
vstack (generic function with 1 method)

julia> vstack([1, 2, 3])
3×5 Array{Int64,2}:
 1  1  1  1  1
 2  2  2  2  2
 3  3  3  3  3

julia> gradient(x -> sum(vstack(x)), [1, 2, 3])
([5.0, 5.0, 5.0],)
```

`Buffer` is not an `AbstractArray` and can't be used for linear algebra
operations like matrix multiplication. This prevents it from being captured by
pullbacks.

`copy` is a semantic copy, but does not allocate memory. Instead the `Buffer`
is made immutable after copying.
"""
mutable struct Buffer{T,A<:AbstractArray{T}}
  data::A
  freeze::Bool
end

Buffer(xs::AbstractArray, args...) =
  Buffer(similar(xs, args...), false)

bufferfrom(xs::AbstractArray) = Buffer(xs, false)

Base.getindex(b::Buffer, i...) = b.data[i...]

function Base.setindex!(b::Buffer, v, i...)
  b.freeze && error("Buffer is frozen")
  b.data[i...] = v
end

function Base.copyto!(b::Buffer, data)
  b.freeze && error("Buffer is frozen")
  copyto!(b.data, data)
end

function Base.push!(b::Buffer, data)
  b.freeze && error("Buffer is frozen")
  push!(b.data, data)
end

function Base.copy(b::Buffer)
  b.freeze = true
  return b.data
end

@forward Buffer.data Base.eltype, Base.length, Base.ndims, Base.size, Base.axes, Base.eachindex, Base.stride, Base.strides

grad_mut(b::Buffer) = fill!(similar(b.data, Any), nothing)
grad_mut(b::Buffer{T}) where T<:Number = fill!(similar(b.data, float(T)), 0)

@nograd Buffer

@adjoint function getindex(b::Buffer, i...)
  b[i...], function (Δ)
    grad = grad_mut(__context__, b)
    grad[i...] = accum(grad[i...], Δ)
    return
  end
end

@adjoint! function setindex!(b::Buffer, v, i...)
  setindex!(b, v, i...), function (_)
    grad = grad_mut(__context__, b)
    v̄ = grad[i...]
    zero = eltype(grad) <: Number ? 0 : nothing
    if i isa NTuple{N,Integer} where N
      grad[i...] = zero
    else
      grad[i...] .= zero
    end
    (nothing, v̄, map(_->nothing, i)...)
  end
end

@adjoint! function copyto!(b::Buffer, xs)
  copyto!(b, xs), function (_)
    grad = grad_mut(__context__, b)
    x̄s = copy(grad)
    grad .= eltype(grad) <: Number ? 0 : nothing
    return (nothing, x̄s)
  end
end

@adjoint! function push!(b::Buffer, x)
  push!(b, x), function (y)
    grad = grad_mut(__context__, b)
    return (nothing, pop!(grad))
  end
end

_pullback(cx::AContext, ::typeof(Broadcast.materialize!), b::Buffer, x::AbstractArray) =
  _pullback(cx, copyto!, b, x)

@adjoint function copy(b::Buffer)
  copy(b), function (b̄)
    grad_mut(__context__, b)[:] = b̄
    return
  end
end
