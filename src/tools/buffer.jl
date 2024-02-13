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

function Base.deleteat!(b::Buffer, i)
  b.freeze && error("Buffer is frozen")
  deleteat!(b.data, i)
  return b
end

@forward Buffer.data  Base.eltype, Base.length, Base.ndims, Base.size, Base.axes,
                      Base.eachindex, Base.stride, Base.strides, Base.findfirst,
                      Base.keys

Base.IteratorSize(::Type{<:Buffer{<:Any, A}}) where {A} = Base.IteratorSize(A)

# Buffer iteration mirrors iteration for AbstractArray
function Base.iterate(b::Buffer, state=(eachindex(b),))
  y = iterate(state...)
  y === nothing && return nothing
  b[y[1]], (state[1], tail(y)...)
end

Base.BroadcastStyle(::Type{Buffer{T,A}}) where {T,A} = Base.BroadcastStyle(A)
