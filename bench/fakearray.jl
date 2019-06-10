import Base: *

struct FakeArray{T,N} <: AbstractArray{T,N}
  size::NTuple{N,Int}
end

FakeArray{T}(sz::Vararg{Int,N}) where {T,N} = FakeArray{T,N}(sz)
FakeArray(sz::Vararg{Int,N}) where N = FakeArray{Float64}(sz...)

FakeArray(x::AbstractArray) = FakeArray(size(x)...)

Base.size(x::FakeArray) = x.size
Base.show(io::IO, x::FakeArray) = print(io, typeof(x), "(", size(x), ")")
Base.print_array(io::IO, x::FakeArray) = println(io, "(no data)")

index_size(x, i...) = length.(Base.index_shape(to_indices(x, i)...))
Base.getindex(x::FakeArray, i...) = FakeArray(index_size(x, i...)...)
Base.getindex(x::FakeArray, i::Integer...) = error("scalar indexing not allowed")

Base.similar(x::FakeArray) = FakeArray(size(x)...)
Base.similar(x::FakeArray, sz::Integer...) = FakeArray(sz...)

a::FakeArray * b::FakeArray = FakeArray(size(a, 1), size(b, 2))

Base.adjoint(x::FakeArray) = FakeArray(reverse(size(x))...)
Base.transpose(x::FakeArray) = FakeArray(reverse(size(x))...)

Broadcast.BroadcastStyle(::Type{<:FakeArray}) = Broadcast.ArrayStyle{FakeArray}()
Broadcast.materialize(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{FakeArray}}) =
  FakeArray(size(bc)...)

Base.reshape(x::FakeArray, i::Union{Int,Colon}...) = FakeArray(Base._reshape_uncolon(x, i)...)
Base.reshape(x::FakeArray, i::Tuple{Vararg{Int}}) = FakeArray(i...)

_mapreduce(f, op, x, dims::Colon) = FakeArray{eltype(x)}()
_mapreduce(f, op, x, dims) = FakeArray(length.(Base.reduced_indices(x, dims))...)

function Base.mapreduce(f, op, x::FakeArray; dims = :)
  _mapreduce(f, op, x, dims)
end

using NNlib

NNlib.softmax(x::FakeArray) = x
NNlib.∇softmax(Δ, xs::FakeArray) = xs

using Zygote

Zygote.sensitivity(x::FakeArray{T,0}) where T = x

Zygote.@adjoint getindex(x::FakeArray, i...) =
  x[i...], ȳ -> (x, map(_ -> nothing, i)...)
