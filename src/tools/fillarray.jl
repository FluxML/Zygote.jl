struct FillArray{T,N} <: AbstractArray{T,N}
  value::T
  size::NTuple{N,Int}
end

Base.size(xs::FillArray) = xs.size

Base.getindex(xs::FillArray, ::Int...) = xs.value
