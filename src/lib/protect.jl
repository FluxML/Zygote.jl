
"""
    NoWrite(Δ::AbstractArray)

This is a trivial wrapper, without `setindex!`, to prevent mutation of gradients
when the same array may be in use elsewhere. It should be applied using `_protect`
in rules like `@adjoint +(A,B) = A+B, Δ -> (_protect(Δ), _protect(Δ))`.
(This will be handled automatically for rules defined via ChainRules.jl.)
"""
struct NoWrite{T,N,P} <: AbstractArray{T,N}
  data::P
  NoWrite(x::P) where {P <: AbstractArray{T,N}} where {T,N} = new{T,N,P}(x)
end

Base.parent(x::NoWrite) = x.data

Base.@propagate_inbounds Base.getindex(A::NoWrite, i...) = getindex(A.data, i...)

for f in (:size, :axes, :length, :similar, :copy, :IndexStyle, :strides, :pointer, :Tuple, :iterate)
  @eval Base.$f(x::NoWrite) = Base.$f(x.data)
end

Base.showarg(io::IO, x::NoWrite, top) = begin print(io, "NoWrite("); Base.showarg(io, x.data, false); print(io, ")") end

_unprotect(A::NoWrite) = parent(A) # for use on the RHS of rules, e.g. to avoid generic matmul
_unprotect(A) = A

_protect(A::DenseArray) = NoWrite(A)
_protect(A::NoWrite) = A # never need to wrap twice
_protect(A::AbstractArray) = _maybewrite(A) ? NoWrite(A) : A  # protect anything which could be upwrapped to be writable?
_protect(A) = A

_maybewrite(A) = false
_maybewrite(A::DenseArray) = true
_maybewrite(A::AbstractArray) = A===parent(A) ? false : _maybewrite(parent(A))

##### For Params & Grads, don't accumulate NoWrite objects

Base.setindex!(dict::IdDict, dx::NoWrite, x) = dict[x] = copy(dx.data)

##### For ChainRules rules, unwrap & re-wrap automatically:

_pointer(A::Array) = pointer(A)  # pointer survives reshape, objectid does not
_pointer(A::AbstractArray) = A===parent(A) ? NaN : _pointer(parent(A)) # not strictly necc
_pointer(A) = nothing  # compares == self

@inline function (s::ZBack)(dy::NoWrite)
  ptr_orig = _pointer(dy.data)
  @debug "unwrapping for chainrules" summary(dy.data) ptr s.back
  dxs = wrap_chainrules_output(s.back(wrap_chainrules_input(dy.data)))
  dxs === nothing && return
  ptrs = map(_pointer, dxs)
  map(dxs) do dx
    ptr = _pointer(dx)
    if ptr === nothing
      dx
    elseif ptr == ptr_orig
      @debug "re-wrapping for chainrules" summary(dy.data) ptr
      _protect(dx)
    elseif count(isequal(ptr), ptrs) > 1
      @debug "wrapping for chainrules" summary(dy.data) ptr
      _protect(dx)
    else
      dx
    end
  end
end


###### For @adjoint rules: 

Broadcast.broadcastable(A::NoWrite) = A.data  # always unwrap on the RHS of broadcasting

Base.mapreduce(f, op, A::NoWrite; kw...) = mapreduce(f, op, A.data; kw...)  # always unwrap within sum, etc.

# Try to keep NoWrite outside, to maximise chances of sucessful unwrapping:
Base._reshape(x::NoWrite,  dims::Tuple{Vararg{Int}}) = NoWrite(reshape(x, dims))
for f in (:transpose, :adjoint, :Transpose, :Adjoint, :Diagonal)
  @eval LinearAlgebra.$f(x::NoWrite) = NoWrite(LinearAlgebra.$f(x.data))
end

using AbstractFFTs  # many rules, easier to overload here:

for f in (:fft, :bfft, :ifft, :rfft, :irfft, :brfft, :fftshift, :ifftshift)
  @eval AbstractFFTs.$f(x::NoWrite, dims...) = AbstractFFTs.$f(_unprotect(x), dims...)
end

# LinearAlgebra.:\(A::AbstractMatrix, B::NoWriteVecOrMat)

# The dispatch for * is very messy, better just to unwrap by hand. For debugging:

NoWriteVector{T} = NoWrite{T,1}
NoWriteMatrix{T} = NoWrite{T,2}
NoWriteVecOrMat{T} = Union{NoWriteVector{T}, NoWriteMatrix{T}}

LinearAlgebra.generic_matvecmul!(C::AbstractVector, tA, A::NoWriteVecOrMat, B::AbstractVector, _add::LinearAlgebra.MulAddMul) = _mulv(C, tA, A, B, _add)
LinearAlgebra.generic_matvecmul!(C::AbstractVector, tA, A::AbstractVecOrMat, B::NoWriteVector, _add::LinearAlgebra.MulAddMul) = _mulv(C, tA, A, B, _add)
LinearAlgebra.generic_matvecmul!(C::AbstractVector, tA, A::NoWriteVecOrMat, B::NoWriteVector, _add::LinearAlgebra.MulAddMul) = _mulv(C, tA, A, B, _add)

function _mulv(C, tA, A, B, _add)
  @debug "generic matrix-vector due to NoWrite" summary(A) summary(B)
  invoke(LinearAlgebra.generic_matvecmul!, Tuple{AbstractVector, Any, AbstractVecOrMat, AbstractVector, LinearAlgebra.MulAddMul}, C, tA, A, B, _add)
end

LinearAlgebra.generic_matmatmul!(C::AbstractMatrix, tA, A::NoWriteMatrix, B::AbstractMatrix, _add::LinearAlgebra.MulAddMul) = _mulm(C, tA, A, B, _add)
LinearAlgebra.generic_matmatmul!(C::AbstractMatrix, tA, A::AbstractMatrix, B::NoWriteMatrix, _add::LinearAlgebra.MulAddMul) = _mulm(C, tA, A, B, _add)
LinearAlgebra.generic_matmatmul!(C::AbstractMatrix, tA, A::NoWriteMatrix, B::NoWriteMatrix, _add::LinearAlgebra.MulAddMul) = _mulm(C, tA, A, B, _add)

function _mulm(C, tA, A, B, _add)
  @debug "generic matrix-matrix multiplication due to NoWrite" summary(A) summary(B)
  invoke(LinearAlgebra.generic_matmatmul!, Tuple{AbstractMatrix, Any, AbstractMatrix, AbstractMatrix, LinearAlgebra.MulAddMul}, C, tA, A, B, _add)
end


