
accum(x::DenseArray, ys::AbstractArray...) = broadcast!(+, x, x, ys...)
accum(x::DenseArray, y::DenseArray, zs::AbstractArray...) = broadcast!(+, x, x, y, zs...)
accum(x::AbstractArray, y::DenseArray, zs::AbstractArray...) = broadcast!(+, y, x, y, zs...)

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


##### For ChainRules rules, unwrap & re-wrap automatically:

_pointer(A::Array) = pointer(A)  # pointer survives reshape, objectid does not
_pointer(A::AbstractArray) = A===parent(A) ? NaN : _pointer(parent(A)) # not strictly necc
_pointer(A) = nothing  # compares == self

@inline function (s::ZBack)(dy::NoWrite)
  ptr = _pointer(dy.data)
  @debug "unwrapping for chainrules" summary(dy.data) ptr s.back
  dxs = wrap_chainrules_output(s.back(wrap_chainrules_input(dy.data)))
  map(dxs) do dx
    if _pointer(dx) == ptr # if the rule retuns dy, a reshape etc. of it, this must still be protected
      @debug "re-wrapping for chainrules" summary(dy.data) ptr
      _protect(dx)
    else
      dx
    end
  end
end

@inline function (s::ZBack)(dy::DenseArray)
  ptr = _pointer(dy)
  dxs = wrap_chainrules_output(s.back(wrap_chainrules_input(dy)))
  map(dxs) do dx
    if _pointer(dx) == ptr
      @debug "wrapping for chainrules" summary(dy) ptr
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

AbstractFFTs.fft(x::NoWrite, dims...) = AbstractFFTs.fft(_unprotect(x), dims...)
AbstractFFTs.bfft(x::NoWrite, dims...) = AbstractFFTs.bfft(_unprotect(x), dims...)
AbstractFFTs.ifft(x::NoWrite, dims...) = AbstractFFTs.ifft(_unprotect(x), dims...)
AbstractFFTs.rfft(x::NoWrite, dims...) = AbstractFFTs.rfft(_unprotect(x), dims...)
AbstractFFTs.irfft(x::NoWrite, d, dims...) = AbstractFFTs.irfft(_unprotect(x), d, dims...)
AbstractFFTs.brfft(x::NoWrite, d, dims...) = AbstractFFTs.brfft(_unprotect(x), d, dims...)
AbstractFFTs.fftshift(x::NoWrite) = AbstractFFTs.fftshift(_unprotect(x))
AbstractFFTs.ifftshift(x::NoWrite) = AbstractFFTs.ifftshift(_unprotect(x))

