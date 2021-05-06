
using LinearAlgebra: Diagonal, UpperTriangular, LowerTriangular
using LinearAlgebra: AdjointAbsVec, TransposeAbsVec, AdjOrTransAbsVec

import ZygoteRules: clamptype

# Booleans

clamptype(::Type{Bool}, dx::Number) = nothing
clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray) = nothing

# Real numbers

clamptype(::Type{<:Real}, dx::Complex) = real(dx)
clamptype(::Type{<:AbstractArray{<:Real}}, dx::AbstractArray{<:Complex}) = real(dx)

_elmap(::Type, ::Type) = identity  # for fusing fusing broadcasts below
_elmap(::Type{T}, ::Type{S}) where {T<:Real, S<:Complex} = real
_maybecast(proj, dx) = proj.(dx)
_maybecast(::typeof(identity), dx) = dx

# LinearAlgebra -- matrix projections with some zeros

for Wrap in [:Diagonal, :UpperTriangular, :LowerTriangular]
  @eval begin
    clamptype(::Type{<:$Wrap{T}}, dx::AbstractMatrix{S}) where {T,S} = 
      _maybewrap($Wrap, _elmap(T,S), dx)

    _maybewrap(::Type{$Wrap}, ::typeof(identity), dx::$Wrap) = dx  # avoids Diagonal(Diagonal(..., and @debug
  end
end

_maybewrap(Wrap, proj, dx) = (@debug "broadcasting $proj & restoring $Wrap" typeof(dx); Wrap(proj.(dx)))
_maybewrap(Wrap, ::typeof(identity), dx) = (@debug "restoring $Wrap" typeof(dx); Wrap(dx))

# LinearAlgebra -- full matrix projections

clamptype(::Type{<:Hermitian{T}}, dx::AbstractMatrix{S}) where {T,S} = hermitian!!(_elmap(T,S), dx)
clamptype(::Type{<:Symmetric{T}}, dx::AbstractMatrix{S}) where {T,S} = symmetric!!(_elmap(T,S), dx)

"""
    hermitian!!(f, dx) == Hermitian(@.f(dx + dx')/2)

Used for projecting gradients. Mutates when `dx::Array{<:AbstractFloat}`, to save time.
"""
hermitian!!(::typeof(identity), dx::Hermitian) = dx
# hermitian!!(::typeof(identity), dx) = ishermitian(dx) ? Hermitian(dx) : Hermitian(_twofold(Base.adjoint, identity, dx))
hermitian!!(proj, dx) = Hermitian(_twofold(transpose, proj, dx))

symmetric!!(::typeof(identity), dx::Symmetric) = dx
# symmetric!!(::typeof(identity), dx) = issymmetric(dx) ? Symmetric(dx) : Symmetric(_twofold(transpose, identity, dx))
symmetric!!(proj, dx) = Symmetric(_twofold(transpose, proj, dx))

_twofold(trans, proj, dx) = proj.(dx .+ trans(dx)) ./ 2
function _twofold(trans, proj, dx::Array{<:AbstractFloat})
  @inbounds for i in axes(dx,1)
    for j in i+1:lastindex(dx,2)
      dx[i,j] = proj((dx[i,j] + trans(dx[j,i])) / 2)
    end
  end
  dx
end

# LinearAlgebra -- row vectors

# clamptype(::Type{<:AdjointAbsVec{T}}, dx::AbstractMatrix{S}) where {T,S} = 
#     _mayberow(Base.adjoint, _elmap(T,S), dx)
# clamptype(::Type{<:TransposeAbsVec{T}}, dx::AbstractMatrix{S}) where {T,S} = 
#     _mayberow(transpose, _elmap(T,S), dx)

# _mayberow(_, ::typeof(identity), dx::AdjOrTransAbsVec) = dx
# _mayberow(trans, proj, dx) = begin
#   v = _maybecast(proj, vec(dx))
#   isreal(v) ? trans(v) : transpose(v)  # making a Transpose is a smaller sin than conj.(v)
# end
