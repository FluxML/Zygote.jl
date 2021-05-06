
using LinearAlgebra: Diagonal, UpperTriangular, LowerTriangular
using LinearAlgebra: AdjointAbsVec, TransposeAbsVec, AdjOrTransAbsVec

import ZygoteRules: clamptype

# Bool, Real, Complex

clamptype(::Type{Bool}, dx) = nothing
clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray) = nothing

clamptype(::Type{<:Real}, dx::Complex) = real(dx)
clamptype(::Type{<:AbstractArray{<:Real}}, dx::AbstractArray{<:Complex}) = real(dx)

clamptype(::Type{Bool}, dx::Complex) = nothing  # ambiguity
clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray{<:Complex}) = nothing

# LinearAlgebra's matrix types

for Wrap in [:Diagonal, :UpperTriangular, :LowerTriangular]
  @eval begin
    clamptype(::Type{<:$Wrap{T,PT}}, dx::$Wrap) where {T,PT} = 
      clamptype(PT, dx)
    clamptype(::Type{<:$Wrap{T,PT}}, dx::AbstractMatrix) where {T,PT} = 
      clamptype(PT, $Wrap(dx))
  end
end

for (trans, Wrap) in [(transpose, :Symmetric), (Base.adjoint, :Hermitian)]
  @eval begin
    clamptype(::Type{<:$Wrap{T,PT}}, dx::$Wrap) where {T,PT} = 
      clamptype(PT, dx)
    clamptype(::Type{<:$Wrap{T,PT}}, dx::AbstractMatrix) where {T,PT} = 
      clamptype(PT, $Wrap(_twofold($trans, dx)))
  end
end

_twofold(trans, dx) = (dx .+ trans(dx)) ./ 2
function _twofold(trans, dx::Array{<:AbstractFloat})
  @inbounds for i in axes(dx,1)
    for j in i+1:lastindex(dx,2)
      dx[i,j] = (dx[i,j] + trans(dx[j,i])) / 2
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
