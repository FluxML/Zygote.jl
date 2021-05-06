
using LinearAlgebra: Diagonal, UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular
using LinearAlgebra: AdjointAbsVec, TransposeAbsVec, AdjOrTransAbsVec

import ZygoteRules: clamptype
# This sees a tuple of argument types, and can modify the resulting tuple of tangents

clamptype(Ts::Tuple{}, dxs::Tuple{}) = ()
clamptype(Ts::Tuple, dxs::Tuple) =
  first(Ts) === GlobalRef ? clamptype(Base.tail(Ts), dxs) :
  (clamptype(first(Ts), first(dxs)), clamptype(Base.tail(Ts), Base.tail(dxs))...)

clamptype(Ts::Tuple{}, dxs::Tuple) = (@error "mismatch!" dxs; dxs)
clamptype(Ts::Tuple, dxs::Tuple{}) = (@error "mismatch!" Ts; ())

# Bool, Real, Complex

# clamptype(::Type{Bool}, dx) = nothing
# clamptype(::Type{Bool}, dx::Complex) = nothing  # ambiguity
# clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray) = nothing
# clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray) = (@info "bool array" summary(dx); nothing)

# clamptype(::Type{Bool}, dx) = (@info "bool, dropping" typeof(dx); nothing)
# clamptype(::Type{Bool}, dx::Complex) = (@info "bool, dropping" typeof(dx); nothing)
# clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray) = (@info "bool array, disabled" summary(dx); dx)

clamptype(::Type{<:Real}, dx::Complex) = real(dx)
clamptype(::Type{<:AbstractArray{<:Real}}, dx::AbstractArray) = real(dx)

# LinearAlgebra's matrix types

for Wrap in [:Diagonal, :UpperTriangular, :UnitUpperTriangular, :LowerTriangular, :UnitLowerTriangular]
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

clamptype(::Type{<:AdjOrTransAbsVec{T,PT}}, dx::AdjOrTransAbsVec) where {T,PT} = 
  clamptype(PT, dx)
clamptype(::Type{<:AdjOrTransAbsVec{T,PT}}, dx::AbstractMatrix) where {T,PT} = 
  clamptype(PT, transpose(vec(dx))) # sometimes wrong wrapper but avoids conjugation


# clamptype(::Type{<:LinearAlgebra.Adjoint{T,PT}}, dx::AbstractMatrix) where {T<:Real,PT} = 
#   clamptype(PT, LinearAlgebra.adjoint(vec(dx)))
# clamptype(::Type{<:LinearAlgebra.Adjoint{T,PT}}, dx::AbstractMatrix) where {T,PT} = 
#   clamptype(PT, transpose(vec(dx))) # wrong wrapper but avoids conjugation

# for (trans, Wrap) in [(transpose, :TransposeAbsVec), (Base.adjoint, :AdjointAbsVec)]
#   @eval begin
#     clamptype(::Type{<:$Wrap{T,PT}}, dx::$Wrap) where {T,PT} = 
#       clamptype(PT, dx)
#     clamptype(::Type{<:$Wrap{T,PT}}, dx::AbstractMatrix) where {T,PT} = 
#       clamptype(PT, $trans(vec(dx)))
#   end
# end

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
