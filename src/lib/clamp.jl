
import ZygoteRules: clamptype
# This sees a tuple of argument types, and can modify the resulting tuple of tangents

clamptype(Ts::Tuple{}, dxs::Tuple{}) = ()
clamptype(Ts::Tuple, dxs::Tuple) =
  first(Ts) === GlobalRef ? clamptype(Base.tail(Ts), dxs) :
  (clamptype(first(Ts), first(dxs)), clamptype(Base.tail(Ts), Base.tail(dxs))...)

clamptype(Ts::Tuple{}, dxs::Tuple) = (@error "mismatch, extra arguments:" dxs; dxs)
clamptype(Ts::Tuple, dxs::Tuple{}) = (@error "mismatch, extra types:" Ts; ())

# Bool, Real, Complex

# clamptype(::Type{Bool}, dx) = nothing
# clamptype(::Type{Bool}, dx::Complex) = nothing  # ambiguity
# clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray) = nothing
# clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray) = (@info "bool array" summary(dx); nothing)

# clamptype(::Type{Bool}, dx) = (@info "bool, dropping" typeof(dx); nothing)
# clamptype(::Type{Bool}, dx::Complex) = (@info "bool, dropping" typeof(dx); nothing)
# clamptype(::Type{<:AbstractArray{Bool}}, dx::AbstractArray) = (@info "bool array, disabled" summary(dx); dx)

clamptype(::Type{<:Real}, dx::Complex) = real(dx)
clamptype(::Type{<:AbstractArray{<:Real}}, dx::AbstractArray{<:Complex}) = real(dx)

using LinearAlgebra: Diagonal, UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular
using LinearAlgebra: AdjointAbsVec, TransposeAbsVec, AdjOrTransAbsVec, AdjOrTrans

# LinearAlgebra's matrix types

for Wrap in [:Diagonal, :UpperTriangular, :LowerTriangular]
  @eval begin
    clamptype(::Type{<:$Wrap{T,PT}}, dx::$Wrap) where {T,PT} = 
      clamptype(PT, dx)
    clamptype(::Type{<:$Wrap{T,PT}}, dx::AbstractMatrix) where {T,PT} = 
      clamptype(PT, $Wrap(dx))

    accum(x::$Wrap, y::$Wrap) = $Wrap(accum.(parent(x), parent(y)))
  end
end

for (trans, Wrap) in [(transpose, :Symmetric), (Base.adjoint, :Hermitian)]
  @eval begin
    clamptype(::Type{<:$Wrap{T,PT}}, dx::$Wrap) where {T,PT} = 
      clamptype(PT, dx)
    clamptype(::Type{<:$Wrap{T,PT}}, dx::AbstractMatrix) where {T,PT} = 
      clamptype(PT, $Wrap(_twofold($trans, dx)))

    accum(x::$Wrap, y::$Wrap) = $Wrap(accum.(parent(x), parent(y)))
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

accum(x::Transpose, y::Transpose) = Transpose(accum.(parent(x), parent(y)))
accum(x::LinearAlgebra.Adjoint, y::LinearAlgebra.Adjoint) = LinearAlgebra.Adjoint(accum.(parent(x), parent(y)))
