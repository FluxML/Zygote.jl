using LinearAlgebra

zerolike(x::AbstractArray) = zerolike.(x)

@tangent function (A::Type{<:Array})(::UndefInitializer, sz::Integer...)
  x = A(UndefInitializer(), sz...)
  x, (_...) -> zerolike(x)
end

@tangent length(A::AbstractArray) = length(A), _ -> 0
@tangent size(A::AbstractArray, i::Integer) = size(A, i), (_, _) -> 0
@tangent size(A::AbstractArray) = size(A), _ -> zerolike(size(A))

@tangent Base.vect(xs...) = Base.vect(xs...), Base.vect

@tangent fill(x, dims::Tuple) = fill(x, dims), (ẋ, _) -> fill(ẋ, dims)

@tangent first(x) = first(x), first

@tangent setindex!(x::AbstractArray, v, inds...) =
  setindex!(x, v, inds...), (ẋ, v̇, _...) -> setindex!(ẋ, v̇, inds...)

@tangent mul!(C, A, B) = mul!(C, A, B), (Ċ, Ȧ, Ḃ) -> Ċ .= Ȧ*B .+ A*Ḃ

@tangent A::AbstractArray * B::AbstractArray = A*B, (Ȧ, Ḃ) -> Ȧ*B .+ A*Ḃ

@tangent sum(x; dims = :) = sum(x; dims = dims), ẋ -> sum(ẋ, dims = dims)
