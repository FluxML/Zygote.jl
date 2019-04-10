using TaylorSeries
import Base: *, +

@adjoint Taylor1(order::Integer) = Taylor1(order), _ -> nothing

struct TaylorAdjoint{T<:Number}
  adjoint::Vector{T}
end

a::TaylorAdjoint * b::Number = TaylorAdjoint([a*b for a in a.adjoint])

a::TaylorAdjoint + b::TaylorAdjoint = TaylorAdjoint(a.adjoint .+ b.adjoint)

derive(t::Taylor1, i::Integer = 1) = constant_term(TaylorSeries.derivative(t, i))
derive(x::Number, i::Integer = 1) = i == 0 ? x : zero(x)

@adjoint function derive(t::Taylor1, i::Integer = 1)
  derive(t, i), ā -> (TaylorAdjoint([i==j ? ā : zero(ā) for j = 0:t.order]), nothing)
end

resolve(x) = x
resolve(t̄::TaylorAdjoint) = sum(derive(t̄.adjoint[i], i-1) for i = 1:length(t̄.adjoint))
resolve(x::AbstractArray) = resolve.(x)

@adjoint convert(T::Type{<:Taylor1}, x::AbstractSeries) =
  convert(T, x), ȳ -> (nothing, ȳ,)

@adjoint convert(T::Type{<:Taylor1}, x::Number) =
  convert(T, x), ȳ -> (nothing, resolve(ȳ))

taylor(x) = x

@adjoint taylor(x) = x, x̄ -> (resolve(x̄),)
