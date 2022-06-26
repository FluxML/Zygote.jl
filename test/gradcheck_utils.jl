using Zygote, Test, Random, LinearAlgebra, Statistics, SparseArrays, FillArrays,
    AbstractFFTs, FFTW, Distances
using Zygote: gradient
using Base.Broadcast: broadcast_shape
using Distributed: pmap, CachingPool, workers
import FiniteDifferences

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

function gradcheck(f, xs...)
  grad_zygote = gradient(f, xs...)
  grad_finite_difference = ngradient(f, xs...)
  return all(isapprox.(grad_zygote, grad_finite_difference; rtol = 1e-5, atol = 1e-5))
end

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

# utilities for using gradcheck with complex matrices
_splitreim(A) = (real(A),)
_splitreim(A::AbstractArray{<:Complex}) = reim(A)

_joinreim(A, B) = complex.(A, B)
_joinreim(A) = A

function _dropimaggrad(A)
  back(Δ) = real(Δ)
  back(Δ::Nothing) = nothing
  return Zygote.hook(back, A)
end
