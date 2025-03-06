@testsetup module GradCheckSetup

export ngradient, gradcheck, gradtest, _splitreim, _joinreim, _dropimaggrad
export _randmatunitary, _randmatseries, realdomainrange, cat_test, test_log1pexp

using Test
using Random
using LinearAlgebra
using Statistics
using SparseArrays
using FillArrays
using AbstractFFTs
using FFTW
using Distances
using Zygote
using Zygote: gradient, Buffer
using Base.Broadcast: broadcast_shape
using Distributed: pmap, CachingPool, workers

import FiniteDifferences
import LogExpFunctions

Random.seed!(0)

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

function gradcheck(f, xs...; rtol = 1e-5, atol = 1e-5)
  grad_zygote = gradient(f, xs...)
  grad_finite_difference = ngradient(f, xs...)
  return all(isapprox.(grad_zygote, grad_finite_difference; rtol = rtol, atol = atol))
end

gradtest(f, xs::AbstractArray...; kwargs...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...; kwargs...)
gradtest(f, dims...; kwargs...) = gradtest(f, rand.(Float64, dims)...; kwargs...)

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

_randmatunitary(rng, T, n) = qr(randn(rng, T, n, n)).Q
function _randvectorin(rng, n, r)
  l, u = r
  isinf(l) && isinf(u) && return randn(rng, n)
  isinf(l) && return rand(rng, n) .+ (u - 1)
  isinf(u) && return rand(rng, n) .+ l
  return rand(rng, n) .* (u - l) .+ l
end

realdomainrange(::Any) = (Inf, Inf)
realdomainrange(::Union{typeof.((acos,asin,atanh))...}) = (-1, 1)
realdomainrange(::typeof(acosh)) = (1, Inf)
realdomainrange(::Union{typeof.((log,sqrt,^))...}) = (0, Inf)

function _randmatseries(rng, f, T, n, domain::Type{Real})
  U = _randmatunitary(rng, T, n)
  λ = _randvectorin(rng, n, realdomainrange(f))
  return U * Diagonal(λ) * U'
end

function _randmatseries(rng, f, T, n, domain::Type{Complex})
  U = _randmatunitary(rng, T, n)
  r = realdomainrange(f)
  r == (Inf, Inf) && return nothing
  λ = _randvectorin(rng, n, r)
  λ[end] -= 2
  return U * Diagonal(λ) * U'
end

_randmatseries(rng, ::typeof(atanh), T, n, domain::Type{Complex}) = nothing

function cat_test(f, A::Union{AbstractVector, AbstractMatrix}...)
  @test gradtest(f, A...)
  Z, back = Zygote.pullback(f, A...)
  Ā = back(randn(size(Z)))
  @test all(map((a, ā)->ā isa typeof(a), A, Ā))
end

function test_log1pexp(T, xs)
  y = T(4.3)
  for x in xs
    @test gradcheck(x->y * LogExpFunctions.log1pexp(x[1]), [x])
  end
end

end
