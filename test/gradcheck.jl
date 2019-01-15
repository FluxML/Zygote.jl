using Zygote, NNlib, Test, Random, LinearAlgebra
using Zygote: gradient
using NNlib: conv
import Random

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

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

Random.seed!(0)

@test gradient(//, 2, 3) === (1//3, -2//9)

@test gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)
@test gradtest((x, W, b) -> logσ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> logσ.(W*x .+ b), (5,3), (2,5), 2)

@test gradtest((w, x) -> w'*x, randn(10, 2), randn(10))
@test gradtest((w, x) -> transpose(w)*x, randn(5,5), randn(5,5))

@test gradtest(x -> sum(x, dims = (2, 3)), (3,4,5))
@test gradtest(x -> prod(x, dims = (2, 3)), (3,4,5))
@test gradtest(x -> prod(x), (3,4,5))

@test gradtest(x -> softmax(x).*(1:3), 3)
@test gradtest(x -> softmax(x).*(1:3), (3,5))
@test gradtest(x -> logsoftmax(x).*(1:3), 3)
@test gradtest(x -> logsoftmax(x).*(1:3), (3,5))

@test gradtest(conv, rand(10, 3, 2), randn(Float64,2, 3, 2))
@test gradtest(conv, rand(10, 10, 3, 2), randn(Float64,2, 2, 3, 2))
@test gradtest(conv, rand(10, 10, 10, 3, 2), randn(Float64,2, 2, 2, 3, 2))

@test gradtest(x -> maxpool(x, (2,2)), rand(10, 10, 3, 2))
@test gradtest(x -> maxpool(x, (2,2,2)), rand(10, 10, 10, 3, 2))

@test gradtest(x -> meanpool(x, (2,2)), rand(10, 10, 3, 2))
@test gradtest(x -> meanpool(x, (2,2,2)), rand(5, 5, 5, 3, 2))

@test gradtest(x -> permutedims(x, [3,1,2]), rand(4,5,6))

@test gradtest(x -> repeat(x; inner=2), rand(5))
@test gradtest(x -> repeat(x; inner=2, outer=3), rand(5))
@test gradtest(x -> repeat(x; inner=(2,2,1), outer=(1,1,3)), rand(5,4,3))

@test gradtest(kron, rand(5), rand(3))
@test gradtest(kron, rand(5), rand(3), rand(8))
@test gradtest(kron, rand(5,1), rand(3,1))
@test gradtest(kron, rand(5,1), rand(3,1), rand(8,1))
@test gradtest(kron, rand(5,2), rand(3,2), rand(8,2))

@testset "maximum" begin
  @test gradtest(maximum, rand(2, 3))

  @test gradtest(x -> maximum(x, dims=1), rand(2, 3))
  @test gradtest(x -> maximum(x, dims=2), rand(2, 3))
  @test gradtest(x -> maximum(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> maximum(x, dims=[1, 2]), rand(2, 3, 4))
end

@testset "minimum" begin
  @test gradtest(minimum, rand(2, 3))

  @test gradtest(x -> minimum(x, dims=1), rand(2, 3))
  @test gradtest(x -> minimum(x, dims=2), rand(2, 3))
  @test gradtest(x -> minimum(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> minimum(x, dims=[1, 2]), rand(2, 3, 4))
end

@testset "backsolve" begin
  rng, P, Q = MersenneTwister(123456), 10, 9
  X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)

  # \
  @test gradtest(X -> X \ Y, X)
  @test gradtest(Y -> X \ Y, Y)
  @test gradtest(X -> X \ y, X)
  @test gradtest(y -> X \ y, y)

  # /
  @test gradtest(X -> Y' / X, X)
  @test gradtest(Y -> Y' / X, Y)
  @test gradtest(X -> y' / X, X)
  @test gradtest(y -> y' / X, y)
end

@testset "Symmetric" begin
  rng, P = MersenneTwister(123456), 7
  A = randn(rng, P, P)
  @test gradtest(Symmetric, A)
end

@testset "diag" begin
  rng, P = MersenneTwister(123456), 10
  A = randn(rng, P, P)
  @test gradtest(diag, A)
end

@testset "dense + UniformScaling" begin
  rng = MersenneTwister(123456)
  A, λ = randn(rng, 10, 10), randn(rng)
  @test gradtest(A->A + 5I, A)
  @test gradtest(λ->A + λ[1] * I, [λ])
end

@testset "cholesky" begin
  rng, N = MersenneTwister(123456), 5
  A = randn(rng, N, N)
  @test gradtest(A->logdet(cholesky(A' * A + 1e-6I)), A)
end

using Distances

@testset "distances" begin
let
  rng, P, Q, D = MersenneTwister(123456), 10, 9, 8

  # Check sqeuclidean.
  let
    x, y = randn(rng, D), randn(rng, D)
    gradtest(x->sqeuclidean(x, y), x)
    gradtest(y->sqeuclidean(x, y), y)
  end

  # Check binary colwise.
  let
    X, Y = randn(rng, D, P), randn(rng, D, P)
    gradtest(X->colwise(SqEuclidean(), X, Y), X)
    gradtest(Y->colwise(SqEuclidean(), X, Y), Y)
  end

  # Check binary pairwise.
  let
    X, Y = randn(rng, D, P), randn(rng, D, Q)
    gradtest(X->pairwise(SqEuclidean(), X, Y), X)
    gradtest(Y->pairwise(SqEuclidean(), X, Y), Y)
  end

  # Check unary pairwise.
  gradtest(X->pairwise(SqEuclidean(), X), randn(rng, D, P))
end
end
