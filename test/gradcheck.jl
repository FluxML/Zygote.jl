using Zygote, NNlib, Test, Random, LinearAlgebra, Statistics, FillArrays, FFTW
using Zygote: gradient
using NNlib: conv, ∇conv_data, depthwiseconv
using Base.Broadcast: broadcast_shape

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
@test gradtest(x -> sum(abs2, x), randn(4, 3, 2))
@test gradtest(x -> sum(abs2, x; dims=1), randn(4, 3, 2))
@test gradtest(x -> prod(x, dims = (2, 3)), (3,4,5))
@test gradtest(x -> prod(x), (3,4,5))

@test gradtest(x -> softmax(x).*(1:3), 3)
@test gradtest(x -> softmax(x).*(1:3), (3,5))
@test gradtest(x -> logsoftmax(x).*(1:3), 3)
@test gradtest(x -> logsoftmax(x).*(1:3), (3,5))

@test gradtest(x -> x', rand(5))

@test gradtest(det, (4, 4))
@test gradtest(logdet, map(x -> x*x', (rand(4, 4),))[1])
@test gradtest(x -> logabsdet(x)[1], (4, 4))

@test gradtest(x -> view(x,:,2,:), (3,4,5))
@test gradtest(x -> view(x,1:2,3:4), (3,4))

@testset "conv" begin
  for spatial_rank in (1, 2, 3)
    x = rand(repeat([10], spatial_rank)..., 3, 2)
    w = rand(repeat([3], spatial_rank)..., 3, 3)
    cdims = DenseConvDims(x, w)
    @test gradtest((x, w) -> conv(x, w, cdims), x, w)
    y = conv(x, w, cdims)
    @test gradtest((y, w) -> ∇conv_data(y, w, cdims), y, w)
    dcdims = DepthwiseConvDims(x, w)
    @test gradtest((x, w) -> depthwiseconv(x, w, dcdims), x, w)
  end
end

@testset "pooling" begin
  for spatial_rank in (1, 2)
    x = rand(repeat([10], spatial_rank)..., 3, 2)
    pdims = PoolDims(x, 2)
    @test gradtest(x -> maxpool(x, pdims), x)
    @test gradtest(x -> meanpool(x, pdims), x)
  end
end

@test gradtest(x -> permutedims(x), rand(2))
@test gradtest(x -> permutedims(x), rand(2,3))
@test gradtest(x -> permutedims(x, [3,1,2]), rand(4,5,6))
@test gradtest(x -> PermutedDimsArray(x, (3,1,2)), rand(4,5,6))
let
  y, back = Zygote.forward(permutedims, randn(3))
  @test first(back(randn(1, 3))) isa Vector
end

@test gradtest(x -> repeat(x; inner=2), rand(5))
@test gradtest(x -> repeat(x; inner=2, outer=3), rand(5))
@test gradtest(x -> repeat(x; inner=(2,2,1), outer=(1,1,3)), rand(5,4,3))

@test gradtest(tr, rand(4, 4))

@testset "fill" begin
  rng, N, M, P = MersenneTwister(123456), 11, 6, 5
  @test gradtest(x->fill(first(x), N), randn(rng, 1))
  @test gradtest(x->fill(first(x), N, M), randn(rng, 1))
  @test gradtest(x->fill(first(x), N, M, P), randn(rng, 1))
end

@testset "dot" begin
  rng = MersenneTwister(123456)
  @test gradtest((x, y)->dot(x[1], y[1]), [randn(rng)], [randn(rng)])
  @test gradtest(dot, randn(rng, 10), randn(rng, 10))
  @test gradtest(dot, randn(rng, 10, 3), randn(rng, 10, 3))
end

@test gradtest(kron, rand(5), rand(3))
@test gradtest(kron, rand(5), rand(3), rand(8))
@test gradtest(kron, rand(5,1), rand(3,1))
@test gradtest(kron, rand(5,1), rand(3,1), rand(8,1))
@test gradtest(kron, rand(5,2), rand(3,2), rand(8,2))

@testset "map" begin
  @test gradtest(xs -> sum(map(x -> x^2, xs)), rand(2,3))
  @test gradtest((xss...) -> sum(map((xs...) -> sqrt(sum(xs.^2)), xss...)), [rand(5) for _ in 1:6]...)
  function foo(y)
    bar = (x) -> x*y
    sum(map(bar, 1:5))
  end
  @test gradtest(foo, 3)
  @test gradient(v -> sum([x for x in v]), [1.1,2.2,3.3]) == ([1, 1, 1],)
end

@testset "mean" begin
  @test gradtest(mean, rand(2, 3))

  @test gradtest(x -> mean(x, dims=1), rand(2, 3))
  @test gradtest(x -> mean(x, dims=2), rand(2, 3))
  @test gradtest(x -> mean(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> mean(x, dims=[1, 2]), rand(2, 3, 4))
end

@testset "maximum" begin
  @test gradtest(maximum, rand(2, 3))

  @test gradtest(x -> maximum(x, dims=1), rand(2, 3))
  @test gradtest(x -> maximum(x, dims=2), rand(2, 3))
  @test gradtest(x -> maximum(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> maximum(x, dims=[1, 2]), rand(2, 3, 4))

  @test gradient(x -> 1 / maximum(x), [1., 2, 3])[1] == [0, 0, -1/9]
end

@testset "minimum" begin
  @test gradtest(minimum, rand(2, 3))

  @test gradtest(x -> minimum(x, dims=1), rand(2, 3))
  @test gradtest(x -> minimum(x, dims=2), rand(2, 3))
  @test gradtest(x -> minimum(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> minimum(x, dims=[1, 2]), rand(2, 3, 4))
end

@testset "(p)inv" begin
  rng, P, Q = MersenneTwister(123456), 13, 11
  A, B, C = randn(rng, P, Q), randn(rng, P, P), randn(Q, P)
  @test gradtest(pinv, A)
  @test gradtest(inv, B)
  @test gradtest(pinv, C)
end

@testset "multiplication" begin
  @testset "matrix-matrix" begin
    rng, M, P, Q = MersenneTwister(123456), 13, 7, 11
    @test gradtest(*, randn(rng, M, P), randn(rng, P, Q))
    @test gradtest(*, randn(rng, M, P), randn(rng, P))
    @test gradtest(*, randn(rng, M, 1), randn(rng, 1, Q))
    @test gradtest(*, randn(rng, M), randn(rng, 1, Q))
    @test gradtest(*, randn(rng, 10)', randn(rng, 10))

    let
      y, back = Zygote.forward(*, randn(rng, M, P), randn(rng, P))
      @test last(back(randn(rng, M))) isa Vector
    end
    let
      y, back = Zygote.forward(*, randn(rng, M), randn(rng, 1, P))
      @test first(back(randn(rng, M, P))) isa Vector
    end
  end
end

@testset "backsolve" begin
  rng, M, P, Q = MersenneTwister(123456), 13, 10, 9
  X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)
  A, B = randn(rng, P, M), randn(P, Q)
  D = collect(Diagonal(randn(rng, P)))
  L = collect(LowerTriangular(randn(rng, P, P)))
  L[diagind(L)] .= 1 .+ 0.01 .* randn(rng, P)
  U = collect(UpperTriangular(randn(rng, P, P)))
  U[diagind(U)] .= 1 .+ 0.01 .* randn(rng, P)

  # \ (Dense square)
  @test gradtest(\, X, Y)
  @test gradtest(\, X, y)

  # \ (Dense rectangular)
  @test gradtest(\, A, Y)
  @test gradtest(\, A, y)
  @test gradtest(\, B, Y)
  @test gradtest(\, B, y)

  # \ (Diagonal)
  @test gradtest(\, D, Y)
  @test gradtest(\, D, y)
  @test gradtest((D, Y)-> Diagonal(D) \ Y, D, Y)
  @test gradtest((D, Y)-> Diagonal(D) \ Y, D, y)

  # \ (LowerTriangular)
  @test gradtest(\, L, Y)
  @test gradtest(\, L, y)
  @test gradtest((L, Y) -> LowerTriangular(L) \ Y, L, Y)
  @test gradtest((L, Y) -> LowerTriangular(L) \ Y, L, y)

  # \ (UpperTriangular)
  @test gradtest(\, U, Y)
  @test gradtest(\, U, y)
  @test gradtest((U, Y) -> UpperTriangular(U) \ Y, U, Y)
  @test gradtest((U, Y) -> UpperTriangular(U) \ Y, U, y)

  # /
  @test gradtest(/, Y', X)
  @test gradtest((y, X)->y' / X, y, X)

  # / (rectangular)
  @test gradtest(/, Y', A')
  @test gradtest((y, A)->y' / A', y, A)
  @test gradtest(/, Y', B')
  @test gradtest((y, A)->y' / A', y, B)

  # / (Diagonal)
  @test gradtest((D, Y) -> Y' / D, D, Y)
  @test gradtest((D, Y) -> Y' / D, D, y)
  @test gradtest((D, Y)-> Y' / Diagonal(D), D, Y)
  @test gradtest((D, Y)-> Y' / Diagonal(D), D, y)

  # / (LowerTriangular)
  @test gradtest((L, Y) -> Y' / L, L, Y)
  @test gradtest((L, Y) -> Y' / L, L, y)
  @test gradtest((L, Y) -> Y' / LowerTriangular(L), L, Y)
  @test gradtest((L, Y) -> Y' / LowerTriangular(L), L, y)

  # / (UpperTriangular)
  @test gradtest((U, Y) -> Y' / U, U, Y)
  @test gradtest((U, Y) -> Y' / U, U, y)
  @test gradtest((U, Y) -> Y' / UpperTriangular(U), U, Y)
  @test gradtest((U, Y) -> Y' / UpperTriangular(U), U, y)

  @testset "Cholesky" begin

    # Check that the forwards pass computes the correct thing.
    @test Zygote.forward(X->cholesky(X * X' + I) \ Y, X)[1] == cholesky(X * X' + I) \ Y
    @test gradtest(X->cholesky(X * X' + I) \ Y, X)
    @test gradtest(Y->cholesky(X * X' + I) \ Y, Y)
    @test gradtest(X->cholesky(X * X' + I) \ y, X)
    @test gradtest(y->cholesky(X * X' + I) \ y, y)
  end
end

@testset "Symmetric" begin
  rng, P = MersenneTwister(123456), 7
  A = randn(rng, P, P)
  @test gradtest(Symmetric, A)
  y, back = Zygote.forward(Symmetric, A)

  @testset "back(::Diagonal)" begin
    D̄ = Diagonal(randn(rng, P))
    @test back(Diagonal(D̄))[1] isa Diagonal
    @test back(Diagonal(D̄))[1] ≈ back(Matrix(D̄))[1]
  end
end

@testset "diag" begin
  rng, P = MersenneTwister(123456), 10
  A = randn(rng, P, P)
  @test gradtest(diag, A)
end

@testset "Diagonal" begin
  rng, P = MersenneTwister(123456), 10
  d = randn(rng, P)
  @test gradtest(Diagonal, d)
  y, back = Zygote.forward(Diagonal, d)
  D̄ = randn(rng, P, P)
  @test back(D̄) == back(Diagonal(D̄))
  @test back(D̄) == back((diag=diag(D̄),))
end

@testset "dense + UniformScaling" begin
  rng = MersenneTwister(123456)
  A, λ = randn(rng, 10, 10), randn(rng)
  @test gradtest(A->A + 5I, A)
  @test gradtest(λ->A + λ[1] * I, [λ])
end

@testset "cholesky" begin
  @testset "cholesky - dense" begin
    rng, N = MersenneTwister(123456), 5
    A = randn(rng, N, N)
    @test cholesky(A' * A + I) == first(Zygote.forward(A->cholesky(A' * A + I), A))
    @test gradtest(A->cholesky(A' * A + I).U, A)
    @test gradtest(A->logdet(cholesky(A' * A + I)), A)
  end
  @testset "cholesky - scalar" begin
    rng = MersenneTwister(123456)
    y, back = Zygote.forward(cholesky, 5.0 * ones(1, 1))
    y′, back′ = Zygote.forward(cholesky, 5.0)
    C̄ = randn(rng, 1, 1)
    @test back′((factors=C̄,))[1] isa Real
    @test back′((factors=C̄,))[1] ≈ back((factors=C̄,))[1][1, 1]
  end
  @testset "cholesky - Diagonal" begin
    rng, N = MersenneTwister(123456), 3
    D = Diagonal(exp.(randn(rng, N)))
    Dmat = Matrix(D)
    y, back = Zygote.forward(cholesky, Dmat)
    y′, back′ = Zygote.forward(cholesky, D)
    C̄ = (factors=randn(rng, N, N),)
    @test back′(C̄)[1] isa Diagonal
    @test diag(back′(C̄)[1]) ≈ diag(back(C̄)[1])
  end
end

@testset "lyap" begin
  rng, N = MersenneTwister(6865943), 5
  for i = 1:5
    A = randn(rng, N, N)
    C = randn(rng, N, N)
    @test gradtest(lyap, A, C)
  end
  @test gradcheck(x->lyap(x[1],x[2]),[3.1,4.6])
end

@testset "matrix exponential" begin
  rng, N = MersenneTwister(6865931), 8
  for i = 1:5
    A = randn(rng, N, N)
    @test gradtest(exp, A)
  end
end

using Distances

Zygote.refresh()

@testset "distances" begin
  rng, P, Q, D = MersenneTwister(123456), 10, 9, 8

   # Check sqeuclidean.
  let
    x, y = randn(rng, D), randn(rng, D)
    @test gradtest(x->sqeuclidean(x, y), x)
    @test gradtest(y->sqeuclidean(x, y), y)
  end

   # Check binary colwise.
  let
    X, Y = randn(rng, D, P), randn(rng, D, P)
    @test gradtest(X->colwise(SqEuclidean(), X, Y), X)
    @test gradtest(Y->colwise(SqEuclidean(), X, Y), Y)
  end

   # Check binary pairwise.
  let
    X, Y = randn(rng, D, P), randn(rng, D, Q)
    @test gradtest(X->pairwise(SqEuclidean(), X, Y; dims=2), X)
    @test gradtest(Y->pairwise(SqEuclidean(), X, Y; dims=2), Y)
  end
  let
    Xt, Yt = randn(rng, P, D), randn(rng, Q, D)
    @test gradtest(Xt->pairwise(SqEuclidean(), Xt, Yt; dims=1), Xt)
    @test gradtest(Yt->pairwise(SqEuclidean(), Xt, Yt; dims=1), Yt)
  end

   # Check unary pairwise.
  @test gradtest(X->pairwise(SqEuclidean(), X; dims=2), randn(rng, D, P))
  @test gradtest(Xt->pairwise(SqEuclidean(), Xt; dims=1), randn(rng, P, D))
end

function cat_test(f, A::Union{AbstractVector, AbstractMatrix}...)
  @test gradtest(f, A...)
  Z, back = Zygote.forward(f, A...)
  Ā = back(randn(size(Z)))
  @test all(map((a, ā)->ā isa typeof(a), A, Ā))
end

@testset "vcat" begin

  # Vector-only.
  cat_test(vcat, randn(1))
  cat_test(vcat, randn(10))

  cat_test(vcat, randn(1), randn(1))
  cat_test(vcat, randn(5), randn(1))
  cat_test(vcat, randn(1), randn(6))
  cat_test(vcat, randn(5), randn(6))

  # Matrix-only.
  for c in [1, 2], r1 in [1, 2], r2 in [1, 2]
    cat_test(vcat, randn(r1, c))
    cat_test(vcat, randn(r1, c), randn(r2, c))
  end

  # Matrix-Vector / Vector-Matrix / Vector-Matrix-Vector.
  for r in [1, 2]
    cat_test(vcat, randn(r, 1), randn(r))
    cat_test(vcat, randn(r), randn(r, 1))
    cat_test(vcat, randn(r), randn(r, 1), randn(r))
  end
end

@testset "hcat" begin

  # Vector-only.
  for r in [1, 2]
    cat_test(hcat, randn(r))
    cat_test(hcat, randn(r), randn(r))
  end

  # Matrix-only.
  for r in [1, 2], c1 in [1, 2], c2 in [1, 2]
    cat_test(hcat, randn(r, c1), randn(r, c2))
  end

  # Matrix-Vector / Vector-Matrix / Vector-Matrix-Vector.
  for r in [1, 2], c in [1, 2]
    cat_test(hcat, randn(r, c), randn(r))
    cat_test(hcat, randn(r), randn(r, c))
    cat_test(hcat, randn(r), randn(r, c), randn(r))
  end
end

@testset "hvcat" begin
  @test gradient(xs -> hvcat((2,2),xs...)[1,1], [1,2,3,4])[1] == (1,0,0,0)
  @test gradient(xs -> hvcat((2,2),xs...)[2,1], [1,2,3,4])[1] == (0,0,1,0)
  @test gradient(xs -> hvcat((2,2),xs...)[1,2], [1,2,3,4])[1] == (0,1,0,0)
  @test gradient(xs -> hvcat((2,2),xs...)[2,2], [1,2,3,4])[1] == (0,0,0,1)
end

@testset "one(s) and zero(s)" begin
  @test Zygote.gradient(x->sum(ones(size(x))), randn(5))[1] isa Nothing
  @test Zygote.gradient(x->sum(one(x)), randn(3, 3))[1] isa Nothing
  @test Zygote.gradient(x->sum(zeros(size(x))), randn(7))[1] isa Nothing
  @test Zygote.gradient(x->sum(zero(x)), randn(3))[1] isa Nothing
end

import StatsFuns

Zygote.refresh()

@testset "xlogx" begin
  @test gradcheck(x->2.5 * StatsFuns.xlogx(x[1]), [1.0])
  @test gradcheck(x->2.5 * StatsFuns.xlogx(x[1]), [2.45])
end

@testset "logistic" begin
  @test gradcheck(x->3.0 * StatsFuns.logistic(x[1]), [-5.0])
  @test gradcheck(x->3.0 * StatsFuns.logistic(x[1]), [-1.0])
  @test gradcheck(x->3.0 * StatsFuns.logistic(x[1]), [-eps()])
  @test gradcheck(x->3.0 * StatsFuns.logistic(x[1]), [0.0])
  @test gradcheck(x->3.0 * StatsFuns.logistic(x[1]), [eps()])
  @test gradcheck(x->3.0 * StatsFuns.logistic(x[1]), [1.0])
  @test gradcheck(x->3.0 * StatsFuns.logistic(x[1]), [5.0])
end

@testset "logit" begin
  @test gradcheck(x->5.0 * StatsFuns.logit(x[1]), [0.1])
  @test gradcheck(x->5.0 * StatsFuns.logit(x[1]), [0.3])
  @test gradcheck(x->5.0 * StatsFuns.logit(x[1]), [0.5])
  @test gradcheck(x->5.0 * StatsFuns.logit(x[1]), [0.7])
  @test gradcheck(x->5.0 * StatsFuns.logit(x[1]), [0.9])
end

function test_log1pexp(T, xs)
  y = T(4.3)
  for x in xs
    @test gradcheck(x->y * StatsFuns.log1pexp(x[1]), [x])
  end
end

@testset "log1pexp" begin
  @testset "Float64" begin
    @testset "x ∈ (-∞, 18.0)" begin
      test_log1pexp(Float64, [-1000.0, -50.0, -25.0, -10.0, 0.0, 10.0, 18.0 - eps()])
    end
    @testset "x ∈ [18.0, 33.3)" begin
      test_log1pexp(Float64, [18.0, 18.0 + eps(), 33.3 - eps()])
    end
    @testset "x ∈ [33.3, ∞)" begin
      test_log1pexp(Float64, [33.3, 33.3 + eps(), 100.0])
    end
  end
end

@testset "log1psq" begin
  rng = MersenneTwister(123456)
  @testset "Float64" begin
    for x in [-10.0, -5.0, -1.0, -eps(), 0.0, eps(), 1.0, 5.0, 10.0]
      @test gradcheck(x->5.1 * StatsFuns.log1psq(x[1]), [x])
    end
  end
end

@testset "logsumexp" begin
  rng = MersenneTwister(123456)
  @testset "Float64" begin
    @test gradtest(StatsFuns.logsumexp, randn(rng, 1))
    @test gradtest(StatsFuns.logsumexp, randn(rng, 1, 1))
    @test gradtest(StatsFuns.logsumexp, randn(rng, 3))
    @test gradtest(StatsFuns.logsumexp, randn(rng, 3, 4, 5))
  end
end

@testset "* sizing" begin
  @test size(Zygote.gradient((x, y)->sum(x * y), randn(1, 1), randn(1, 10))[1]) == (1, 1)
  @test size(Zygote.gradient((x, y)->sum(x * y), randn(1, 1), randn(1, 10))[2]) == (1, 10)
end

@testset "broadcast" begin
  if !Zygote.usetyped
    @test gradient(x -> sum(sin.(x)), Diagonal(randn(3)))[1][2] == 1
  end
end

using Zygote: Buffer

@testset "Buffer" begin
  @test gradient([1, 2, 3]) do x
    b = Buffer(x)
    b[:] = x
    return sum(copy(b))
  end == ([1,1,1],)

  function vstack(xs)
    buf = Buffer(xs, length(xs), 5)
    for i = 1:5
      buf[:, i] = xs
    end
    return copy(buf)
  end

  @test gradient(x -> sum(vstack(x)), [1, 2, 3]) == ([5, 5, 5],)

  buf = Buffer([1, 2, 3])
  buf[1] = 1
  copy(buf)
  @test_throws ErrorException buf[1] = 1
  @test eltype(buf) === Int
  @test length(buf) === 3
  @test ndims(buf) === 1
  @test size(buf) === (3, )
  @test size(buf, 2) === 1
  @test axes(buf) == (1:3, )
  @test axes(buf, 2) == 1:1
  @test eachindex(buf) == 1:3
  @test stride(buf, 2) === 3
  @test strides(buf) === (1, )
end

@testset "FillArrays" begin
  @test gradcheck(x->sum(Fill(x[], (2, 2))), [0.1])
  @test first(Zygote.gradient(sz->sum(Ones(sz)), 6)) === nothing
  @test first(Zygote.gradient(sz->sum(Zeros(sz)), 6)) === nothing
end

@testset "AbstractArray Addition / Subtraction / Negation" begin
  rng, M, N, P = MersenneTwister(123567), 3, 7, 11
  A, B = randn(rng, M, N, P), randn(rng, M, N, P)
  @test gradtest(+, A, B)
  @test gradtest(-, A, B)
  @test gradtest(-, A)
end

@testset "FFTW" begin
  x=[-0.353213 -0.789656 -0.270151; -0.95719 -1.27933 0.223982]
  # gradient of ifft(rfft) must be 1
  @test gradient((x)->real(ifft(fft(x))[1]),x)[1][1] == 1.0+0.0im
  @test gradient((x)->real(fft(ifft(x))[1]),x)[1][1] == 1.0+0.0im

  # check ffts for individual dimensions
  @test gradient((x)->sum(abs.(FFTW.fft(x))),x)[1] ≈ gradient((x)->sum(abs.(FFTW.fft(FFTW.fft(x,1),2))),x)[1]
  @test gradient((x)->abs(sum((FFTW.fft(x)))),x)[1] ≈ gradient((x)->abs(sum(FFTW.fft(FFTW.fft(x,1),2))),x)[1]
  @test gradient((x, dims)->sum(abs.(FFTW.fft(x,dims))),x,(1,2))[1] ≈ gradient((x)->sum(abs.(FFTW.fft(x))),x)[1]
  @test gradient((x)->sum(abs.(FFTW.fft(x,(1,2)))),x)[1] ≈ gradient((x)->sum(abs.(FFTW.fft(FFTW.fft(x,1),2))),x)[1]
  @test gradient((x, dims)->sum(abs.(FFTW.ifft(x,dims))),x,(1,2))[1] ≈ gradient((x)->sum(abs.(FFTW.ifft(x))),x)[1]
  @test gradient((x)->sum(abs.(FFTW.ifft(x,(1,2)))),x)[1] ≈ gradient((x)->sum(abs.(FFTW.ifft(FFTW.ifft(x,1),2))),x)[1]

  @test gradcheck(x->sum(abs.(FFTW.fft(x))), x)
  @test gradcheck(x->sum(abs.(FFTW.ifft(x))), x)
  @test gradcheck(x->sum(abs.(FFTW.fft(x, 1))), x)
  @test gradcheck(x->sum(abs.(FFTW.ifft(x, 1))), x)

end

@testset "FillArrays" begin
  rng, M, N = MersenneTwister(123456), 7, 11
  x, y = randn(rng), randn(rng)
  @test Zygote.gradient(x->sum(Fill(x, N)), x)[1] == N
  @test Zygote.gradient(x->sum(Fill(x, N, 3, 4)), x)[1] == N * 3 * 4
  @test Zygote.gradient((x, y)->sum(Fill(x, N)), x, y) == (N, nothing)

  let
    out, back = Zygote.forward(sum, Fill(x, N))
    @test back(nothing) isa Nothing
  end

  z = randn(rng, N)
  @test gradtest(x->Fill(first(x), N), [x])
  let
    out, back = Zygote.forward(x->Fill(x, N), x)
    @test out == Fill(x, N)
    @test first(back(Fill(y, N))) ≈ y * N
  end

  # Test unary broadcasting gradients.
  out, back = Zygote.forward(x->exp.(x), Fill(x, N))
  @test out isa Fill
  @test out == Fill(exp(x), N)
  @test back(Ones(N))[1] isa Fill
  @test back(Ones(N))[1] == Ones(N) .* exp(x)
  @test back(ones(N))[1] isa Vector
  @test back(ones(N))[1] == ones(N) .* exp(x)
  @test gradtest(x->exp.(Fill(3 * first(x), N)), [x])

  @testset "broadcast + and *" begin
    for sx in [(M, N), (M, 1), (1, N), (1, 1)]
      for sy in [(M, N), (M, 1), (1, N), (1, 1)]
        z = randn(rng, broadcast_shape(sx, sy))

        # Addition
        @test gradtest((x, y)->Fill(first(x), sx...) .+ Fill(first(y), sy...), [x], [y])
        @test gradtest(x->Fill(first(x), sx...) .+ Ones(sy...), [x])
        @test gradtest(x->Fill(first(x), sx...) .+ Zeros(sy...), [x])
        @test gradtest(y->Ones(sx...) .+ Fill(first(y), sy...), [y])
        @test gradtest(y->Zeros(sx...) .+ Fill(first(y), sy...), [y])

        # Multiplication
        @test gradtest((x, y)->Fill(first(x), sx...) .* Fill(first(y), sy...), [x], [y])
        @test gradtest(x->Fill(first(x), sx...) .* Ones(sy...), [x])
        @test gradtest(x->Fill(first(x), sx...) .* Zeros(sy...), [x])
        @test gradtest(y->Ones(sx...) .* Fill(first(y), sy...), [y])
        @test gradtest(y->Zeros(sx...) .* Fill(first(y), sy...), [y])
      end
    end
  end
end
