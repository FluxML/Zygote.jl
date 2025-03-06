@testitem "gradcheck pt. 3" setup=[GradCheckSetup] begin

using Random
using LinearAlgebra
using Statistics
using SparseArrays
using FillArrays
using AbstractFFTs
using FFTW
using Distances
using Zygote: gradient, Buffer
using Base.Broadcast: broadcast_shape
using Distributed: pmap, CachingPool, workers

import FiniteDifferences
import LogExpFunctions

@testset "distances" begin
  rng, P, Q, D = MersenneTwister(123456), 5, 4, 3

  for (f, metric) in ((euclidean, Euclidean()), (sqeuclidean, SqEuclidean()))

    @testset "scalar input" begin
      x, y = randn(rng), randn(rng)
      @test gradtest(x -> f(x[1], y), [x])
      @test gradtest(x -> evaluate(metric, x[1], y), [x])
      @test gradtest(y -> f(x, y[1]), [y])
      @test gradtest(y -> evaluate(metric, x, y[1]), [y])
    end

    @testset "vector input" begin
      x, y = randn(rng, D), randn(rng, D)
      @test gradtest(x -> f(x, y), x)
      @test gradtest(x -> evaluate(metric, x, y), x)
      @test gradtest(y -> f(x, y), y)
      @test gradtest(y -> evaluate(metric, x, y), y)
      @test gradtest(x -> f(x, x), x)
    end

    @testset "binary colwise" begin
      X, Y = randn(rng, D, P), randn(rng, D, P)
      @test gradtest(X -> colwise(metric, X, Y), X)
      @test gradtest(Y -> colwise(metric, X, Y), Y)
      @test gradtest(X -> colwise(metric, X, X), X)
    end

    @testset "binary pairwise" begin
      X, Y = randn(rng, D, P), randn(rng, D, Q)
      @test gradtest(X -> pairwise(metric, X, Y; dims=2), X)
      @test gradtest(Y -> pairwise(metric, X, Y; dims=2), Y)

      @testset "X == Y" begin
        # Zygote's gradtest isn't sufficiently accurate to assess this, so we use
        # FiniteDifferences.jl instead.
        Y = copy(X)
        Δ = randn(P, P)
        Δ_fd = FiniteDifferences.j′vp(
                  FiniteDifferences.central_fdm(5, 1),
                  X -> pairwise(metric, X, Y; dims=2),
                  Δ, X)
        _, pb = Zygote.pullback(X -> pairwise(metric, X, Y; dims=2), X)

        # This is impressively inaccurate, but at least it doesn't produce a NaN.
        @test first(Δ_fd) ≈ first(pb(Δ)) atol=1e-3 rtol=1e-3
      end

      @testset "repeated X" begin
        Δ = randn(P, P)
        X = repeat(randn(rng, D), 1, P)

        # Single input matrix
        Δ_fd = FiniteDifferences.j′vp(
          FiniteDifferences.central_fdm(5, 1), X -> pairwise(metric, X; dims=2), Δ, X
        )
        _, pb = Zygote.pullback(X -> pairwise(metric, X; dims=2), X)

        # This is impressively inaccurate, but at least it doesn't produce a NaN.
        @test first(Δ_fd) ≈ first(pb(Δ)) atol=1e-3 rtol=1e-3

        # Two input matrices
        Y = copy(X)
        Δ_fd = FiniteDifferences.j′vp(
          FiniteDifferences.central_fdm(5, 1), X -> pairwise(metric, X, Y; dims=2), Δ, X
        )
        _, pb = Zygote.pullback(X -> pairwise(metric, X, Y; dims=2), X)

        # This is impressively inaccurate, but at least it doesn't produce a NaN.
        @test first(Δ_fd) ≈ first(pb(Δ)) atol=1e-3 rtol=1e-3
      end
    end

    @testset "binary pairwise - X and Y close" begin
      X = randn(rng, D, P)
      Y = X .+ 1e-10
      dist = pairwise(metric, X, Y; dims=2)
      @test first(pullback((X, Y)->pairwise(metric, X, Y; dims=2), X, Y)) ≈ dist
    end

    let
      Xt, Yt = randn(rng, P, D), randn(rng, Q, D)
      @test gradtest(Xt->pairwise(metric, Xt, Yt; dims=1), Xt)
      @test gradtest(Yt->pairwise(metric, Xt, Yt; dims=1), Yt)
    end

    @testset "unary pairwise" begin
      @test gradtest(X->pairwise(metric, X; dims=2), randn(rng, D, P))
      @test gradtest(Xt->pairwise(metric, Xt; dims=1), randn(rng, P, D))
    end
  end
end

@testset "vcat" begin
  # Scalar
  @test gradient((x,y) -> sum(vcat(x,y)), 1,2) == (1,1)
  @test gradient((x,y) -> sum([x;y]), 1,2) == (1,1)

  # Scalar + Vector
  @test gradient(x -> sum(vcat(x, 1, x)), rand(3)) == ([2,2,2],)
  @test gradient((x,y) -> sum(vcat(x, y, y)), rand(3), 4) == ([1,1,1], 2)

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
  # Scalar
  @test gradient((x,y) -> sum(hcat(x,y)), 1,2) == (1,1)
  @test gradient((x,y) -> sum([x y]), 1,2) == (1,1)
  @test gradient((a,b,c,d) -> sum(sqrt, [a b;c d]), 1,1,1,4) == (0.5, 0.5, 0.5, 0.25)

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
  @test gradient(xs -> hvcat((2,2),xs...)[1,1], [1,2,3,4])[1] == [1,0,0,0]
  @test gradient(xs -> hvcat((2,2),xs...)[2,1], [1,2,3,4])[1] == [0,0,1,0]
  @test gradient(xs -> hvcat((2,2),xs...)[1,2], [1,2,3,4])[1] == [0,1,0,0]
  @test gradient(xs -> hvcat((2,2),xs...)[2,2], [1,2,3,4])[1] == [0,0,0,1]
  # https://github.com/FluxML/Zygote.jl/issues/513
  @test gradient(x -> hvcat((2,2),1,2,3,x)[4], 4.0) == (1.0,)
end

@testset "cat(..., dims = $dim)" for dim in 1:5
  catdim = (x...) -> cat(x..., dims = dim)
  @test gradtest(catdim, rand(5), rand(5))
  @test gradtest(catdim, rand(2,5), rand(2,5), rand(2,5))
  @test gradtest(catdim, rand(2,5,3), rand(2,5,3), rand(2,5,3))
end

@testset "cat(..., dims = Val($dim))" for dim in 1:5
  catdim = (x...) -> cat(x..., dims = Val(dim))
  @test gradtest(catdim, rand(5), rand(5))
  @test gradtest(catdim, rand(2,5), rand(2,5), rand(2,5))
  @test gradtest(catdim, rand(2,5,3), rand(2,5,3), rand(2,5,3))
end

@testset "cat empty" begin
  catdim = (x...) -> cat(x..., dims = (1, 2))
  @test gradtest(catdim, rand(0,5,3), rand(2,5,3), rand(2,5,3))
end

@testset "one(s) and zero(s)" begin
  @test Zygote.gradient(x->sum(ones(size(x))), randn(5))[1] isa Nothing
  @test Zygote.gradient(x->sum(one(x)), randn(3, 3))[1] isa Nothing
  @test Zygote.gradient(x->sum(zeros(size(x))), randn(7))[1] isa Nothing
  @test Zygote.gradient(x->sum(zero(x)), randn(3))[1] isa Nothing
end

@testset "fma and muladd" begin
    @test gradcheck(x -> fma(x[1], x[2], x[3]), [2.0, 3.0, 5.0])
    @test gradcheck(x -> muladd(x[1], x[2], x[3]), [2.0, 3.0, 5.0])
end

@testset "xlogx" begin
  @test gradcheck(x->2.5 * LogExpFunctions.xlogx(x[1]), [1.0])
  @test gradcheck(x->2.5 * LogExpFunctions.xlogx(x[1]), [2.45])
  @test gradtest(x -> LogExpFunctions.xlogx.(x), (3,3))
end

@testset "xlogy" begin
  @test gradcheck(x -> LogExpFunctions.xlogy(x[1], x[2]), [1.0, 2.0])
  @test gradcheck(x -> LogExpFunctions.xlogy(x[1], x[2]), [0.0, 2.0])
  @test gradtest((x,y) -> LogExpFunctions.xlogy.(x,y), (3,3), (3,3))
end

@testset "logistic" begin
  @test gradcheck(x->3.0 * LogExpFunctions.logistic(x[1]), [-5.0])
  @test gradcheck(x->3.0 * LogExpFunctions.logistic(x[1]), [-1.0])
  @test gradcheck(x->3.0 * LogExpFunctions.logistic(x[1]), [-eps()])
  @test gradcheck(x->3.0 * LogExpFunctions.logistic(x[1]), [0.0])
  @test gradcheck(x->3.0 * LogExpFunctions.logistic(x[1]), [eps()])
  @test gradcheck(x->3.0 * LogExpFunctions.logistic(x[1]), [1.0])
  @test gradcheck(x->3.0 * LogExpFunctions.logistic(x[1]), [5.0])
end

@testset "logit" begin
  @test gradcheck(x->5.0 * LogExpFunctions.logit(x[1]), [0.1])
  @test gradcheck(x->5.0 * LogExpFunctions.logit(x[1]), [0.3])
  @test gradcheck(x->5.0 * LogExpFunctions.logit(x[1]), [0.5])
  @test gradcheck(x->5.0 * LogExpFunctions.logit(x[1]), [0.7])
  @test gradcheck(x->5.0 * LogExpFunctions.logit(x[1]), [0.9])
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
  @test gradcheck(x->2.5 * LogExpFunctions.log1pexp(x[1]), [1.0])
  @test gradcheck(x->2.5 * LogExpFunctions.log1pexp(x[1]), [2.45])
  @test gradtest(x -> LogExpFunctions.log1pexp.(x), (3,3))
end

@testset "log1psq" begin
  rng = MersenneTwister(123456)
  @testset "Float64" begin
    for x in [-10.0, -5.0, -1.0, -eps(), 0.0, eps(), 1.0, 5.0, 10.0]
      @test gradcheck(x->5.1 * LogExpFunctions.log1psq(x[1]), [x])
    end
  end
end

@testset "logaddexp" begin
  @test gradcheck(x -> LogExpFunctions.logaddexp(x[1], x[2]), [1.0, 2.0])
  @test gradcheck(x -> LogExpFunctions.logaddexp(x[1], x[2]), [1.0, -1.0])
  @test gradcheck(x -> LogExpFunctions.logaddexp(x[1], x[2]), [-2.0, -3.0])
  @test gradcheck(x -> LogExpFunctions.logaddexp(x[1], x[2]), [5.0, 5.0])
  @test gradtest((x,y) -> LogExpFunctions.logaddexp.(x,y), (3,3), (3,3))
end

@testset "logsubexp" begin
  @test gradcheck(x -> LogExpFunctions.logsubexp(x[1], x[2]), [1.0, 2.0])
  @test gradcheck(x -> LogExpFunctions.logsubexp(x[1], x[2]), [1.0, -1.0])
  @test gradcheck(x -> LogExpFunctions.logsubexp(x[1], x[2]), [-2.0, -3.0])
  @test gradtest((x,y) -> LogExpFunctions.logsubexp.(x,y), (3,3), (3,3))
end

@testset "logsumexp" begin
  rng = MersenneTwister(123456)
  @testset "Float64" begin
    @test gradtest(LogExpFunctions.logsumexp, randn(rng, 1))
    @test gradtest(LogExpFunctions.logsumexp, randn(rng, 1, 1))
    @test gradtest(LogExpFunctions.logsumexp, randn(rng, 3))
    @test gradtest(LogExpFunctions.logsumexp, randn(rng, 3, 4, 5))
    @test gradtest(x -> sum(LogExpFunctions.logsumexp(x; dims=1)), randn(rng, 4, 4))
  end
end

@testset "* sizing" begin
  @test size(Zygote.gradient((x, y)->sum(x * y), randn(1, 1), randn(1, 10))[1]) == (1, 1)
  @test size(Zygote.gradient((x, y)->sum(x * y), randn(1, 1), randn(1, 10))[2]) == (1, 10)
end

@testset "broadcast" begin
  # Before https://github.com/FluxML/Zygote.jl/pull/1001 this gave [1 1 1; 1 0 1; 1 1 -1]
  @test gradient(x -> sum(sin.(x)), Diagonal([0,pi/2,pi]))[1] ≈ [1 0 0; 0 0 0; 0 0 -1]

  a = rand(3)
  b = rand(2,2)

  @test gradcheck(x -> sum(sum(diag.((x,) .* a))), b)
  @test gradcheck(x -> sum(sum(diag.(Ref(x) .* a))), b)
  @test gradcheck(x -> sum(sum(diag.([x] .* a))), b)

  # tests for https://github.com/FluxML/Zygote.jl/issues/724
  x1 = rand(3, 3)
  @test gradient(x -> sum(x .== 0.5), x1)[1] === nothing
  @test gradient(x -> sum(x .* (x .== maximum(x, dims=1))), x1)[1] == (x1 .== maximum(x1, dims=1))

  # tests for un-broadcasting *, / via scalar rules
  @test all(gradient((x,y) -> sum(x .* y), [1,2], 5) .≈ ([5, 5], 3))
  @test all(gradient((x,y) -> sum(x .* y), 5, [1,2]) .≈ (3, [5, 5]))
  @test all(gradient((x,y) -> sum(x .* y), [1,2], [3 4 5]) .≈ ([12, 12], [3 3 3]))
  @test all(gradient((x,y) -> sum(x ./ y), [1,2], 5) .≈ ([0.2, 0.2], -0.12))

  # https://github.com/FluxML/Zygote.jl/pull/1171
  sm = sprand(5, 5, 0.5)
  @test gradient(x -> sum(abs2, Float32.(x)), sm)[1] ≈ gradient(x -> sum(abs2, x), Matrix{Float32}(sm))[1]
  @test gradient(x -> real(sum(ComplexF32.(x) .+ 1 .+ im)), sm)[1] isa SparseMatrixCSC{Float64}

  # https://github.com/FluxML/Zygote.jl/issues/1178
  function f1179(x)
    fs = Ref.(x)
    getindex.(fs)
  end
  @test gradient(sum∘f1179, ones(2)) == ([2.0, 2.0],)
end

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
  @test collect(buf) == collect(copy(buf))

  @test gradient([1, 2, 3]) do xs
    b = Zygote.Buffer(xs)
    b .= xs .* 2
    return sum(copy(b))
  end == ([2,2,2],)

  @test gradient([1, 2, 3]) do xs
    b = Zygote.Buffer(xs)
    b .= 2
    return sum(copy(b))
  end == (nothing,)

  @test gradient(1.1) do p
    b = Zygote.Buffer(zeros(3))
    b .= (p*i for i in eachindex(b))
    return sum(copy(b) .* (2:4))
  end[1] ≈ 1*2 + 2*3 + 3*4

  @test gradient(1.1) do p
    b = Zygote.Buffer(zeros(3))
    copyto!(b, [p*i for i in eachindex(b)])
    return sum(copy(b) .* (2:4))
  end[1] ≈ 1*2 + 2*3 + 3*4

  @test gradient(1.1) do p
    b = Zygote.Buffer(zeros(3))
    copyto!(b, (p*i for i in eachindex(b)))
    return sum(copy(b) .* (2:4))
  end[1] ≈ 1*2 + 2*3 + 3*4

  @test_broken gradient(1.1) do p
    b = Zygote.Buffer(zeros(3))
    copyto!(b, p)
    return sum(copy(b) .* (2:4))
  end[1] ≈ 1*2

  @test gradient(2) do x
    b = Zygote.Buffer([])
    push!(b, x)
    push!(b, 3)
    prod(copy(b))
  end == (3,)

  # backwards pass Buffer widening (#1349)
  @test Zygote.hessian(1.) do A
    buf = Zygote.Buffer([0, 0])
    buf[:] = [1, 2]
    sum(A^2 .* copy(buf))
  end == 6
  @test Zygote.hessian(1.) do A
    buf = Zygote.Buffer([0, 0])
    buf[1] = 1
    A^2 * buf[1]
  end == 2

  # Buffer storing arrays test
  W1 = ones(3, 3)
  W2 = ones(3, 3)
  x = ones(3, 1)

  function buffer_arrays(W1, W2, x)
    b = Buffer([])
    push!(b, W1 * x)
    push!(b, W2 * x)
    return sum(vcat(copy(b)...))
  end

  ∇W1, ∇W2, ∇x = gradient((W1, W2, x) -> buffer_arrays(W1, W2, x), W1, W2, x)

  @test ∇W1 == W1
  @test ∇W2 == W2
  @test ∇x == 6 .* x

  # reduced mwe of #1352
  @test Zygote.gradient([0,0]) do x
      buf = Zygote.Buffer(similar(x))
      buf[:] = x
      sum(copy(buf[1:2]))
  end == ([1,1],)

end

end
