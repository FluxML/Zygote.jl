using Zygote, NNlib, Test, Random, LinearAlgebra, Statistics, FillArrays,
    AbstractFFTs, FFTW, Distances
using Zygote: gradient
using NNlib: conv, ∇conv_data, depthwiseconv, batched_mul
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

Random.seed!(0)

@testset "println, show, print" begin
  function foo(x)
    Base.show(x)
    Base.print(x)
    Base.println(x)
    Core.show(x)
    Core.print(x)
    Core.println(x)
    return x
  end
  @test gradtest(foo, [5.0])
end

@testset "string, repr" begin
  function foo(x)
    string(x)
    repr(x)
    return x
  end
  @test gradtest(foo, [5.0])
end


@test gradient(//, 2, 3) === (1//3, -2//9)

@testset "power" begin
  @test gradient(x -> x^2, -2) == (-4,)
  @test gradient(x -> x^10, -1.0) == (-10,) # literal_pow
  pow = 10
  @test gradient(x -> x^pow, -1.0) == (-pow,)
  @test gradient(p -> real(2^p), 2)[1] ≈ 4*log(2)

  @test gradient(xs ->sum(xs .^ 2), [2, -1]) == ([4, -2],)
  @test gradient(xs ->sum(xs .^ 10), [3, -1]) == ([10*3^9, -10],)
  @test gradient(xs ->sum(xs .^ pow), [4, -1]) == ([pow*4^9, -10],)

  @test gradient(x -> real((1+3im) * x^2), 5+7im) == (-32 - 44im,)
  @test gradient(p -> real((1+3im) * (5+7im)^p), 2)[1] ≈ (-234 + 2im)*log(5 - 7im)
  # D[(1+3I)x^p, p] /. {x->5+7I, p->2} // Conjugate
end

@test gradtest((a,b)->sum(reim(acosh(complex(a[1], b[1])))), [-2.0], [1.0])

@test gradtest((x, W, b) -> identity.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> identity.(W*x .+ b), (5,3), (2,5), 2)

@test gradtest((x, W, b) -> relu.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> relu.(W*x .+ b), (5,3), (2,5), 2)

@test gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)
@test gradtest((x, W, b) -> logσ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> logσ.(W*x .+ b), (5,3), (2,5), 2)

@test gradtest((w, x) -> w'*x, randn(10, 2), randn(10))
@test gradtest((w, x) -> Adjoint(w)*x, randn(10, 2), randn(10))
@test gradtest((w, x) -> transpose(w)*x, randn(5,5), randn(5,5))
@test gradtest((w, x) -> Transpose(w)*x, randn(5,5), randn(5,5))

@test gradtest((w, x) -> parent(w)*x, randn(5,5)', randn(5,5))
@test gradtest((w, x) -> parent(w)*x, transpose(randn(5,5)), randn(5,5))

@testset "sum, prod, cumsum" begin 
  @test gradtest(x -> sum(x, dims = (2, 3)), (3,4,5))
  @test gradtest(x -> sum(abs2, x), randn(4, 3, 2))
  @test gradtest(x -> sum(abs2, x; dims=1), randn(4, 3, 2))
  @test gradtest(x -> sum(x[i] for i in 1:length(x)), randn(10))
  @test gradtest(x -> sum(i->x[i], 1:length(x)), randn(10)) #  issue #231
  @test gradtest(x -> sum((i->x[i]).(1:length(x))), randn(10))
  @test gradtest(X -> sum(x -> x^2, X), randn(10))
  @test gradtest(X -> sum(sum(x -> x^2, X; dims=1)), randn(10)) # issue #681

  # https://github.com/FluxML/Zygote.jl/issues/314
  @test gradient((x,y) -> sum(yi -> yi*x, y), 1, [1,1]) == (2, [1, 1])
  @test gradient((x,y) -> prod(yi -> yi*x, y), 1, [1,1]) == (2, [1, 1])

  @test gradient((x,y) -> sum(map(yi -> yi*x, y)), 1, [1,1]) == (2, [1, 1])
  @test gradient((x,y) -> prod(map(yi -> yi*x, y)), 1, [1,1]) == (2, [1, 1])

  @test gradtest(x -> prod(x, dims = (2, 3)), (3,4,5))
  @test gradtest(x -> prod(x), (3,4))
  @test gradient(x -> prod(x), (1,2,3))[1] == (6,3,2)

  @test gradtest(x -> cumsum(x, dims=2), (3,4,5))
  @test gradtest(x -> cumsum(x, dims=1), (3,))
  @test gradtest(x -> cumsum(x), (4,))
  @test gradtest(x -> cumsum(x, dims=3), (5,))  # trivial
  @test gradtest(x -> cumsum(x, dims=3), (3,4)) # trivial
end

@test gradtest(x -> softmax(x).*(1:3), 3)
@test gradtest(x -> softmax(x).*(1:3), (3,5))
@test gradtest(x -> softmax(x, dims=2).*(1:3), (3,5))
@test gradtest(x -> logsoftmax(x).*(1:3), 3)
@test gradtest(x -> logsoftmax(x).*(1:3), (3,5))
@test gradtest(x -> logsoftmax(x, dims=2).*(1:3), (3,5))

@test gradtest(x -> x', rand(5))

@test gradtest(det, (4, 4))
@test gradtest(logdet, map(x -> x*x', (rand(4, 4),))[1])
@test gradtest(x -> logabsdet(x)[1], (4, 4))
@test gradient(det, 2.0)[1] == 1
@test gradient(logdet, 2.0)[1] == 0.5

@testset "getindex" begin
  @test gradtest(x -> x[:, 2, :], (3, 4, 5))
  @test gradtest(x -> x[1:2, 3:4], (3, 4))

  imat = [1 2; 3 4]
  @test gradtest(x -> x[:, imat], (3, 4))
  @test gradtest(x -> x[:, [1, 2, 2]], (3, 4))
  irep = [1 2; 2 2]
  @test gradtest(x -> x[1, irep], (3, 4))

  # https://github.com/invenia/Nabla.jl/issues/139
  x = rand(3)
  z = [1, 2, 3, 3]
  y(x, z) = dot(ones(4), x[z])
  @test gradient(y, x, z) == ([1, 1, 2], nothing)

  # https://github.com/FluxML/Zygote.jl/issues/376
  _, back = Zygote._pullback(x->x[1]*im, randn(2))
  @test back(1.0)[2] == [-im, 0]

  # _droplike
  @test gradient(x -> sum(inv, x[1, :]'), ones(2, 2)) == ([-1 -1; 0 0],)
  @test gradient(x -> sum(inv, x[1:1, :]'), ones(2, 2)) == ([-1 -1; 0 0],)
  @test gradient(x -> sum(inv, transpose(view(x, 1, :))), ones(2, 2)) == ([-1 -1; 0 0],)

  # https://github.com/FluxML/Zygote.jl/issues/513
  @test gradient(p -> sum(Float32[1, 0] - p), [2, 3]) == ([-1, -1],)
  @test gradient(x -> sum(Float32[1, x] .+ x), 4) == (3.0f0,)

  # Ensure that nothings work with numeric types.
  _, back = Zygote.pullback(getindex, randn(4), [1])
  @test back([nothing]) == (zeros(4), nothing)

  # Ensure that nothings work with non-numeric types.
  _, back = Zygote.pullback(getindex, [randn(2) for _ in 1:3], [1])
  @test back([nothing]) == ([nothing for _ in 1:3], nothing)
end

@testset "view" begin
  @test gradtest(x -> view(x,:,2,:), (3,4,5))
  @test gradtest(x -> view(x,1:2,3:4), (3,4))
  @test gradtest(x -> view(x,:,[1,2,2]), (3,4))

  # https://github.com/FluxML/Zygote.jl/issues/272
  g(x) = view(x,1:2)[1]
  @test gradient(g, ones(3)) == ([1,0,0],)
end

@testset "collect" begin
  @test gradient(x -> sum(inv, collect(x)), (1,2)) === ((-1.0, -1/4),)

  @test gradient(x -> sum(collect(view(x, 1:1))), rand(2)) == ([1,0],)
  @test gradient(x -> sum(inv, collect(view(x', 1,:))), ones(2,2)) == ([-1 0; -1 0],)

  @test gradient(xs -> sum(inv, [x^2 for x in xs]), ones(2)) == ([-2, -2],)
end

@testset "conv: spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
  x = rand(repeat([5], spatial_rank)..., 3, 2)
  w = rand(repeat([3], spatial_rank)..., 3, 3)
  cdims = DenseConvDims(x, w)
  @test gradtest((x, w) -> conv(x, w, cdims), x, w)
  @test gradtest((x, w) -> sum(conv(x, w, cdims)), x, w)  # https://github.com/FluxML/Flux.jl/issues/1055

  y = conv(x, w, cdims)
  @test gradtest((y, w) -> ∇conv_data(y, w, cdims), y, w)
  if spatial_rank == 3
    @test_broken gradtest((y, w) -> sum(∇conv_data(y, w, cdims)), y, w)
  else
    @test gradtest((y, w) -> sum(∇conv_data(y, w, cdims)), y, w)
  end

  dcdims = DepthwiseConvDims(x, w)
  @test gradtest((x, w) -> depthwiseconv(x, w, dcdims), x, w)

  y = depthwiseconv(x, w, dcdims)
  @test gradtest((y, w) -> ∇depthwiseconv_data(y, w, dcdims), y, w)
  if spatial_rank == 3
    @test_broken gradtest((y, w) -> sum(∇depthwiseconv_data(y, w, dcdims)), y, w)
  else
    @test gradtest((y, w) -> sum(∇depthwiseconv_data(y, w, dcdims)), y, w)
  end
end

@testset "pooling: spatial_rank=$spatial_rank" for spatial_rank in (1, 2)
  x = rand(repeat([10], spatial_rank)..., 3, 2)
  pdims = PoolDims(x, 2)
  @test gradtest(x -> maxpool(x, pdims), x)
  @test gradtest(x -> meanpool(x, pdims), x)
  @test gradtest(x -> sum(maxpool(x, pdims)), x)
  @test gradtest(x -> sum(meanpool(x, pdims)), x)

  #https://github.com/FluxML/NNlib.jl/issues/188
  k = ntuple(_ -> 2, spatial_rank)  # Kernel size of pool in ntuple format
  @test gradtest(x -> maxpool(x, k), x)
  @test gradtest(x -> meanpool(x, k), x)
  @test gradtest(x -> sum(maxpool(x, k)), x)
  @test gradtest(x -> sum(meanpool(x, k)), x)
end

@test gradtest(x -> reverse(x), rand(17))
@test gradtest(x -> reverse(x, 8), rand(17))
@test gradtest(x -> reverse(x, 8, 13), rand(17))
@test gradtest(x -> reverse(x, dims=2), rand(17, 42))

@test gradtest(x -> permutedims(x), rand(2))
@test gradtest(x -> permutedims(x), rand(2,3))
@test gradtest(x -> permutedims(x, [3,1,2]), rand(4,5,6))
@test gradtest(x -> PermutedDimsArray(x, (3,1,2)), rand(4,5,6))
let
  y, back = Zygote.pullback(permutedims, randn(3))
  @test first(back(randn(1, 3))) isa Vector
end

@test gradtest(x -> repeat(x; inner=2), rand(5))
@test gradtest(x -> repeat(x; inner=2, outer=3), rand(5))
@test gradtest(x -> repeat(x; inner=(2,2,1), outer=(1,1,3)), rand(5,4,3))

@test gradtest(x -> repeat(x, 3), rand(5))
@test gradtest(x -> repeat(x, 2, 3), rand(5))
@test gradtest(x -> repeat(x, 5), rand(5,7))
@test gradtest(x -> repeat(x, 3, 2), rand(5,3))

@test gradtest(tr, rand(4, 4))

@testset "fill" begin
  rng, N, M, P = MersenneTwister(123456), 11, 6, 5
  @test gradtest(x->fill(first(x), N), randn(rng, 1))
  @test gradtest(x->fill(first(x), N, M), randn(rng, 1))
  @test gradtest(x->fill(first(x), N, M, P), randn(rng, 1))
end

@testset "circshift" begin
  L = 5
  for D ∈ 1:5, reps ∈ 1:5
    x0 = zeros(ntuple(d->L, D))
    g = gradient(x -> x[1], x0)[1] #Zero shift gradient
    shift = ntuple(_ -> rand(-L:L), D) #Random shift
    @test gradient(x -> circshift(x, shift)[1], x0)[1] == circshift(g, map(-, shift))
  end
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

@testset "sort" begin
  @test gradtest(sort, 5)
  correct = [
      [2,3,1],
      [1, 2, 3],
      [1,2,3],
      [2,1,3]
  ]
  for i = 1:3
    @test gradient(v->sort(v)[i], [3.,1,2])[1][correct[1][i]] == 1
    @test gradient(v->sort(v)[i], [1.,2,3])[1][correct[2][i]] == 1
    @test gradient(v->sort(v,by=x->x%10)[i], [11,2,99])[1][correct[3][i]] == 1
    @test gradient(v->sort(v,by=x->x%10)[i], [2,11,99])[1][correct[4][i]] == 1
  end
end

@testset "filter" begin
  @test gradtest(xs -> filter(x -> x > 0.5, xs), 20)

  @test gradient(x -> sum(log, filter(iseven, x)), 1:10) ==
    (map(x -> iseven(x) ? 1/x : 0, 1:10),)
  @test gradient(x -> sum(abs2, im .+ filter(iseven, x)), 1:10) ==
    (map(x -> iseven(x) ? 2x+2im : 0, 1:10),)
end


@testset "mean" begin
  @test gradtest(mean, rand(2, 3))

  @test gradtest(x -> mean(x, dims=1), rand(2, 3))
  @test gradtest(x -> mean(x, dims=2), rand(2, 3))
  @test gradtest(x -> mean(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> mean(x, dims=[1, 2]), rand(2, 3, 4))
end

@testset "var" begin
  @test gradtest(var, rand(2, 3))
  @test gradtest(x -> var(x, dims=1), rand(2, 3))
  @test gradtest(x -> var(x, dims=2), rand(2, 3))
  @test gradtest(x -> var(x, dims=3), rand(2, 3, 4))
  @test gradtest(x -> var(x, dims=[1, 2]), rand(2, 3, 4))


  @test gradtest(x -> var(x, corrected=false), rand(2, 3))
  @test gradtest(x -> var(x, dims=1, corrected=false), rand(2, 3))
  @test gradtest(x -> var(x, dims=2, corrected=false), rand(2, 3))
  @test gradtest(x -> var(x, dims=3, corrected=false), rand(2, 3, 4))
  @test gradtest(x -> var(x, dims=[1, 2], corrected=false), rand(2, 3, 4))

  @test gradtest(x -> var(x, mean=mean(x)), rand(2, 3))
  @test gradtest(x -> var(x, dims=1, mean=mean(x, dims=1)), rand(2, 3))
  @test gradtest(x -> var(x, dims=2, mean=mean(x, dims=2)), rand(2, 3))
  @test gradtest(x -> var(x, dims=3, mean=mean(x, dims=3)), rand(2, 3, 4))
  @test gradtest(x -> var(x, dims=[1, 2], mean=mean(x, dims=[1, 2])), rand(2, 3, 4))

  @test gradtest(x -> var(x, corrected=false, mean=mean(x)), rand(2, 3))
  @test gradtest(x -> var(x, dims=1, corrected=false, mean=mean(x, dims=1)), rand(2, 3))
  @test gradtest(x -> var(x, dims=2, corrected=false, mean=mean(x, dims=2)), rand(2, 3))
  @test gradtest(x -> var(x, dims=3, corrected=false, mean=mean(x, dims=3)), rand(2, 3, 4))
  @test gradtest(x -> var(x, dims=[1, 2], corrected=false, mean=mean(x, dims=[1, 2])), rand(2, 3, 4))
end

@testset "std" begin
  @test gradtest(std, rand(2, 3))
  @test gradtest(x -> std(x, dims=1), rand(2, 3))
  @test gradtest(x -> std(x, dims=2), rand(2, 3))
  @test gradtest(x -> std(x, dims=3), rand(2, 3, 4))
  @test gradtest(x -> std(x, dims=[1, 2]), rand(2, 3, 4))


  @test gradtest(x -> std(x, corrected=false), rand(2, 3))
  @test gradtest(x -> std(x, dims=1, corrected=false), rand(2, 3))
  @test gradtest(x -> std(x, dims=2, corrected=false), rand(2, 3))
  @test gradtest(x -> std(x, dims=3, corrected=false), rand(2, 3, 4))
  @test gradtest(x -> std(x, dims=[1, 2], corrected=false), rand(2, 3, 4))

  @test gradtest(x -> std(x, mean=mean(x)), rand(2, 3))
  @test gradtest(x -> std(x, dims=1, mean=mean(x, dims=1)), rand(2, 3))
  @test gradtest(x -> std(x, dims=2, mean=mean(x, dims=2)), rand(2, 3))
  @test gradtest(x -> std(x, dims=3, mean=mean(x, dims=3)), rand(2, 3, 4))
  @test gradtest(x -> std(x, dims=[1, 2], mean=mean(x, dims=[1, 2])), rand(2, 3, 4))

  @test gradtest(x -> std(x, corrected=false, mean=mean(x)), rand(2, 3))
  @test gradtest(x -> std(x, dims=1, corrected=false, mean=mean(x, dims=1)), rand(2, 3))
  @test gradtest(x -> std(x, dims=2, corrected=false, mean=mean(x, dims=2)), rand(2, 3))
  @test gradtest(x -> std(x, dims=3, corrected=false, mean=mean(x, dims=3)), rand(2, 3, 4))
  @test gradtest(x -> std(x, dims=[1, 2], corrected=false, mean=mean(x, dims=[1, 2])), rand(2, 3, 4))
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

@testset "dropdims" begin
  @test gradtest(x -> dropdims(x, dims = 3), rand(2, 2, 1, 2))
  @test gradtest(x -> dropdims(x, dims = (2, 3)), rand(2, 1, 1, 3))
  @test gradtest(x -> dropdims(x, dims = (1, 2, 3)), rand(1, 1, 1, 3))
end

@testset "$f(::AbstractArray)" for f in (real, conj, imag)
  rng, N = MersenneTwister(123456), 3
  Ts = (Float64, ComplexF64)
  @testset "$f(::Array{$IT})" for IT in Ts
    A = randn(IT, N, N)
    y, back = Zygote.pullback(f, A)
    y2, back2 = Zygote.pullback(x->f.(x), A)
    @test y == y2
    @testset "back(::Array{$BT})" for BT in Ts
      ȳ = randn(BT, N, N)
      @test back(ȳ)[1] == back2(ȳ)[1]
    end
  end
end

@testset "(p)inv" begin
  rng, P, Q = MersenneTwister(123456), 13, 11
  A, B, C = randn(rng, P, Q), randn(rng, P, P), randn(Q, P)
  @test gradtest(pinv, A)
  @test gradtest(inv, B)
  @test gradtest(pinv, C)
  @test gradient(inv, 2.0)[1] == -0.25
end

@testset "multiplication" begin
  rng, M, P, Q = MersenneTwister(123456), 13, 7, 11
  @testset "matrix-matrix" begin

    @test gradtest(*, randn(rng, M, P), randn(rng, P, Q))
    @test gradtest(*, randn(rng, M, P), randn(rng, P))
    @test gradtest(*, randn(rng, M, 1), randn(rng, 1, Q))
    @test gradtest(*, randn(rng, M), randn(rng, 1, Q))
    @test gradtest(*, randn(rng, 10)', randn(rng, 10))
    @test gradtest(*, randn(rng, 10)', randn(rng, 10))

    let
      y, back = Zygote.pullback(*, randn(rng, M, P), randn(rng, P))
      @test last(back(randn(rng, M))) isa Vector
    end
    let
      y, back = Zygote.pullback(*, randn(rng, M), randn(rng, 1, P))
      @test first(back(randn(rng, M, P))) isa Vector
    end
  end

  @testset "batched matrix multiplication" begin
    B = 3
    @test gradtest(batched_mul, randn(rng, M, P, B), randn(rng, P, Q, B))
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

  # / (UnitLowerTriangular)
  @test gradtest((L, Y) -> Y' / L, L, Y)
  @test gradtest((L, Y) -> Y' / L, L, y)
  @test gradtest((L, Y) -> Y' / UnitLowerTriangular(L), L, Y)
  @test gradtest((L, Y) -> Y' / UnitLowerTriangular(L), L, y)

  # / (UnitUpperTriangular)
  @test gradtest((U, Y) -> Y' / U, U, Y)
  @test gradtest((U, Y) -> Y' / U, U, y)
  @test gradtest((U, Y) -> Y' / UnitUpperTriangular(U), U, Y)
  @test gradtest((U, Y) -> Y' / UnitUpperTriangular(U), U, y)

  @testset "Cholesky" begin
    # Check that the forwards pass computes the correct thing.
    f(X, Y) = cholesky(X * X' + I) \ Y
    @test Zygote.pullback(X -> f(X, Y), X)[1] == cholesky(X * X' + I) \ Y
    @test gradtest(X -> f(X, Y), X)
    @test gradtest(Y -> f(X, Y), Y)
    @test gradtest(X -> f(X, y), X)
    @test gradtest(y -> f(X, y), y)
    g(X) = cholesky(X * X' + I)
    @test Zygote.pullback(g, X)[2]((factors=LowerTriangular(X),)) ==
      Zygote.pullback(g, X)[2]((factors=Matrix(LowerTriangular(X)),))
    @test_throws PosDefException Zygote.pullback(X -> cholesky(X, check = false), X)[2]((factors=X,))
  end
end

@testset "Symmetric" begin
  @testset "real" begin
    rng, P = MersenneTwister(123456), 7
    A = randn(rng, P, P)
    @testset "uplo=$uplo" for uplo in (:U, :L)
      @test gradtest(x->Symmetric(x, uplo), A)
      y, back = Zygote.pullback(Symmetric, A, uplo)
      @test y isa Symmetric

      @testset "back(::Diagonal)" begin
        D̄ = Diagonal(randn(rng, P))
        @test back(Diagonal(D̄))[1] isa Diagonal
        @test back(Diagonal(D̄))[1] ≈ back(Matrix(D̄))[1]
      end

      @testset "back(::$TTri)" for TTri in (LowerTriangular,UpperTriangular)
        D̄ = TTri(randn(rng, P, P))
        @test back(D̄)[1] isa Matrix
        @test back(D̄)[2] === nothing
        @test back(D̄)[1] ≈ back(Matrix(D̄))[1]
      end
    end
  end

  @testset "complex" begin
    rng, P = MersenneTwister(123456), 7
    Re = randn(rng, P, P)
    Im = randn(rng, P, P)
    A = complex.(Re, Im)

    @testset "gradcheck dense" begin
      for uplo in (:U, :L)
        @test gradcheck(Re,Im) do a, b
          c = Symmetric(complex.(a, b), uplo)
          d = exp.(c)
          sum(real.(d) + imag.(d))
        end
      end
    end

    @testset "uplo=$uplo" for uplo in (:U, :L)
      y, back = Zygote.pullback(Symmetric, A, uplo)
      @test y isa Symmetric

      @testset "back(::Diagonal)" begin
        D̄ = Diagonal(complex.(randn(rng, P), randn(rng, P)))
        @test back(Diagonal(D̄))[1] isa Diagonal
        @test back(Diagonal(D̄))[1] ≈ back(Matrix(D̄))[1]
      end

      @testset "back(::$TTri)" for TTri in (LowerTriangular,UpperTriangular)
        D̄ = TTri(complex.(randn(rng, P, P), randn(rng, P, P)))
        @test back(D̄)[1] isa Matrix
        @test back(D̄)[2] === nothing
        @test back(D̄)[1] ≈ back(Matrix(D̄))[1]
      end
    end
  end
end

@testset "Hermitian" begin
  rng, P = MersenneTwister(123456), 7
  Re = randn(rng, P, P)
  Im = randn(rng, P, P)
  A = complex.(Re, Im)

  @testset "gradcheck dense" begin
    for uplo in (:U, :L)
      @test gradcheck(Re,Im) do a, b
        c = Hermitian(complex.(a, b), uplo)
        d = exp.(c)
        sum(real.(d) + imag.(d))
      end
    end
  end

  @testset "uplo=$uplo" for uplo in (:U, :L)
    y, back = Zygote.pullback(Hermitian, A, uplo)
    _, back_sym = Zygote.pullback(Symmetric, A, uplo)
    @test y isa Hermitian

    @testset "back" begin
      D̄ = randn(rng, P, P)
      @test back(D̄)[1] ≈ back_sym(D̄)[1]
    end

    @testset "back(::Diagonal)" begin
      D̄ = Diagonal(complex.(randn(rng, P), randn(rng, P)))
      @test back(D̄)[1] isa Diagonal
      @test back(D̄)[2] === nothing
      @test back(D̄)[1] ≈ back(Matrix(D̄))[1]
      @test back(real(D̄))[1] ≈ back_sym(real(D̄))[1]
    end

    @testset "back(::$TTri)" for TTri in (LowerTriangular,UpperTriangular)
      D̄ = TTri(complex.(randn(rng, P, P), randn(rng, P, P)))
      @test back(D̄)[1] isa Matrix
      @test back(D̄)[2] === nothing
      @test back(D̄)[1] ≈ back(Matrix(D̄))[1]
      @test back(real(D̄))[1] ≈ back_sym(real(D̄))[1]
    end
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
  y, back = Zygote.pullback(Diagonal, d)
  D̄ = randn(rng, P, P)
  @test back(D̄) == back(Diagonal(D̄))
  @test back(D̄) == back((diag=diag(D̄),))
end

@testset "dense + UniformScaling" begin
  rng = MersenneTwister(123456)
  A, λ = randn(rng, 10, 10), randn(rng)
  @test gradtest(A->A + 5I, A)
  @test gradtest(A->5I - A, A)
  @test gradtest(λ->A + λ[1] * I, [λ])
end

@testset "cholesky" begin
  @testset "cholesky - dense" begin
    rng, N = MersenneTwister(123456), 5
    A = randn(rng, N, N)
    @test cholesky(A' * A + I) == first(Zygote.pullback(A->cholesky(A' * A + I), A))
    @test gradtest(A->cholesky(A' * A + I).U, A)
    @test gradtest(A->logdet(cholesky(A' * A + I)), A)
    @test gradtest(B->cholesky(Symmetric(B)).U, A * A' + I)
    @test gradtest(B->logdet(cholesky(Symmetric(B))), A * A' + I)
  end
  @testset "cholesky - scalar" begin
    rng = MersenneTwister(123456)
    y, back = Zygote.pullback(cholesky, 5.0 * ones(1, 1))
    y′, back′ = Zygote.pullback(cholesky, 5.0)
    C̄ = randn(rng, 1, 1)
    @test back′((factors=C̄,))[1] isa Real
    @test back′((factors=C̄,))[1] ≈ back((factors=C̄,))[1][1, 1]
  end
  @testset "cholesky - Diagonal" begin
    rng, N = MersenneTwister(123456), 3
    D = Diagonal(exp.(randn(rng, N)))
    Dmat = Matrix(D)
    y, back = Zygote.pullback(cholesky, Dmat)
    y′, back′ = Zygote.pullback(cholesky, D)
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
  @testset "real dense" begin
    rng, N = MersenneTwister(6865931), 8
    for i = 1:5
      A = randn(rng, N, N)
      @test gradtest(exp, A)

      @testset "similar eigenvalues" begin
        λ, V = eigen(A)
        λ[1] = λ[3] + sqrt(eps(real(eltype(λ)))) / 10
        A2 = real.(V * Diagonal(λ) / V)
        @test gradtest(exp, A2)
      end
    end
  end

  @testset "complex dense" begin
    rng, N = MersenneTwister(6865931), 8
    for i = 1:5
      A = randn(rng, ComplexF64, N, N)
      @test gradcheck(reim(A)...) do a,b
        c = complex.(a, b)
        d = exp(c)
        return sum(real.(d) + 2 .* imag.(d))
      end

      @testset "similar eigenvalues" begin
        λ, V = eigen(A)
        λ[1] = λ[3] + sqrt(eps(real(eltype(λ)))) / 10
        A2 = V * Diagonal(λ) / V
        @test gradcheck(reim(A2)...) do a,b
          c = complex.(a, b)
          d = exp(c)
          return sum(real.(d) + 2 .* imag.(d))
        end
      end
    end
    A = [ 0.0    1.0    0.0
          0.0    0.0    1.0
          -4.34 -18.31  -0.43]
    _,back = Zygote.pullback(exp,A)
    Ȳ = rand(3,3)
    @test isreal(back(Ȳ)[1])
  end
end

_hermsymtype(::Type{<:Symmetric}) = Symmetric
_hermsymtype(::Type{<:Hermitian}) = Hermitian

function _gradtest_hermsym(f, ST, A)
  gradtest(_splitreim(collect(A))...) do (args...)
    B = f(ST(_joinreim(_dropimaggrad.(args)...)))
    return sum(_splitreim(B))
  end
end

@testset "eigen(::RealHermSymComplexHerm)" begin
  MTs = (Symmetric{Float64}, Hermitian{Float64}, Hermitian{ComplexF64})
  rng, N = MersenneTwister(123), 7
  @testset "eigen(::$MT)" for MT in MTs
    T = eltype(MT)
    ST = _hermsymtype(MT)

    A = ST(randn(rng, T, N, N))
    U = eigvecs(A)

    @test _gradtest_hermsym(ST, A) do (A)
      d, U = eigen(A)
      return U * Diagonal(exp.(d)) * U'
    end

    y = Zygote.pullback(eigen, A)[1]
    y2 = eigen(A)
    @test y.values ≈ y2.values
    @test y.vectors ≈ y2.vectors

    @testset "low rank" begin
      A2 = Symmetric(U * Diagonal([randn(rng), zeros(N-1)...]) * U')
      @test_broken _gradtest_hermsym(ST, A2) do (A)
        d, U = eigen(A)
        return U * Diagonal(exp.(d)) * U'
      end
    end
  end
end

@testset "eigvals(::RealHermSymComplexHerm)" begin
  MTs = (Symmetric{Float64}, Hermitian{Float64}, Hermitian{ComplexF64})
  rng, N = MersenneTwister(123), 7
  @testset "eigvals(::$MT)" for MT in MTs
    T = eltype(MT)
    ST = _hermsymtype(MT)

    A = ST(randn(rng, T, N, N))
    @test _gradtest_hermsym(A ->eigvals(A), ST, A)
    @test Zygote.pullback(eigvals, A)[1] ≈ eigvals(A)
  end
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

@testset "Hermitian/Symmetric power series functions" begin
  MTs = (Symmetric{Float64}, Hermitian{Float64}, Hermitian{ComplexF64})
  rng, N = MersenneTwister(123), 7
  domains = (Real, Complex)
  @testset "$(nameof(f))(::RealHermSymComplexHerm)" for f in (exp, log, cos, sin, tan, cosh, sinh, tanh, acos, asin, atan, acosh, asinh, atanh, sqrt)
    @testset "$(nameof(f))(::$MT)" for MT in MTs
      T = eltype(MT)
      ST = _hermsymtype(MT)
      @testset "domain $domain" for domain in domains
        preA = _randmatseries(rng, f, T, N, domain)
        preA === nothing && continue
        A = ST(preA)
        λ, U = eigen(A)

        @test _gradtest_hermsym(f, ST, A)

        y = Zygote.pullback(f, A)[1]
        y2 = f(A)
        @test y ≈ y2
        @test typeof(y) == typeof(y2)

        @testset "similar eigenvalues" begin
          λ[1] = λ[3] + sqrt(eps(eltype(λ))) / 10
          A2 = U * Diagonal(λ) * U'
          @test _gradtest_hermsym(f, ST, A2)
        end

        if f ∉ (log, sqrt) # only defined for invertible matrices
          @testset "low rank" begin
            A3 = U * Diagonal([rand(rng), zeros(N-1)...]) * U'
            @test _gradtest_hermsym(f, ST, A3)
          end
        end
      end
    end
  end

  @testset "sincos(::RealHermSymComplexHerm)" begin
    @testset "sincos(::$MT)" for MT in MTs
      T = eltype(MT)
      ST = _hermsymtype(MT)
      A = ST(_randmatseries(rng, sincos, T, N, Real))
      λ, U = eigen(A)

      @test gradtest(_splitreim(collect(A))...) do (args...)
        S,C = sincos(ST(_joinreim(_dropimaggrad.(args)...)))
        return vcat(vec.(_splitreim(S))..., vec.(_splitreim(C))...)
      end

      y = Zygote.pullback(sincos, A)[1]
      y2 = sincos(A)
      @test y[1] ≈ y2[1]
      @test typeof(y[1]) == typeof(y2[1])
      @test y[2] ≈ y2[2]
      @test typeof(y[2]) == typeof(y2[2])

      @testset "similar eigenvalues" begin
        λ[1] = λ[3] + sqrt(eps(eltype(λ))) / 10
        A2 = U * Diagonal(λ) * U'
        @test gradtest(_splitreim(collect(A2))...) do (args...)
          S,C = sincos(ST(_joinreim(_dropimaggrad.(args)...)))
          return vcat(vec.(_splitreim(S))..., vec.(_splitreim(C))...)
        end
      end

      @testset "low rank" begin
        A3 = U * Diagonal([rand(rng), zeros(N-1)...]) * U'
        @test gradtest(_splitreim(collect(A3))...) do (args...)
          S,C = sincos(ST(_joinreim(_dropimaggrad.(args)...)))
          return vcat(vec.(_splitreim(S))..., vec.(_splitreim(C))...)
        end
      end
    end
  end

  @testset "^(::RealHermSymComplexHerm, p::Real)" begin
    @testset for p in (-1.0, -0.5, 0.5, 1.0, 1.5)
      @testset "^(::$MT, $p)" for MT in MTs
        T = eltype(MT)
        ST = _hermsymtype(MT)
        @testset "domain $domain" for domain in domains
          A = ST(_randmatseries(rng, ^, T, N, domain))
          λ, U = eigen(A)

          @test gradcheck(_splitreim(collect(A))..., [p]) do (args...)
            p = _dropimaggrad(args[end][1])
            A = ST(_joinreim(_dropimaggrad.(args[1:end-1])...))
            B = A^p
            return abs(sum(sin.(B)))
          end

          y = Zygote.pullback(^, A, p)[1]
          y2 = A^p
          @test y ≈ y2
          @test typeof(y) == typeof(y2)

          @testset "similar eigenvalues" begin
            λ[1] = λ[3] + sqrt(eps(eltype(λ))) / 10
            A2 = U * Diagonal(λ) * U'
            @test gradcheck(_splitreim(collect(A2))..., [p]) do (args...)
              p = _dropimaggrad(args[end][1])
              A = ST(_joinreim(_dropimaggrad.(args[1:end-1])...))
              B = A^p
              return abs(sum(sin.(B)))
            end
          end
        end
      end
    end
  end
end

@testset "^(::Union{Symmetric,Hermitian}, p::Integer)" begin
  MTs = (Symmetric{Float64}, Symmetric{ComplexF64},
         Hermitian{Float64}, Hermitian{ComplexF64})
  rng, N = MersenneTwister(123), 7
  @testset for p in -3:3
    @testset "^(::$MT, $p)" for MT in MTs
      T = eltype(MT)
      ST = _hermsymtype(MT)
      A = ST(randn(rng, T, N, N))

      if p == 0
        @test gradient(_splitreim(collect(A))...) do (args...)
          A = ST(_joinreim(_dropimaggrad.(args)...))
          B = A^p
          return sum(sin.(vcat(vec.(_splitreim(B))...)))
        end === map(_->nothing, _splitreim(A))
      else
        @test gradtest(_splitreim(collect(A))...) do (args...)
          A = ST(_joinreim(_dropimaggrad.(args)...))
          B = A^p
          return vcat(vec.(_splitreim(B))...)
        end
      end

      y = Zygote.pullback(^, A, p)[1]
      y2 = A^p
      @test y ≈ y2
      @test typeof(y) === typeof(y2)
    end
  end
end

@testset "distances" begin
  rng, P, Q, D = MersenneTwister(123456), 10, 9, 8

  for (f, metric) in ((euclidean, Euclidean()), (sqeuclidean, SqEuclidean()))
    let
      x, y = randn(rng), randn(rng)
      @test gradtest(x -> f(x[1], y), [x])
      @test gradtest(x -> evaluate(metric, x[1], y), [x])
      @test gradtest(y -> f(x, y[1]), [y])
      @test gradtest(y -> evaluate(metric, x, y[1]), [y])
    end

    let
      x, y = randn(rng, D), randn(rng, D)
      @test gradtest(x -> f(x, y), x)
      @test gradtest(x -> evaluate(metric, x, y), x)
      @test gradtest(y -> f(x, y), y)
      @test gradtest(y -> evaluate(metric, x, y), y)
    end

    # Check binary colwise.
    let
      X, Y = randn(rng, D, P), randn(rng, D, P)
      @test gradtest(X->colwise(metric, X, Y), X)
      @test gradtest(Y->colwise(metric, X, Y), Y)
    end

    # Check binary pairwise.
    let
      X, Y = randn(rng, D, P), randn(rng, D, Q)
      @test gradtest(X->pairwise(metric, X, Y; dims=2), X)
      @test gradtest(Y->pairwise(metric, X, Y; dims=2), Y)
    end

    # Check binary pairwise when X and Y are close.
    let
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

    # Check unary pairwise.
    @test gradtest(X->pairwise(metric, X; dims=2), randn(rng, D, P))
    @test gradtest(Xt->pairwise(metric, Xt; dims=1), randn(rng, P, D))
  end
end

function cat_test(f, A::Union{AbstractVector, AbstractMatrix}...)
  @test gradtest(f, A...)
  Z, back = Zygote.pullback(f, A...)
  Ā = back(randn(size(Z)))
  @test all(map((a, ā)->ā isa typeof(a), A, Ā))
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
  @test gradient(xs -> hvcat((2,2),xs...)[1,1], [1,2,3,4])[1] == (1,0,0,0)
  @test gradient(xs -> hvcat((2,2),xs...)[2,1], [1,2,3,4])[1] == (0,0,1,0)
  @test gradient(xs -> hvcat((2,2),xs...)[1,2], [1,2,3,4])[1] == (0,1,0,0)
  @test gradient(xs -> hvcat((2,2),xs...)[2,2], [1,2,3,4])[1] == (0,0,0,1)
  # https://github.com/FluxML/Zygote.jl/issues/513
  @test gradient(x -> hvcat((2,2),1,2,3,x)[4], 4.0) == (1.0,)
end

@testset "cat(..., dims = $dim)" for dim in 1:5
  catdim = (x...) -> cat(x..., dims = dim)
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

import StatsFuns

Zygote.refresh()

@testset "xlogx" begin
  @test gradcheck(x->2.5 * StatsFuns.xlogx(x[1]), [1.0])
  @test gradcheck(x->2.5 * StatsFuns.xlogx(x[1]), [2.45])
  @test gradtest(x -> StatsFuns.xlogx.(x), (3,3))
end

@testset "xlogy" begin
  @test gradcheck(x -> StatsFuns.xlogy(x[1], x[2]), [1.0, 2.0])
  @test gradcheck(x -> StatsFuns.xlogy(x[1], x[2]), [0.0, 2.0])
  @test gradtest((x,y) -> StatsFuns.xlogy.(x,y), (3,3), (3,3))
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
  @test gradcheck(x->2.5 * StatsFuns.log1pexp(x[1]), [1.0])
  @test gradcheck(x->2.5 * StatsFuns.log1pexp(x[1]), [2.45])
  @test gradtest(x -> StatsFuns.log1pexp.(x), (3,3))
end

@testset "log1psq" begin
  rng = MersenneTwister(123456)
  @testset "Float64" begin
    for x in [-10.0, -5.0, -1.0, -eps(), 0.0, eps(), 1.0, 5.0, 10.0]
      @test gradcheck(x->5.1 * StatsFuns.log1psq(x[1]), [x])
    end
  end
end

@testset "logaddexp" begin
  @test gradcheck(x -> StatsFuns.logaddexp(x[1], x[2]), [1.0, 2.0])
  @test gradcheck(x -> StatsFuns.logaddexp(x[1], x[2]), [1.0, -1.0])
  @test gradcheck(x -> StatsFuns.logaddexp(x[1], x[2]), [-2.0, -3.0])
  @test gradcheck(x -> StatsFuns.logaddexp(x[1], x[2]), [5.0, 5.0])
  @test gradtest((x,y) -> StatsFuns.logaddexp.(x,y), (3,3), (3,3))
end

@testset "logsubexp" begin
  @test gradcheck(x -> StatsFuns.logsubexp(x[1], x[2]), [1.0, 2.0])
  @test gradcheck(x -> StatsFuns.logsubexp(x[1], x[2]), [1.0, -1.0])
  @test gradcheck(x -> StatsFuns.logsubexp(x[1], x[2]), [-2.0, -3.0])
  @test gradtest((x,y) -> StatsFuns.logsubexp.(x,y), (3,3), (3,3))
end

@testset "logsumexp" begin
  rng = MersenneTwister(123456)
  @testset "Float64" begin
    @test gradtest(StatsFuns.logsumexp, randn(rng, 1))
    @test gradtest(StatsFuns.logsumexp, randn(rng, 1, 1))
    @test gradtest(StatsFuns.logsumexp, randn(rng, 3))
    @test gradtest(StatsFuns.logsumexp, randn(rng, 3, 4, 5))
    @test gradtest(x -> sum(StatsFuns.logsumexp(x; dims=1)), randn(rng, 4, 4))
  end
end

@testset "* sizing" begin
  @test size(Zygote.gradient((x, y)->sum(x * y), randn(1, 1), randn(1, 10))[1]) == (1, 1)
  @test size(Zygote.gradient((x, y)->sum(x * y), randn(1, 1), randn(1, 10))[2]) == (1, 10)
end

@testset "broadcast" begin
  @test gradient(x -> sum(sin.(x)), Diagonal(randn(3)))[1][2] == 1

  a = rand(3)
  b = rand(2,2)

  @test gradcheck(x -> sum(sum(diag.((x,) .* a))), b)
  @test gradcheck(x -> sum(sum(diag.(Ref(x) .* a))), b)
  @test gradcheck(x -> sum(sum(diag.([x] .* a))), b)
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
  @test collect(buf) == collect(copy(buf))

  @test gradient([1, 2, 3]) do xs
    b = Zygote.Buffer(xs)
    b .= xs .* 2
    return sum(copy(b))
  end == ([2,2,2],)

  @test gradient(2) do x
    b = Zygote.Buffer([])
    push!(b, x)
    push!(b, 3)
    prod(copy(b))
  end == (3,)
end

@testset "FillArrays" begin
  @test gradcheck(x->sum(Fill(x[], (2, 2))), [0.1])
  @test first(Zygote.gradient(sz->sum(Ones(sz)), 6)) === nothing
  @test first(Zygote.gradient(sz->sum(Zeros(sz)), 6)) === nothing
  @test gradcheck(x->Fill(x[], 5).value, [0.1])
  @test gradcheck(x->FillArrays.getindex_value(Fill(x[], 5)), [0.1])

  @test first(Zygote.pullback(Ones{Float32}, 10)) isa Ones{Float32}
  @test first(Zygote.pullback(Zeros{Float32}, 10)) isa Zeros{Float32}
end

@testset "AbstractArray Addition / Subtraction / Negation" begin
  rng, M, N, P = MersenneTwister(123567), 3, 7, 11
  A, B = randn(rng, M, N, P), randn(rng, M, N, P)
  @test gradtest(+, A, B)
  @test gradtest(-, A, B)
  @test gradtest(-, A)
end

@testset "AbstractFFTs" begin
  findicateMat(i,j,n1,n2) = [(k==i) && (l==j) ? 1.0 : 0.0 for k=1:n1,
                             l=1:n2]
  mirrorIndex(i,N) = i - 2*max(0,i - (N>>1+1))
  for sizeX in [(2,3), (10,10), (13,15)]
    X = randn(sizeX)
    X̂r = rfft(X)
    X̂ = fft(X)
    N = prod(sizeX)
    for i=1:size(X,1), j=1:size(X,2)
      indicateMat = [(k==i) && (l==j) ? 1.0 : 0.0 for k=1:size(X, 1),
                     l=1:size(X,2)]
      # gradient of ifft(fft) must be (approximately) 1 (for various cases)
      @test gradient((X)->real.(ifft(fft(X))[i, j]), X)[1] ≈ indicateMat
      # same for the inverse
      @test gradient((X̂)->real.(fft(ifft(X̂))[i, j]), X̂)[1] ≈ indicateMat
      # same for rfft(irfft)
      @test gradient((X)->real.(irfft(rfft(X), size(X,1)))[i, j], X)[1] ≈ real.(indicateMat)
      # rfft isn't actually surjective, so rffft(irfft) can't really be tested this way.

      # the gradients are actually just evaluating the inverse transform on the
      # indicator matrix
      mirrorI = mirrorIndex(i,sizeX[1])
      FreqIndMat = findicateMat(mirrorI, j, size(X̂r,1), sizeX[2])
      listOfSols = [(fft, bfft(indicateMat), bfft(indicateMat*im),
                     plan_fft(X), i, X),
                    (ifft, 1/N*fft(indicateMat), 1/N*fft(indicateMat*im),
                     plan_fft(X), i, X),
                    (bfft, fft(indicateMat), fft(indicateMat*im), nothing, i,
                     X),
                    (rfft, real.(brfft(FreqIndMat, sizeX[1])),
                     real.(brfft(FreqIndMat*im, sizeX[1])), plan_rfft(X),
                     mirrorI, X),
                    ((K)->(irfft(K,sizeX[1])), 1/N * rfft(indicateMat),
                     zeros(size(X̂r)), plan_rfft(X), i, X̂r)]
      for (trans, solRe, solIm, P, mI, evalX) in listOfSols
        @test gradient((X)->real.(trans(X))[mI, j], evalX)[1] ≈
          solRe
        @test gradient((X)->imag.(trans(X))[mI, j], evalX)[1] ≈
          solIm
        if typeof(P) <:AbstractFFTs.Plan && maximum(trans .== [fft,rfft])
          @test gradient((X)->real.(P * X)[mI, j], evalX)[1] ≈
            solRe
          @test gradient((X)->imag.(P * X)[mI, j], evalX)[1] ≈
            solIm
        elseif typeof(P) <: AbstractFFTs.Plan
          @test gradient((X)->real.(P \ X)[mI, j], evalX)[1] ≈
            solRe
          # for whatever reason the rfft_plan doesn't handle this case well,
          # even though irfft does
          if eltype(evalX) <: Real
            @test gradient((X)->imag.(P \ X)[mI, j], evalX)[1] ≈
              solIm
          end
        end
      end
    end
  end

  x = [-0.353213 -0.789656 -0.270151; -0.95719 -1.27933 0.223982]
  # check ffts for individual dimensions
  for trans in (fft, ifft, bfft)
    @test gradient((x)->sum(abs.(trans(x))), x)[1] ≈
      gradient( (x) -> sum(abs.(trans(trans(x,1),2))),  x)[1]
    # switch sum abs order
    @test gradient((x)->abs(sum((trans(x)))),x)[1] ≈
      gradient( (x) -> abs(sum(trans(trans(x,1),2))),  x)[1]
    # dims parameter for the function
    @test gradient((x, dims)->sum(abs.(trans(x,dims))), x, (1,2))[1] ≈
      gradient( (x) -> sum(abs.(trans(x))), x)[1]
    # (1,2) should be the same as no index
    @test gradient( (x) -> sum(abs.(trans(x,(1,2)))), x)[1] ≈
      gradient( (x) -> sum(abs.(trans(trans(x,1),2))), x)[1]
    @test gradcheck(x->sum(abs.(trans(x))), x)
    @test gradcheck(x->sum(abs.(trans(x, 2))), x)
  end

  @test gradient((x)->sum(abs.(rfft(x))), x)[1] ≈
    gradient( (x) -> sum(abs.(fft(rfft(x,1),2))),  x)[1]
  @test gradient((x, dims)->sum(abs.(rfft(x,dims))), x, (1,2))[1] ≈
    gradient( (x) -> sum(abs.(rfft(x))), x)[1]
end

@testset "FillArrays" begin
  rng, M, N = MersenneTwister(123456), 7, 11
  x, y = randn(rng), randn(rng)
  @test Zygote.gradient(x->sum(Fill(x, N)), x)[1] == N
  @test Zygote.gradient(x->sum(Fill(x, N, 3, 4)), x)[1] == N * 3 * 4
  @test Zygote.gradient((x, y)->sum(Fill(x, N)), x, y) == (N, nothing)

  let
    out, back = Zygote.pullback(sum, Fill(x, N))
    @test back(nothing) isa Nothing
  end

  z = randn(rng, N)
  @test gradtest(x->Fill(first(x), N), [x])
  let
    out, back = Zygote.pullback(x->Fill(x, N), x)
    @test out == Fill(x, N)
    @test first(back(Fill(y, N))) ≈ y * N
  end

  # Test unary broadcasting gradients.
  out, back = Zygote.pullback(x->exp.(x), Fill(x, N))
  @test out isa Fill
  @test out == Fill(exp(x), N)
  @test back(Ones(N))[1] isa Fill
  @test back(Ones(N))[1] == Ones(N) .* exp(x)
  @test back(ones(N))[1] isa Vector
  @test back(ones(N))[1] == ones(N) .* exp(x)
  @test gradtest(x->exp.(Fill(3 * first(x), N)), [x])

  @testset "broadcast +, -, *, /" begin
    for sx in [(M, N), (M, 1), (1, N), (1, 1)]
      for sy in [(M, N), (M, 1), (1, N), (1, 1)]

        #addition, subtraction, multiplication
        for f ∈ (+, -, *)
          @test gradtest((x, y) -> f.(Fill(first(x), sx...), Fill(first(y), sy...)), [x], [y])
          @test gradtest(x -> f.(Fill(first(x), sx...), Ones(sy...)), [x])
          @test gradtest(x -> f.(Fill(first(x), sx...), Zeros(sy...)), [x])
          @test gradtest(y -> f.(Ones(sx...), Fill(first(y), sy...)), [y])
          @test gradtest(y -> f.(Zeros(sx...), Fill(first(y), sy...)), [y])
        end

        #division
        @test gradtest((x, y) -> Fill(first(x), sx...) ./ Fill(first(y), sy...), [x], [y])
        @test gradtest(x -> Fill(first(x), sx...) ./ Ones(sy...), [x])
        @test gradtest(y -> Ones(sx...) ./ Fill(first(y), sy...), [y])
        @test gradtest(y -> Zeros(sx...) ./ Fill(first(y), sy...), [y])
      end
    end
  end
end

@testset "@nograd" begin
  @test gradient(x -> findfirst(ismissing, x), [1, missing]) == (nothing,)
  @test gradient(x -> findlast(ismissing, x), [1, missing]) == (nothing,)
  @test gradient(x -> findall(ismissing, x)[1], [1, missing]) == (nothing,)
  @test gradient(x -> Zygote.ignore(() -> x*x), 1) == (nothing,)
  @test gradient(x -> Zygote.@ignore(x*x), 1) == (nothing,)
  @test gradient(1) do x
    y = Zygote.@ignore x
    x * y
  end == (1,)
end

@testset "fastmath" begin
  @test gradient(x -> begin @fastmath sin(x) end, 1) == gradient(x -> sin(x), 1)
  @test gradient(x -> begin @fastmath tanh(x) end, 1) == gradient(x -> tanh(x), 1)
  @test gradient((x, y) -> begin @fastmath x*y end, 3, 2) == gradient((x, y) -> x*y, 3, 2)
  @test gradient(x -> begin @fastmath real(log(x)) end, 1 + 2im) == gradient(x -> real(log(x)), 1 + 2im)
end

@testset "UniformScaling to Matrix" begin
  @test gradient(x -> (Matrix(I, 2, 2); 1.0), [1.0]) == (nothing,)
  @test gradient(x -> (Matrix(I, (2, 2)); 1.0), [1.0]) == (nothing,)
  @test gradient(x -> (Matrix{Float64}(I, 2, 2); 1.0), [1.0]) == (nothing,)
  @test gradient(x -> (Matrix{Float64}(I, (2, 2)); 1.0), [1.0]) == (nothing,)

  @test gradient(x -> sum(Matrix(x[1]*I, 2, 2)), [1.0]) == ([2.0],)
  @test gradient(x -> sum(Matrix(x[1]*I, (2, 2))), [1.0]) == ([2.0],)
  @test gradient(x -> sum(Matrix{Float64}(x[1]*I, 2, 2)), [1.0]) == ([2.0],)
  @test gradient(x -> sum(Matrix{Float64}(x[1]*I, (2, 2))), [1.0]) == ([2.0],)
end

@testset "random" begin
  @test gradient(x -> rand(), 1) == (nothing,)
  @test gradient(x -> sum(rand(4)), 1) == (nothing,)
  @test gradient(x -> sum(rand(4)), 1) == (nothing,)
  @test gradient(x -> sum(rand(Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(rand(Float32, (1,1))), 1) == (nothing,)
  @test gradient(x -> sum(randn(Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(randn(Float32, (1,1))), 1) == (nothing,)
  @test gradient(x -> sum(randexp(Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(randexp(Float32, (1,1))), 1) == (nothing,)

  @test gradient(x -> rand(), 1) == (nothing,)
  @test gradient(x -> sum(rand(Random.GLOBAL_RNG, 4)), 1) == (nothing,)
  @test gradient(x -> sum(rand(Random.GLOBAL_RNG, 4)), 1) == (nothing,)
  @test gradient(x -> sum(rand(Random.GLOBAL_RNG, Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(rand(Random.GLOBAL_RNG, Float32, (1,1))), 1) == (nothing,)
  @test gradient(x -> sum(randn(Random.GLOBAL_RNG, Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(randn(Random.GLOBAL_RNG, Float32, (1,1))), 1) == (nothing,)
  @test gradient(x -> sum(randexp(Random.GLOBAL_RNG, Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(randexp(Random.GLOBAL_RNG, Float32, (1,1))), 1) == (nothing,)

  @static if VERSION > v"1.3"
    @test gradient(x -> sum(rand(Random.default_rng(), 4)), 1) == (nothing,)
    @test gradient(x -> sum(rand(Random.default_rng(), Float32, 1,1)), 1) == (nothing,)
    @test gradient(x -> sum(rand(Random.default_rng(), Float32, (1,1))), 1) == (nothing,)
    @test gradient(x -> sum(randn(Random.default_rng(), Float32, 1,1)), 1) == (nothing,)
    @test gradient(x -> sum(randn(Random.default_rng(), Float32, (1,1))), 1) == (nothing,)
    @test gradient(x -> sum(randexp(Random.default_rng(), Float32, 1,1)), 1) == (nothing,)
    @test gradient(x -> sum(randexp(Random.default_rng(), Float32, (1,1))), 1) == (nothing,)
  end
end
