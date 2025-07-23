@testitem "gradcheck pt. 1" setup=[GradCheckSetup] begin

using Random
using LinearAlgebra
using Statistics
using SparseArrays
using FillArrays
using Zygote: gradient
using Base.Broadcast: broadcast_shape
using Distributed: pmap, CachingPool, workers

import FiniteDifferences
import LogExpFunctions

@testset "println, show, string, etc" begin
  function foo(x)
    Base.show(x)
    Base.print(x)
    Base.print(stdout, x)
    Base.println(x)
    Base.println(stdout, x)
    Core.show(x)
    Core.print(x)
    Core.println(x)
    return x
  end
  println("The following printout is from testing that `print` doesn't upset gradients:")
  @test gradtest(foo, [5.0])

  function bar(x)
    string(x)
    repr(x)
    return x
  end
  @test gradtest(bar, [5.0])
end

@test gradient(//, 2, 3) === (1//3, -2//9)

@testset "power" begin
  @test gradient(x -> x^2, -2) == (-4,)
  @test gradient(x -> x^10, -1.0) == (-10,) # literal_pow
  _pow = 10
  @test gradient(x -> x^_pow, -1.0) == (-_pow,)
  @test gradient(p -> real(2^p), 2)[1] ≈ 4*log(2)

  @test gradient(xs ->sum(xs .^ 2), [2, -1]) == ([4, -2],)
  @test gradient(xs ->sum(xs .^ 10), [3, -1]) == ([10*3^9, -10],)
  @test gradient(xs ->sum(xs .^ _pow), [4, -1]) == ([_pow*4^9, -10],)

  @test gradient(x -> real((1+3im) * x^2), 5+7im) == (-32 - 44im,)
  @test gradient(p -> real((1+3im) * (5+7im)^p), 2)[1] ≈ real((-234 + 2im)*log(5 - 7im))
  # D[(1+3I)x^p, p] /. {x->5+7I, p->2} // Conjugate
end

@test gradtest((a,b)->sum(reim(acosh(complex(a[1], b[1])))), [-2.0], [1.0])

@test gradtest((x, W, b) -> identity.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> identity.(W*x .+ b), (5,3), (2,5), 2)

@test gradtest((x, W, b) -> tanh.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> tanh.(W*x .+ b), (5,3), (2,5), 2)

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

  # Non-differentiable sum of booleans
  @test gradient(sum, [true, false, true]) == (nothing,)
  @test gradient(x->sum(x .== 0.0), [1.2, 0.2, 0.0, -1.1, 100.0]) == (nothing,)

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
  _, back = Zygote.pullback(x -> x[1] * im, randn(2))
  @test back(1.0)[1] == real([-im, 0]) == [0, 0]

  # _droplike
  @test gradient(x -> sum(inv, x[1, :]'), ones(2, 2)) == ([-1 -1; 0 0],)
  @test gradient(x -> sum(inv, x[1:1, :]'), ones(2, 2)) == ([-1 -1; 0 0],)
  @test gradient(x -> sum(inv, transpose(view(x, 1, :))), ones(2, 2)) == ([-1 -1; 0 0],)

  # https://github.com/FluxML/Zygote.jl/issues/513
  @test gradient(p -> sum(Float32[1, 0] - p), [2, 3]) == ([-1, -1],)
  @test gradient(x -> sum(Float32[1, x] .+ x), 4) == (3.0f0,)

  # Ensure that nothings work with numeric types.
  _, back = Zygote.pullback(getindex, randn(4), [1])
  @test back([nothing]) === nothing

  # Ensure that nothings work with non-numeric types.
  _, back = Zygote.pullback(getindex, [randn(2) for _ in 1:3], [1])
  @test back([nothing]) == nothing
end

@testset "view" begin
  @test gradtest(x -> view(x,:,2,:), (3,4,5))
  @test gradtest(x -> view(x,1:2,3:4), (3,4))
  @test gradtest(x -> view(x,:,[1,2,2]), (3,4))

  # https://github.com/FluxML/Zygote.jl/issues/272
  g(x) = view(x,1:2)[1]
  @test gradient(g, ones(3)) == ([1,0,0],)
end

@testset "eachcol" begin
    @test gradtest(x -> map(sum, eachcol(x)), (3,4))
    @test gradtest(x -> map(sum, eachcol(transpose(x))), (3,4))

    @test gradtest(x -> map(norm, eachcol(x)), (3,4))
    @test gradtest(x -> map(norm, eachrow(x)), (3,4))
    @test gradtest(x -> map(norm, eachslice(x, dims=3)), (3,4,5))

    # some slices may have gradient nothing
    @test gradient(x -> sum(y -> rand()>0.5 ? 0 : first(y), eachcol(x)), rand(3,10))[1] isa Matrix

    # strange errors
    @test_skip gradient(x -> sum(norm, eachcol(x)), [1 2 3; 4 5 6])[1] isa Matrix
    @test_skip gradient(x -> sum(norm, eachcol(x)), rand(3,400))[1] isa Matrix
end

@testset "collect" begin
  @test gradient(x -> sum(inv, collect(x)), (1,2)) === ((-1.0, -1/4),)

  @test gradient(x -> sum(collect(view(x, 1:1))), rand(2)) == ([1,0],)
  @test gradient(x -> sum(inv, collect(view(x', 1,:))), ones(2,2)) == ([-1 0; -1 0],)

  @test gradient(xs -> sum(inv, [x^2 for x in xs]), ones(2)) == ([-2, -2],)

  # adjoint of generators is available and should support generic arrays and iterators
  # generator of array
  @test gradient(p -> sum(collect(p*i for i in [1.0, 2.0, 3.0])), 2.0) == (6.0,)
  # generator of iterator with HasShape
  @test gradient(p -> sum(collect(p*i for (i,) in zip([1.0, 2.0, 3.0]))), 2.0) == (6.0,)
  # generator of iterator with HasLength
  @test gradient(p -> sum(collect(p*i for i in Iterators.take([1.0, 2.0, 3.0], 3))), 2.0) == (6.0,)
  @test gradient(p -> sum(collect(p*i for i in Iterators.take(p*[1.0, 2.0, 3.0], 2))), 2.0) == (12.0,)
  # generator 0-d behavior handled incorrectly
  @test_broken gradient(p -> sum(collect(p*i for i in 1.0)), 2.0) == (1.0,)
  @test gradient(p -> sum(collect(p*i for i in fill(1.0))), 2.0) == (1.0,)

  # adjoints for iterators
  @test gradient(x -> sum(collect(Iterators.take([x*i for i in 1:5], 4))), 1.0) == (10.0,)
  @test gradient(x -> sum(collect(Iterators.take([x*i for i in 1:5], 5))), 1.0) == (15.0,)
  @test_broken gradient(sum∘collect, 1.0) == (1.0,) # broken since no generic adjoint
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

  # fill(struct, ...) handled by ChainRules after
  # https://github.com/FluxML/Zygote.jl/pull/1051
  @test gradient(x -> fill(x, 3)[1][1], (1,2)) === ((1.0, nothing),)
  @test gradient(x -> fill(x, 3)[1].a, (a=1, b=2)) == ((a=1.0, b=nothing),)  # 1 not 1.0
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
@test gradtest(kron, rand(5), rand(3, 2))
@test gradtest(kron, rand(3, 2), rand(5))

for mapfunc in [map,pmap]
  @testset "$mapfunc" begin
    @test gradtest(xs -> sum(mapfunc(x -> x^2, xs)), rand(2,3))
    @test gradtest((xss...) -> sum(mapfunc((xs...) -> sqrt(sum(xs.^2)), xss...)), [rand(5) for _ in 1:6]...)
    function foo(y)
      bar = (x) -> x*y
      sum(mapfunc(bar, 1:5))
    end
    @test gradtest(foo, 3)
    @test gradient(v -> sum([x for x in v]), [1.1,2.2,3.3]) == ([1, 1, 1],)
  end

  @testset "Tuple adjoint" begin
    x = randn(3)
    _, pb = Zygote.pullback(x -> map(abs2, x), x)
    Δy = randn(3)
    @test first(pb((Δy..., ))) ≈ first(pb(Δy))
  end

  @testset "empty tuples" begin
    out, pb = Zygote.pullback(map, -, ())
    @test pb(out) === (nothing, ())

    out, pb = Zygote.pullback(map, +, (), ())
    @test pb(()) === (nothing, (), ())

    function build_foo(z)
      foo(x) = x * z
      return foo
    end
    out, pb = Zygote.pullback(map, build_foo(5.0), ())
    @test pb(()) === (nothing, ())
  end

  @testset "Vector{Nothing} cotangent" begin
    Δ = Vector{Nothing}(nothing, 5)

    # Unary stateless
    out, pb = Zygote.pullback(map, -, randn(5))
    @test pb(Δ)[2] isa Vector{Nothing}

    # Binary stateless
    out, pb = Zygote.pullback(map, +, randn(5), randn(5))
    @test pb(Δ)[2] isa Vector{Nothing}
    @test pb(Δ)[3] isa Vector{Nothing}

    # Stateful
    function build_foo(z)
      foo(x) = x * z
      return foo
    end
    out, pb = Zygote.pullback(map, build_foo(5.0), randn(5))
    @test pb(Δ)[2] isa Vector{Nothing}
  end
end

# Check that map infers correctly. pmap still doesn't infer.
@testset "map inference" begin
  @testset "$name" for (name, f, ȳ, xs) in [
    ("unary empty vector", sin, Float64[], (Float64[], )),
    ("unary vector", sin, randn(3), (randn(3), )),
    ("unary empty tuple", sin, (), ((), )),
    ("unary tuple", sin, (randn(), randn()), ((randn(), randn()), )),
    ("binary empty vector", +, Float64[], (Float64[], Float64[])),
    ("binary vector", +, randn(2), (randn(2), randn(2))),
    ("binary empty tuple", +, (), ((), ())),
    ("binary tuple", +, (randn(), randn()), ((randn(), randn()), (randn(), randn()))),
  ]
    @inferred Zygote._pullback(Zygote.Context(), map, f, xs...)
    y, pb = Zygote._pullback(Zygote.Context(), map, f, xs...)
    @inferred pb(ȳ)
  end
end

@testset "map and tuples" begin
  # arrays of tuples, ChainRules's Tangent should not escape
  @test gradient(x -> sum(map(first, x)), [(1,2), (3,4)]) == ([(1.0, nothing), (1.0, nothing)],)
  @test gradient(x -> sum(first, x), [(1,2), (3,4)]) == ([(1.0, nothing), (1.0, nothing)],)

  @test gradient(x -> map(+, x, (1,2,3))[1], (4,5,6)) == ((1.0, nothing, nothing),)
  @test gradient(x -> map(+, x, [1,2,3])[1], (4,5,6)) == ((1.0, 0.0, 0.0),)
  @test gradient(x -> map(+, x, (1,2,3))[1], [4,5,6]) == ([1,0,0],)

  # mismatched lengths, should zip
  @test gradient(x -> map(+, x, [1,2,3,99])[1], (4,5,6)) == ((1.0, 0.0, 0.0),)
  @test gradient(x -> map(+, x, [1,2,3])[1], (4,5,6,99)) == ((1.0, 0.0, 0.0, nothing),)
end

@testset "Alternative Pmap Dispatch" begin
    cache_and_map(f,xs...) = pmap(f, CachingPool(workers()), xs...; batch_size = 1)
    @test gradtest(xs -> sum(cache_and_map(x -> x^2, xs)), rand(2,3))
    @test gradtest((xss...) -> sum(cache_and_map((xs...) -> sqrt(sum(xs.^2)), xss...)), [rand(5) for _ in 1:6]...)
    function foo(y)
      bar = (x) -> x*y
      sum(cache_and_map(bar, 1:5))
    end
    @test gradtest(foo, 3)
    @test gradient(v -> sum([x for x in v]), [1.1,2.2,3.3]) == ([1, 1, 1],)
end

@testset "Stateful Map" begin
  s = 0
  f(x) = (s += x)
  @test_broken gradient(x -> sum(f.(x)), 1:10) == (10:-1:1,)
  s = 0
  @test gradient(x -> sum(map(f, x)), 1:10) == (10:-1:1,)
end

@testset "vararg map" begin
  # early stop
    # In Julia 1.4 and earlier, map(*,rand(5),[1,2,3]) is a DimensionMismatch
  @test gradient(x -> sum(map(*,x,[1,2,3])), rand(5)) == ([1,2,3,0,0],)
  @test gradient(x -> sum(map(*,x,(1,2,3))), rand(5)) == ([1,2,3,0,0],)
  @test gradient(x -> sum(map(*,x,[1,2,3])), Tuple(rand(5))) == ((1.0, 2.0, 3.0, nothing, nothing),)

  # mixed shapes
  @test gradient((x,y) -> sum(map(*,x,y)), [1,2,3,4], [1 2; 3 4]) == ([1,3,2,4], [1 3; 2 4])
  @test gradient((x,y) -> sum(map(*,x,y)), [1,2,3], [1 2; 3 4]) == ([1,3,2], [1 3; 2 0])
  @test gradient((x,y) -> sum(map(*,x,y)), (1,2,3), [1 2; 3 4]) == ((1,3,2), [1 3; 2 0])
  @test gradient((x,y) -> sum(map(*,x,y)), [1,2,3,4,5], [1 2; 3 4]) == ([1,3,2,4,0], [1 3; 2 4])
  @test gradient((x,y) -> sum(map(*,x,y)), (1,2,3,4,5), [1 2; 3 4]) == ((1,3,2,4,nothing), [1 3; 2 4])
end

@testset "map: issye 1374" begin
  # The code to reverse iteration in map was very sloppy, could reverse fwd & not reverse, wtf.
  # https://github.com/FluxML/Zygote.jl/issues/1374
  struct Affine1374
    W
    b
  end
  (m::Affine1374)(x) = [sum(x.*r) for r in eachrow(m.W)] + m.b
  m = Affine1374(zeros(3,3), zeros(3,1))
  x = [ 1.0,  2.0,  3.0]
  y = [-1.0, -2.0, -3.0]
  l1374(y,ŷ) = sum(abs2.(y - ŷ))/2
  grads = gradient(m -> l1374(y,m(x)), m)
  @test grads[1].W ≈ [1 2 3; 2 4 6; 3 6 9]
end

@testset "sort" begin
  @test gradtest(sort, 5)
  correct = [
      [2,3,1],
      [1, 2, 3],
      [1,2,3],
      [2,1,3],
      [1,3,2],
      [3,2,1]
  ]
  for i = 1:3
    @test gradient(v->sort(v)[i], [3.,1,2])[1][correct[1][i]] == 1
    @test gradient(v->sort(v)[i], [1.,2,3])[1][correct[2][i]] == 1
    @test gradient(v->sort(v,by=x->x%10)[i], [11,2,99])[1][correct[3][i]] == 1
    @test gradient(v->sort(v,by=x->x%10)[i], [2,11,99])[1][correct[4][i]] == 1
    @test gradient(v->sort(v,rev=true)[i], [3.,1,2])[1][correct[5][i]] == 1
    @test gradient(v->sort(v,rev=true)[i], [1.,2,3])[1][correct[6][i]] == 1
  end
end

@testset "filter" begin
  @test gradtest(xs -> filter(x -> x > 0.5, xs), 20)

  @test gradient(x -> sum(log, filter(iseven, x)), 1:10) ==
    (map(x -> iseven(x) ? 1/x : 0, 1:10),)
  @test gradient(x -> sum(abs2, im .+ filter(iseven, x)), 1:10) ==
    (map(x -> iseven(x) ? 2x : 0, 1:10),)
    # (map(x -> iseven(x) ? 2x+2im : 0, 1:10),)
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

  # issue 1224, second order
  f1244(w, x) = sum(maximum((w * x).^2, dims=1))
  g1244(w, x) = sum(gradient(f1244, w, x)[2].^2)
  h1244(w, x) = gradient(g1244, w, x)[2]
  @test h1244([1 2 3; 4 5 6.0], [7,8,9.0]) ≈ [300608, 375760, 450912]
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
      ȳ = randn(BT, N, N)
      @test back(ȳ)[1] == back2(ȳ)[1]
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
end

end
