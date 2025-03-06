@testitem "gradcheck pt. 4" setup=[GradCheckSetup] begin

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

@testset "AbstractArray Addition / Subtraction / Negation" begin
  rng, M, N, P = MersenneTwister(123567), 3, 7, 11
  A, B = randn(rng, M, N, P), randn(rng, M, N, P)
  @test gradtest(+, A, B)
  @test gradtest(-, A, B)
  @test gradtest(-, A)
end

@testset "AbstractFFTs" begin

  # Eventually these rules and tests will be moved to AbstractFFTs.jl
  # Rules for direct invocation of [i,r,b]fft have already been defined in
  # https://github.com/JuliaMath/AbstractFFTs.jl/pull/58

  # ChainRules involving AbstractFFTs.Plan are not yet part of AbstractFFTs,
  # but there is a WIP PR:
  # https://github.com/JuliaMath/AbstractFFTs.jl/pull/67
  # After the above is merged, this testset can probably be removed entirely.

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
      @test_broken gradient((X)->real.(irfft(rfft(X), size(X,1)))[i, j], X)[1] ≈ real.(indicateMat)
      # rfft isn't actually surjective, so rfft(irfft) can't really be tested this way.

      # the gradients are actually just evaluating the inverse transform on the
      # indicator matrix
      mirrorI = mirrorIndex(i,sizeX[1])
      FreqIndMat = findicateMat(mirrorI, j, size(X̂r,1), sizeX[2])
      listOfSols = [(X -> fft(X, (1, 2)), real(bfft(indicateMat)), real(bfft(indicateMat*im)),
                     plan_fft(X), i, X, true),
                    (K -> ifft(K, (1, 2)), 1/N*real(fft(indicateMat)), 1/N*real(fft(indicateMat*im)),
                     plan_fft(X), i, X, false),
                    (X -> bfft(X, (1, 2)), real(fft(indicateMat)), real(fft(indicateMat*im)), nothing, i,
                     X, false),
      ]
      for (trans, solRe, solIm, P, mI, evalX, fft_or_rfft) in listOfSols
        @test gradient((X)->real.(trans(X))[mI, j], evalX)[1] ≈
          solRe
        @test gradient((X)->imag.(trans(X))[mI, j], evalX)[1] ≈
          solIm
        if typeof(P) <:AbstractFFTs.Plan && fft_or_rfft
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
    @test gradient((x)->sum(abs.(trans(x, (1, 2)))), x)[1] ≈
      gradient( (x) -> sum(abs.(trans(trans(x,1),2))),  x)[1]
    # switch sum abs order
    @test gradient((x)->abs(sum((trans(x)))),x)[1] ≈
      gradient( (x) -> abs(sum(trans(trans(x,1),2))),  x)[1]
    # dims parameter for the function
    @test gradient((x, dims)->sum(abs.(trans(x,dims))), x, (1,2))[1] ≈
      gradient( (x) -> sum(abs.(trans(x, (1, 2)))), x)[1]
    @test gradcheck(x->sum(abs.(trans(x, (1, 2)))), x)
    @test gradcheck(x->sum(abs.(trans(x, 2))), x)
  end

  @test gradient((x)->sum(abs.(rfft(x, (1, 2)))), x)[1] ≈
    gradient( (x) -> sum(abs.(fft(rfft(x,1),2))),  x)[1]
  @test gradient((x, dims)->sum(abs.(rfft(x,dims))), x, (1,2))[1] ≈
      gradient( (x) -> sum(abs.(rfft(x, (1, 2)))), x)[1]

  # Test type stability of fft

  x = randn(Float64,16)
  P = plan_fft(x)
  @test typeof(gradient(x->sum(abs2,ifft(fft(x, 1), 1)),x)[1]) == Array{Float64,1}
  @test typeof(gradient(x->sum(abs2,P\(P*x)),x)[1]) == Array{Float64,1}
  @test typeof(gradient(x->sum(abs2,irfft(rfft(x, 1),16, 1)),x)[1]) == Array{Float64,1}

  x = randn(Float64,16,16)
  @test typeof(gradient(x->sum(abs2,ifft(fft(x,1),1)),x)[1]) == Array{Float64,2}
  @test typeof(gradient(x->sum(abs2,irfft(rfft(x,1),16,1)),x)[1]) == Array{Float64,2}

  x = randn(Float32,16)
  P = plan_fft(x)
  @test typeof(gradient(x->sum(abs2,ifft(fft(x, 1), 1)),x)[1]) == Array{Float32,1}
  @test typeof(gradient(x->sum(abs2,P\(P*x)),x)[1]) == Array{Float32,1}
  @test typeof(gradient(x->sum(abs2,irfft(rfft(x, 1),16, 1)),x)[1]) == Array{Float32,1}

  x = randn(Float32,16,16)
  @test typeof(gradient(x->sum(abs2,ifft(fft(x,1),1)),x)[1]) == Array{Float32,2}
  @test typeof(gradient(x->sum(abs2,irfft(rfft(x,1),16,1)),x)[1]) == Array{Float32,2}
end

@testset "FillArrays" begin

  @test gradcheck(x->sum(Fill(x[], (2, 2))), [0.1])
  @test first(Zygote.gradient(sz->sum(Ones(sz)), 6)) === nothing
  @test first(Zygote.gradient(sz->sum(Zeros(sz)), 6)) === nothing
  @test gradcheck(x->Fill(x[], 5).value, [0.1])
  @test gradcheck(x->FillArrays.getindex_value(Fill(x[], 5)), [0.1])

  @test first(Zygote.pullback(Ones{Float32}, 10)) isa Ones{Float32}
  @test first(Zygote.pullback(Zeros{Float32}, 10)) isa Zeros{Float32}

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
  @test gradient(x->eachindex([10,20,30])[1], 11) == (nothing,)

  #These are defined in ChainRules, we test them here to check we are handling them right
  @test gradient(x -> findfirst(ismissing, x), [1, missing]) == (nothing,)
  @test gradient(x -> findlast(ismissing, x), [1, missing]) == (nothing,)
  @test gradient(x -> findall(ismissing, x)[1], [1, missing]) == (nothing,)
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

  # Check we haven't broken the forward pass:
  @test first(Zygote.pullback(x->Matrix(x*I, 2,2), 8.0)) == Matrix(8.0*I, 2,2)
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

  @test gradient(x -> sum(rand(Random.default_rng(), 4)), 1) == (nothing,)
  @test gradient(x -> sum(rand(Random.default_rng(), Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(rand(Random.default_rng(), Float32, (1,1))), 1) == (nothing,)
  @test gradient(x -> sum(randn(Random.default_rng(), Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(randn(Random.default_rng(), Float32, (1,1))), 1) == (nothing,)
  @test gradient(x -> sum(randexp(Random.default_rng(), Float32, 1,1)), 1) == (nothing,)
  @test gradient(x -> sum(randexp(Random.default_rng(), Float32, (1,1))), 1) == (nothing,)
end

@testset "broadcasted($op, Array, Bool)" for op in (+,-,*)
  @testset "with $bool and sizes $s" for s in ((4,), (2,3)), bool in (true,false)
    r = rand(Int8, s) .+ 0.0
    z = fill(bool, s) .+ 0.0

    @testset "Explicit" begin
      fun(args...) = pullback((x, y) -> sum(op.(x,y)), args...)[1]
      gfun(args...) = gradient((x, y) -> sum(op.(x,y)), args...)

      @test fun(r, z) == fun(r, bool)
      @test gfun(r, bool) == (gfun(r, z)[1], nothing)

      @test fun(z, r) == fun(bool, r)
      @test gfun(bool, r) == (nothing, gfun(z, r)[2])
    end

    @testset "Implicit" begin
      gfun(args...) = gradient(() -> sum(op.(args...)), Params(filter(a->a isa Array, collect(args))  ))

      g = gfun(r, z)
      gres = gfun(r, bool)
      @test gres[r] == g[r]

      g = gfun(z, r)
      gres = gfun(bool, r)
      @test gres[r] == g[r]
    end
  end
end

@testset "norm" begin
    # rrule for norm is defined in ChainRules. These tests just check various norm-related
    # issues are resolved

    # check that type is not unnecessarily promoted
    # https://github.com/FluxML/Zygote.jl/issues/663
    @test gradient(norm, randn(Float32, 2, 2)) isa Tuple{Matrix{Float32}}
    @test gradient(norm, randn(Float32, 2, 2), 3) isa Tuple{Matrix{Float32},Float64}
    @test gradient(norm, randn(Float32, 2, 2), 3f0) isa Tuple{Matrix{Float32},Float32}
    @test gradient(norm, randn(ComplexF32, 2, 2), 3.5f0) isa Tuple{Matrix{ComplexF32},Float32}

    # just check that these do not error
    # https://github.com/FluxML/Zygote.jl/issues/331
    gradient(x->norm(x*[1, 1]), 1.23)
    gradient(x->norm(x*[1 1]), 1.23)
    gradient(x->norm(x*[1im, 1]), 1.23)
    gradient(x->norm(x*[1im 1]), 1.23)
end

@testset "zip & Iterators.product" begin
  # roughly from https://github.com/FluxML/Zygote.jl/issues/221
  d = rand(7)
  @test gradient(rand(11)) do s
    tot = 0
    for (a, b) in zip(s, d)
      tot += 13a + 17b
    end
    tot
  end == ([13, 13, 13, 13, 13, 13, 13, 0, 0, 0, 0],)

  @test gradient([1,2,3,4], [1 2; 3 4]) do x, y # mismatched shapes
    tot = 0
    for (a,b) in zip(x,y)
      tot += a * b
    end
    tot
  end == ([1, 3, 2, 4], [1 3; 2 4]) # Δy is a matrix

  @test gradient((1,2,3), [1 2; 3 4]) do x, y # ... and lengths, and a tuple
    tot = 0
    for (a,b) in zip(x,y)
      tot += a * b
    end
    tot
  end == ((1, 3, 2), [1 3; 2 0]) # map stops early, Δy reshaped to a matrix

  # similar for enumertate -- tests NamedTuple adjoint
  @test gradient([2,3,4]) do x
    tot = 0
    for (i, x) in enumerate(x)
      tot += x^i
    end
    tot
  end == ([1, 6, 3 * 4^2],)

  # and for Iterators.product
  @test gradient([3,4,5], [6,7,8]) do x, y
    tot = 0
    for (a,b) in Iterators.product(x, y)
      tot += a^2 + 10b
    end
    tot
  end == ([18, 24, 30], [30, 30, 30])

  @test gradient([3,4], [1,2,3]) do x, y
    tot = 0
    for ab in Iterators.product(x, y)
      tot += *(ab...)
    end
    tot
  end == ([6,6], [7,7,7])

  # from https://github.com/FluxML/Zygote.jl/pull/785#issuecomment-740562889
  @test gradient(A -> sum([A[i,j] for i in 1:3, j in 1:3]), ones(3,3)) == (ones(3,3),)
end

# https://github.com/FluxML/Zygote.jl/issues/804
@testset "Unused comprehension" begin
    # Comprehension is used.
    io = IOBuffer()
    s = 0.0
    gs = gradient([1.0, 2.0]) do xs
        sum([(print(io, x); s += x; s * x) for x in xs])
    end
    @test String(take!(io)) == "1.02.0"
    @test s == 3.0
    @test gs == ([4.0, 5.0],)

    # Comprehension is not used.
    io = IOBuffer()
    s = 0.0
    gs = gradient([1.0, 2.0]) do xs
        sum([(print(io, x); s += x; s * x) for x in xs])
        0.0
    end
    @test String(take!(io)) == "1.02.0"
    @test s == 3.0
    @test gs == (nothing,)

    # Comprehension is empty and not used.
    io = IOBuffer()
    s = 0.0
    gs = gradient([]) do xs
        [(print(io, x); s += x; s * x) for x in xs]
        0.0
    end
    @test String(take!(io)) == ""
    @test s == 0.0
    @test gs == (nothing,)
end

@testset "Fix1 and Fix2" begin
    @test gradcheck(x -> prod(Base.Fix1(+, 1), x), randn(100))
    @test gradcheck(x -> prod(Base.Fix2(+, 1), x), randn(100))

#= regression tests are not included to reduce CI times
    # check the execution times compared with a closure
    # https://github.com/FluxML/Zygote.jl/issues/957
    x = randn(100)
    tclosure = @belapsed(gradient($(x -> prod(y -> y + 1, x)), $x))
    tfix1 = @belapsed(gradient($(x -> prod(Base.Fix1(+, 1), x)), $x))
    tfix2 = @belapsed(gradient($(x -> prod(Base.Fix2(+, 1), x)), $x))
    @test tfix1 < 2 * tclosure
    @test tfix2 < 2 * tclosure
=#
end

# https://github.com/FluxML/Zygote.jl/issues/996
a = rand(3)
@test Zygote.gradient(x->sum(x .+ rand.()), a) == (ones(3),)

@testset "Zygote 660" begin
  # https://github.com/FluxML/Zygote.jl/pull/660
  function example(x,N)
      ax = axes(x)
      extraAxe = ax[2+N:end]
      filledLoc = fill(1, N)
      return x[:, filledLoc..., extraAxe...]
  end
  y, back = pullback(example, randn(5,3,4,3), 2)
  @test back(zero(y).=1) isa Tuple{Array{Float64,4}, Nothing}
end

@testset "CRC issue 440" begin
  # https://github.com/JuliaDiff/ChainRulesCore.jl/issues/440
  f(x,y) = sum(sum, [[x[i],y[i]] for i=1:length(x)])
  g(x,y) = sum(sum, [(x[i],y[i]) for i=1:length(x)])
  @test gradient(f, rand(3), rand(3)) == ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
  @test gradient(g, rand(3), rand(3)) == ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
end

@testset "CR issue 537" begin
  # https://github.com/JuliaDiff/ChainRules.jl/issues/537
  struct BV{F,T}
    A::F
    α::T
  end
  function Base.:*(c, km::BV)
      new_A = c*km.A
      other_params = getfield.([km], propertynames(km))[2:end]
      BV(new_A, other_params...)
  end
  function (bv::BV)(V_app, ox::Bool; kT::Real = 0.026)
      local exp_arg
      if ox
          exp_arg = (bv.α .* V_app) ./ kT
      else
          exp_arg = -((1 .- bv.α) .* V_app) ./ kT
      end
      bv.A .* exp.(exp_arg)
  end
  Zygote.@adjoint function BV{T,S}(A, α) where {T,S}
    BV(A, α), Δ -> begin
      (Δ.A, Δ.α)
    end
  end
  bv = BV(1.0, 0.1)
  I_vals, V = rand(81), rand(81)

  g2 = gradient(V, bv) do V, bv
    res = fill(bv, length(V))
    r1 = map((m,v) -> m(v, true), res, V)
    r2 = map((m,v) -> m(v, false), res, V)
    sum(r1 .- r2)
  end
  @test size(g2[1]) == size(V)
  @test g2[2] isa NamedTuple
  @test g2[2].A isa Number

  g1 = gradient(bv, V) do bv, V
    res = map(x -> x * bv, V)
    sum(x -> x.A, res)
  end
  @test g1[1] isa NamedTuple
  @test g1[1].A isa Number
  @test size(g1[2]) == size(V)
end

@testset "Zygote #1162" begin
  function zygote1162(as, bs)
      results = [f1162(a, b) for (a, b) in zip(as, bs)]
      return results[2][1] + results[2][2]
  end
  function f1162(a, b)
      return [a^2, b^2]
  end

  as = (1.0, 2.0, 3.0)
  bs = (4.0, 5.0, 6.0)

  g = Zygote.gradient(zygote1162, as, bs)
  @test g == ((nothing, 2*as[2], nothing), (nothing, 2*bs[2], nothing))
end

@testset "Zygote #1184" begin
  n, d = 3, 2
  x = [randn(d) for _ in 1:n]

  f = sin
  g(x) = sum.((f,), x)
  h(x) = sum(abs2, g(x))
  @test gradient(h, x)[1] isa typeof(x)
end

@testset "Zygote #796" begin
    function foo(z::Float64)
        x = 1.0
        y = 1.0 + z
        while abs(x - y) > 1e-6
            y, x = (x + y) / 2, y
        end
        return y
    end

    @test gradcheck(foo ∘ first, [0.0])
    @test gradcheck(foo ∘ first, [2.0])
    @test gradcheck(foo ∘ first, [-1e-5])
    @test gradient(foo, 1024.0)[1] ≈ 2//3
end

@testset "Zygote #1399" begin
  function f1(t)  # this works
    r = 5.0  # (changed to make answers the same)
    sum(@. exp(-t*r))
  end
  @test gradient(f1, [1.0, 0.2])[1] ≈ [-0.03368973499542734, -1.8393972058572117]

  function f2(t)  # this works, too
    sum(@. exp(-t*5))
  end
  @test gradient(f2, [1.0, 0.2])[1] ≈ [-0.03368973499542734, -1.8393972058572117]

  function f3(t)  # but this didn't work
    r = 1.0
    sum(@. exp(-t*r*5))
  end
  @test gradient(f3, [1.0, 0.2])[1] ≈ [-0.03368973499542734, -1.8393972058572117]

  # Also test 4-arg case
  function f4(t)
    r = -0.5
    sum(@. exp(t*r*5*2))
  end
  @test gradient(f4, [1.0, 0.2])[1] ≈ [-0.03368973499542734, -1.8393972058572117]

  # Check that trivial scalar broadcast hasn't gone weird:
  @test gradient(x -> @.(x * x * x), 2.0) == gradient(x -> x * (x * x), 2.0)
  @test gradient(x -> @.(3.0*x*2.0*x), 2.0) == gradient(x -> 6(x^2), 2.0)
end

end
