using ChainRulesTestUtils
using LinearAlgebra: Diagonal, Hermitian, LowerTriangular, UpperTriangular
using LinearAlgebra: UnitLowerTriangular, UnitUpperTriangular
using Zygote: ZygoteRuleConfig, _pullback, _reverse

# issue 897

test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), ones(2); rrule_f=rrule_via_ad, check_inferred=false)
test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), rand(3); rrule_f=rrule_via_ad, check_inferred=false)

@testset "adjoints of Iterators.product" begin
    y, back = _pullback(Iterators.product, 1:5, 1:3, 1:2)
    @test back(collect(y)) == (nothing, [6.0, 12.0, 18.0, 24.0, 30.0], [10.0, 20.0, 30.0], [15.0, 30.0])
    @test back([(nothing, j, k) for i in 1:5, j in 1:3, k in 1:2]) == (nothing, nothing, [10.0, 20.0, 30.0], [15.0, 30.0])
    @test back([(i, nothing, k) for i in 1:5, j in 1:3, k in 1:2]) == (nothing, [6.0, 12.0, 18.0, 24.0, 30.0], nothing, [15.0, 30.0])
    @test back([(i, j, nothing) for i in 1:5, j in 1:3, k in 1:2]) == (nothing, [6.0, 12.0, 18.0, 24.0, 30.0], [10.0, 20.0, 30.0], nothing)

    # This was wrong before https://github.com/FluxML/Zygote.jl/pull/1170
    @test gradient(x -> sum([y[2] * y[3] for y in Iterators.product(x, x, x, x)]), [1,2,3,4])[1] ≈ [320, 320, 320, 320]
    @test gradient(x -> sum(y[2] * y[3] for y in Iterators.product(x, x, x, x)), [1,2,3,4])[1] ≈ [320, 320, 320, 320]

    # Numbers failed before https://github.com/FluxML/Zygote.jl/pull/1489
    for p in (1.0, fill(1.0), [1.0])
        @test gradient(p -> sum([x*q for q in p, x in 1:3]), p) == (6p,)
        @test gradient(p -> sum(x*q for (q, x) in Iterators.product(p, 1:3)), p) == (6p,)
    end

    # inference would also fail before #1489
    y, back = _pullback(Iterators.product, 1:5, fill(1))
    @test @inferred back(collect(y)) == (nothing, [1.0, 2.0, 3.0, 4.0, 5.0], fill(5.0))
end

@testset "adjoints of Iterators.zip" begin
    y, back = _pullback(Iterators.zip, 1:5, 1:3, 1:2)
    @test back(collect(y)) == (nothing, [1.0, 2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 2.0])
    @test back([(nothing, j, k) for (i,j,k) in zip(1:5, 1:3, 1:2)]) == (nothing, nothing, [1.0, 2.0, 0.0], [1.0, 2.0])
    @test back([(i, nothing, k) for (i,j,k) in zip(1:5, 1:3, 1:2)]) == (nothing, [1.0, 2.0, 0.0, 0.0, 0.0], nothing, [1.0, 2.0])
    @test back([(i, j, nothing) for (i,j,k) in zip(1:5, 1:3, 1:2)]) == (nothing, [1.0, 2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.0], nothing)


    @test gradient(x -> sum([y[2] * y[3] for y in Iterators.zip(x, x, x, x)]), [1,2,3,4])[1] ≈ [2, 4, 6, 8]
    @test gradient(x -> sum(y[2] * y[3] for y in Iterators.zip(x, x, x, x)), [1,2,3,4])[1] ≈ [2, 4, 6, 8]

    for p in (1.0, fill(1.0), [1.0])
        @test gradient(p_ -> sum(map(prod, Iterators.zip(p_, p))), p) == (p,)
        @test gradient(p_ -> sum(x*q for (q, x) in Iterators.zip(p_, p)), p) == (p,)
    end

    y, back = _pullback(Iterators.zip, 1:5, fill(1))
    @test @inferred back(collect(y)) == (nothing, [1.0, 0.0, 0.0, 0.0, 0.0], fill(1.0))
end

@testset "adjoints of Iterators.take" begin
    y, back = _pullback(Iterators.take, 1:5, 3)
    @test back(collect(y)) == (nothing, [1.0, 2.0, 3.0, 0.0, 0.0], nothing)
    @test back([nothing for i in 1:3]) === nothing

    @test gradient(x -> sum([2y for y in Iterators.take(x, 4)]), [1,2,3,4])[1] ≈ [2, 2, 2, 2]
    @test gradient(x -> sum(2y for y in Iterators.take(x, 4)), [1,2,3,4])[1] ≈ [2, 2, 2, 2]

    for p in (1.0, fill(1.0), [1.0])
        @test gradient(p_ -> sum(map(prod, Iterators.take(p_, 1))), p) == (p,)
        @test gradient(p_ -> sum(x for x in Iterators.take(p_, 1)), p) == (p,)
    end

    y, back = _pullback(Iterators.take, ones(2, 2), 3)
    @test @inferred back(collect(y)) == (nothing, [1.0 1.0; 1.0 0.0], nothing)
end

@testset "collect" begin
    @testset "Dict" begin
        d = Dict(1 => 5, 2 => 6)
        k = 2
        i = findfirst(p -> p[1] == k, collect(d))
        g = gradient(d -> collect(d)[i][2], d)[1]
        @test g isa Dict{Int64, <:Union{Nothing, Int64}}
        @test g[k] == 1

        g = gradient(d -> sum(v^2 for (_,v) in collect(d)), d)[1]
        @test g isa Dict{Int,Int}
        @test g == Dict(1 => 10, 2 => 12)
    end

    @testset "NamedTuple" begin
        t = (a=1, b=2)
        g = gradient(d -> sum(x^2 for x in collect(d)), t)[1]
        @test g === (a = 2.0, b = 4.0)
    end

    @testset "Tuple" begin
        t = (1, 2)
        g = gradient(d -> sum(x^2 for x in collect(d)), t)[1]
        @test g === (2.0, 4.0)
    end

    @testset "Iterators.ProductIterator" begin
        p = Iterators.product(1:3, 1:2)
        g = gradient(p -> sum(prod, collect(p)), p)[1]
        @test g == (iterators=(3ones(3), 6ones(2)),)

        @test gradient(x -> sum(broadcast(prod, Iterators.product(x,x))), ones(4)) == (2*4ones(4),)
        @test gradient(x -> sum(broadcast(prod, Iterators.product(x .^ 2, x))), ones(4)) == (3*4ones(4),)
        @test gradient(x -> sum(broadcast(prod, Iterators.product(x, x .^ 2))), ones(4)) == (3*4ones(4),)
        @test gradient(x -> sum(broadcast(prod, Iterators.product(x .^ 2, x .^ 2))), ones(4)) == (4*4ones(4),)
    end

    @testset "Iterators.Zip" begin
        z = Iterators.zip(1:3, 1:2)
        g = gradient(z -> sum(prod, collect(z)), z)[1]
        @test g == (is=([1.0, 2.0, 0.0], [1.0, 2.0]),)

        @test gradient(x -> sum(broadcast(prod, Iterators.zip(x,x))), ones(4)) == (2ones(4),)
        @test gradient(x -> sum(broadcast(prod, Iterators.zip(x.^2,x))), ones(4)) == (3ones(4),)
        @test gradient(x -> sum(broadcast(prod, Iterators.zip(x,x.^2))), ones(4)) == (3ones(4),)
        @test gradient(x -> sum(broadcast(prod, Iterators.zip(x.^2,x.^2))), ones(4)) == (4ones(4),)
    end


    @testset "Iterators.Take" begin
        z = Iterators.take(1:3, 2)
        g = gradient(z -> sum(collect(z)), z)[1]
        @test g == (xs=[1.0, 1.0, 0.0], n=nothing)

        @test gradient(x -> sum(broadcast(prod, Iterators.take(x,2))), ones(4)) == ([1.0,1.0,0.0,0.0],)
        @test gradient(x -> sum(broadcast(prod, Iterators.take(x.^2,2))), ones(4)) == (2*[1.0,1.0,0.0,0.0],)
    end
end

@testset "dictionary comprehension" begin
    d = Dict(1 => 5, 2 => 6)
    g = gradient(d -> sum([v^2 for (_,v) in d]), d)[1]
    @test g isa Dict{Int, Int}
    @test g == Dict(1 => 10, 2 => 12)


    w = randn(5)
    function f_generator(w)
        d = Dict{Int, Float64}(i => v for (i,v) in enumerate(w))
        sum(v for (_, v) in d)
    end
    @test gradient(f_generator, w)[1] == ones(5)

    function f_comprehension(w)
        d = Dict{Int, Float64}(i => v for (i,v) in enumerate(w))
        sum(v for (_, v) in d)
    end
    @test gradient(f_comprehension, w)[1] == ones(5)

    w = [randn(5); NaN]
    function f_generator_conditional(w)
        d = Dict{Int, Float64}(i => v for (i,v) in enumerate(w) if !isnan(v))
        sum(v for (_, v) in d)
    end
    @test gradient(f_generator_conditional, w)[1] == [ones(5); nothing]

    function f_comprehension_conditional(w)
        d = Dict{Int, Float64}(i => v for (i,v) in enumerate(w) if !isnan(v))
        sum(v for (_, v) in d)
    end
    @test gradient(f_comprehension_conditional, w)[1] == [ones(5); nothing]
end

@testset "_reverse" begin
    m = [1 2 3; 4 5 6; 7 8 9]
    @testset "$wrapper" for wrapper in [
        Hermitian, Symmetric, LowerTriangular, UpperTriangular,
        UnitLowerTriangular, UnitUpperTriangular,
    ]
        M = wrapper(m)
        @test collect(_reverse(M)) == _reverse(collect(M))
    end
end

@testset "rrule for `map`" begin
    @testset "MWE from #1393" begin
        # https://github.com/FluxML/Zygote.jl/issues/1393#issuecomment-1468496804
        struct Foo1393 x::Float64 end
        (f::Foo1393)(x) = f.x * x
        x = randn(5, 5)
        out, pb = Zygote.pullback(x -> map(Foo1393(5.0), x), x)
        @testset "$wrapper" for wrapper in [
            Hermitian, Symmetric, LowerTriangular, UpperTriangular,
            UnitLowerTriangular, UnitUpperTriangular,
        ]
            m = wrapper(rand(5, 5))
            res = only(pb(m))
            @test res == 5m
        end
    end
end

@testset "parent" begin
    @testset "$constructor" for constructor in [LowerTriangular, UpperTriangular]
        x = randn(2, 2)
        y, pb = Zygote.pullback(x) do x
            sum(parent(constructor(2 .* x)))
        end
        @test first(pb(one(y))) ≈ constructor(2 * ones(2, 2))
    end
end
