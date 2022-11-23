using ChainRulesTestUtils
using LinearAlgebra
using Zygote: ZygoteRuleConfig, _pullback

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
        @test g isa NamedTuple{(:a, :b), Tuple{Float64, Float64}}
        @test  g == (a = 2, b = 4)
    end
end

@testset "dictionary comprehension" begin
    d = Dict(1 => 5, 2 => 6)
    g = gradient(d -> sum([v^2 for (_,v) in d]), d)[1]
    @test g isa Dict{Int, Int}
    @test  g == Dict(1 => 10, 2 => 12)
end
