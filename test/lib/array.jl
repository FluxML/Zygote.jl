using ChainRulesTestUtils
using LinearAlgebra
using Zygote: ZygoteRuleConfig

# issue 897

test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), ones(2); rrule_f=rrule_via_ad, check_inferred=false)
test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), rand(3); rrule_f=rrule_via_ad, check_inferred=false)

@testset "dictionary comprehension" begin
    d = Dict(1 => 5, 2 => 6)
    @test gradient(d -> sum([v^2 for (_,v) in d]), d) == (Dict(1 => 10, 2 => 12),)
end

@testset "collect" begin
    @testset "Dict" begin
        d = Dict(1 => 5, 2 => 6)
        k = 2
        i = findfirst(p -> p[1] == k, collect(d))    
        
        @test gradient(d -> collect(d)[i][2], d)[1][k] == 1
        @test gradient(d -> sum(v^2 for (_,v) in collect(d)), d) == (Dict(1 => 10, 2 => 12),)
    end

    @testset "NamedTuple" begin
        t = (a=1, b=2)
        @test gradient(d -> sum(x^2 for x in collect(d)), t) == ((a = 2, b = 4),)
    end
end

@testset "Dictionary iteration" begin
    # issue https://github.com/FluxML/Zygote.jl/issues/1065
    
    d = Dict(:a => 5, :b => 6)
    @test  gradient(d -> first(d)[2], d) == (Dict(:a => 1, :b => nothing),)
    @test_broken gradient(d -> sum(v for v in values(d)) , d) == (Dict(:a => 1, :b => 1),)
    @test_broken gradient(d -> sum(v for (k,v) in d) , d) == (Dict(:a => 1, :b => 1),)
    
    function f(d)
        s = 0
        for (k,v) in d
            s += v
        end
        s
    end
    
    @test_broken gradient(f, d) == (Dict(:a => 1, :b => 1),)

    function fval(d)
        s = 0
        for v in values(d)
            s += v
        end
        s
    end
    
    @test_broken gradient(fval, d) == (Dict(:a => 1, :b => 1),)
end
