using ChainRulesTestUtils
using LinearAlgebra
using Zygote: ZygoteRuleConfig

# issue 897

test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), ones(2); rrule_f=rrule_via_ad, check_inferred=false)
test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), rand(3); rrule_f=rrule_via_ad, check_inferred=false)

@testset "Zero Dimensional Array Indexing" begin
    x = Array{Float64,0}(undef)
    x[1] = 0.7
    @test Zygote.gradient(x->x[1],x)[1][1] == 1.0
end
