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
end