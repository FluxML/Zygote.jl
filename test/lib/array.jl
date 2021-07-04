using ChainRulesTestUtils
using LinearAlgebra
using Zygote: ZygoteRuleConfig

# issue 897

test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), ones(2); rrule_f=rrule_via_ad, check_inferred=false)
test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), rand(3); rrule_f=rrule_via_ad, check_inferred=false)

using Test, ChainRulesTestUtils, FiniteDifferences, Zygote
@testset "sum(f, x)" begin
    mutable struct F
        s
    end 
    function (f::F)(x) 
        f.s += x
        return f.s
    end
    gfd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1),
        x -> begin  
            f = F(0)
            sum(f, x)
            end
        , [1.0, 2.0, 3.0])[1]

    
    gad = gradient([1.,2.,3.]) do x
                f = F(0.)
                sum(f, x)
            end[1]
            
    @test gad ≈ gfd
end
