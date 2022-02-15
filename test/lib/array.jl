using ChainRulesTestUtils
using LinearAlgebra
using Zygote: ZygoteRuleConfig

# issue 897

test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), ones(2); rrule_f=rrule_via_ad, check_inferred=false)
test_rrule(ZygoteRuleConfig(), x->sum(sin, Diagonal(x)), rand(3); rrule_f=rrule_via_ad, check_inferred=false)

# issue 44179

@test Zygote._restore([nothing, 1, nothing], Val(3)) == (nothing, 1, nothing)