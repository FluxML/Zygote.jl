using Zygote, Test, ChainRules

const cr_inner_demo_rrule_hitcount = Ref(0)
const cr_inner_demo_pullback_hitcount = Ref(0)
cr_inner_demo(x) = 5x
function ChainRules.rrule(::typeof(cr_inner_demo), x)
    cr_inner_demo_rrule_hitcount[] += 1
    function cr_inner_demo_pullback(Δx)
        cr_inner_demo_pullback_hitcount[] += 1
        return ChainRules.NO_FIELDS, 5.0*Δx
    end
    return cr_inner_demo(x), cr_inner_demo_pullback
end

function cr_outer_demo(x)
    2 + 10cr_inner_demo(x)
end

@testset "ChainRules Integration" begin
    @testset "gradient inner" begin
        cr_inner_demo_rrule_hitcount[] = 0
        cr_inner_demo_pullback_hitcount[] = 0
        @test (5.0,) == gradient(cr_inner_demo, 11)
        @test cr_inner_demo_rrule_hitcount[] == 1
        @test cr_inner_demo_pullback_hitcount[] == 1
    end

    @testset "gradient outer" begin
        cr_inner_demo_rrule_hitcount[] = 0
        cr_inner_demo_pullback_hitcount[] = 0
        @test (50.0,) == gradient(cr_outer_demo, 11)
        @test cr_inner_demo_rrule_hitcount[] == 1
        @test cr_inner_demo_pullback_hitcount[] == 1
    end

    @testset "pullback inner" begin
        cr_inner_demo_rrule_hitcount[] = 0
        cr_inner_demo_pullback_hitcount[] = 0
        y, pb = pullback(cr_inner_demo, 11)
        @test y == 55
        @test cr_inner_demo_rrule_hitcount[] == 1
        @test cr_inner_demo_pullback_hitcount[] == 0
        @test pb(1)==(5.0,);
        @test pb(2)==(10.0,);
        @test pb(3)==(15.0,);
        @test cr_inner_demo_pullback_hitcount[] == 3
        @test cr_inner_demo_rrule_hitcount[] == 1
    end
end
