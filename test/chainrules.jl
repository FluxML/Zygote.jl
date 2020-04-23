using Zygote, Test, ChainRules


@testset "ChainRules Integration" begin
    @testset "basic" begin
        cr_inner_demo_rrule_hitcount = Ref(0)
        cr_inner_demo_pullback_hitcount = Ref(0)
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

    @testset "Multiple output single input" begin
        simo_rrule_hitcount = Ref(0)
        simo_pullback_hitcount = Ref(0)
        simo(x) = (5x, 7x)
        function ChainRules.rrule(::typeof(simo), x)
            simo_rrule_hitcount[] += 1
            function simo_pullback(Δa, Δb)
                simo_pullback_hitcount[] += 1
                return ChainRules.NO_FIELDS, 5*Δa + 7*Δb
            end
            return simo(x), simo_pullback
        end

        simo_outer(x) = sum(simo(x))

        @assert simo_rrule_hitcount[] == 0
        @assert simo_pullback_hitcount[] == 0
        @test (12,) == Zygote.gradient(simo_outer, π)
        @test simo_rrule_hitcount[] == 1
        @test simo_pullback_hitcount[] == 1
    end

    @testset "multiple input, Single output" begin
        miso_rrule_hitcount = Ref(0)
        miso_pullback_hitcount = Ref(0)
        miso(a, b) = 5a + 7b
        function ChainRules.rrule(::typeof(miso), a, b)
            miso_rrule_hitcount[] += 1
            function miso_pullback(Δy)
                miso_pullback_hitcount[] += 1
                return ChainRules.NO_FIELDS, 5Δy , 7Δy
            end
            return miso(a, b), miso_pullback
        end

        miso_outer(x) = miso(100x, 10x)

        @assert miso_rrule_hitcount[] == 0
        @assert miso_pullback_hitcount[] == 0
        @test (570,) == Zygote.gradient(miso_outer, π)
        @test miso_rrule_hitcount[] == 1
        @test miso_pullback_hitcount[] == 1
    end

    @testset "multiple input multiple output" begin
        mimo_rrule_hitcount = Ref(0)
        mimo_pullback_hitcount = Ref(0)
        mimo(a, b) = (5a + 7b, 100a, 10b)
        function ChainRules.rrule(::typeof(mimo), a, b)
            mimo_rrule_hitcount[] += 1
            function mimo_pullback(Δx, Δy, Δz)
                mimo_pullback_hitcount[] += 1
                return ChainRules.NO_FIELDS, 5Δx + 100Δy , 7Δx + 10Δz
            end
            return mimo(a, b), mimo_pullback
        end

        @assert mimo_rrule_hitcount[] == 0
        @assert mimo_pullback_hitcount[] == 0
        _, pb = Zygote.pullback(mimo, π, 2π)
        @test (105, 17) == pb((1, 1, 1))
        @test mimo_rrule_hitcount[] == 1
        @test mimo_pullback_hitcount[] == 1

        mimo_outer(x) = sum(mimo(x, x))

        mimo_rrule_hitcount[] = 0
        mimo_pullback_hitcount[] = 0
        @test (122,) == gradient(mimo_outer, π)
        @test mimo_rrule_hitcount[] == 1
        @test mimo_pullback_hitcount[] == 1
    end
end

@test_broken gradient(2.0) do x
  @fastmath x^2.0
end == (4.0,)
