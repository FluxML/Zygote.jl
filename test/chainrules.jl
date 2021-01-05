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
            function simo_pullback((Δa, Δb))
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
            function mimo_pullback((Δx, Δy, Δz))
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

    @testset "all AbstractZero partials" begin
        # while ChainRules always has a partial for every input, Zygote combined them all
        # to a single `nothing` if they are all zero-like.

        not_diff_eg(x, i) = [10, 20][i]
        function ChainRules.rrule(::typeof(not_diff_eg), x, i)
            function not_diff_eg_pullback(Δ)
                return ChainRules.NO_FIELDS, ChainRules.Zero(), ChainRules.DoesNotExist()
            end
            return not_diff_eg(x, i), not_diff_eg_pullback
        end

        _, pb = Zygote.pullback(not_diff_eg, 10.4, 2)
        @test pb(1.2) === nothing
    end

    @testset "nested AD hitting identity(::Tuple) pullback" begin
        # This is is  a particularly fiddly case.
        # Its kind of a simplified version of `sin'''(0.5)` but different in some places.

        f(x) = tuple(x, 2x, 3x)

        function g(y)
            a1, pb1 = Zygote.pullback(f, π)
            pb1((y,0,0))
        end

        @test (1,) == g(1)

        function h(n)
            a2, pb2 = Zygote.pullback(g, 1)
            pb2(n)
        end

        @test (1,) == h(1)

        if VERSION >= v"1.6-"
            @test_broken begin
                a3, pb3 = Zygote.pullback(h, 1)
                ((1,),) == pb3(1)
            end
        else
            a3, pb3 = Zygote.pullback(h, 1)
            @test ((1,),) == pb3(1)
        end
    end

    @testset "kwargs" begin
        kwfoo_rrule_hitcount = Ref(0)
        kwfoo_pullback_hitcount = Ref(0)
        kwfoo(x; k=10) = x + k
        function ChainRules.rrule(::typeof(kwfoo), x; k=10)
            kwfoo_rrule_hitcount[] += 1
            function kwfoo_pullback(Δy)
                kwfoo_pullback_hitcount[] += 1
                return ChainRules.NO_FIELDS, Δy
            end
            return kwfoo(x; k=k), kwfoo_pullback
        end

        kwfoo_outer_unused(x) = kwfoo(x)
        kwfoo_outer_used(x) = kwfoo(x; k=-15)

        @testset "$outer" for outer in (kwfoo_outer_used, kwfoo_outer_unused)
            kwfoo_rrule_hitcount[] = 0
            kwfoo_pullback_hitcount[] = 0
            @test (1,) == Zygote.gradient(outer, π)
            @test kwfoo_rrule_hitcount[] == 1
            @test kwfoo_pullback_hitcount[] == 1
        end
    end


    @testset "kwarg, with all AbstractZero partials" begin
        # while ChainRules always has a partial for every input, Zygote combined them all
        # to a single `nothing` if they are all zero-like.

        not_diff_kw_eg(x, i; kw=1.0) = [10, 20][i]
        function ChainRules.rrule(::typeof(not_diff_kw_eg), x, i; kwargs...)
            function not_diff_kw_eg_pullback(Δ)
                return ChainRules.NO_FIELDS, ChainRules.Zero(), ChainRules.DoesNotExist()
            end
            return not_diff_kw_eg(x, i; kwargs...), not_diff_kw_eg_pullback
        end

        @test (nothing,) == Zygote.gradient(x->not_diff_kw_eg(x, 2), 10.4)
        @test (nothing,) == Zygote.gradient(x->not_diff_kw_eg(x, 2; kw=2.0), 10.4)
    end
end

@testset "FastMath support" begin
    @test gradient(2.0) do x
      @fastmath x^2.0
    end == (4.0,)
end
