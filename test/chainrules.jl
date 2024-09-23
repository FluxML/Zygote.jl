using ChainRulesCore, ChainRulesTestUtils, Zygote
using Zygote: ZygoteRuleConfig

@testset "ChainRules integration" begin
    @testset "ChainRules basics" begin
        cr_inner_demo_rrule_hitcount = Ref(0)
        cr_inner_demo_pullback_hitcount = Ref(0)
        cr_inner_demo(x) = 5x
        function ChainRulesCore.rrule(::typeof(cr_inner_demo), x)
            cr_inner_demo_rrule_hitcount[] += 1
            function cr_inner_demo_pullback(Δx)
                cr_inner_demo_pullback_hitcount[] += 1
                return NoTangent(), 5.0*Δx
            end
            return cr_inner_demo(x), cr_inner_demo_pullback
        end

        function cr_outer_demo(x)
            2 + 10cr_inner_demo(x)
        end

        #

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
        function ChainRulesCore.rrule(::typeof(simo), x)
            simo_rrule_hitcount[] += 1
            function simo_pullback((Δa, Δb))
                simo_pullback_hitcount[] += 1
                return NoTangent(), 5*Δa + 7*Δb
            end
            return simo(x), simo_pullback
        end
        
        simo_outer(x) = sum(simo(x))

        simo_rrule_hitcount[] = 0
        simo_pullback_hitcount[] = 0
        @test (12,) == Zygote.gradient(simo_outer, π)
        @test simo_rrule_hitcount[] == 1
        @test simo_pullback_hitcount[] == 1
    end

    @testset "multiple input, Single output" begin
        miso_rrule_hitcount = Ref(0)
        miso_pullback_hitcount = Ref(0)
        miso(a, b) = 5a + 7b
        function ChainRulesCore.rrule(::typeof(miso), a, b)
            miso_rrule_hitcount[] += 1
            function miso_pullback(Δy)
                miso_pullback_hitcount[] += 1
                return NoTangent(), 5Δy , 7Δy
            end
            return miso(a, b), miso_pullback
        end
        

        miso_outer(x) = miso(100x, 10x)

        miso_rrule_hitcount[] = 0
        miso_pullback_hitcount[] = 0
        @test (570,) == Zygote.gradient(miso_outer, π)
        @test miso_rrule_hitcount[] == 1
        @test miso_pullback_hitcount[] == 1
    end

    @testset "multiple input multiple output" begin
        mimo_rrule_hitcount = Ref(0)
        mimo_pullback_hitcount = Ref(0)
        mimo(a, b) = (5a + 7b, 100a, 10b)
        function ChainRulesCore.rrule(::typeof(mimo), a, b)
            mimo_rrule_hitcount[] += 1
            function mimo_pullback((Δx, Δy, Δz))
                mimo_pullback_hitcount[] += 1
                return NoTangent(), 5Δx + 100Δy , 7Δx + 10Δz
            end
            return mimo(a, b), mimo_pullback
        end

        mimo_rrule_hitcount[] = 0
        mimo_pullback_hitcount[] = 0
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
        function ChainRulesCore.rrule(::typeof(not_diff_eg), x, i)
            function not_diff_eg_pullback(Δ)
                return NoTangent(), ZeroTangent(), NoTangent()
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
            @test begin
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
        function ChainRulesCore.rrule(::typeof(kwfoo), x; k=10)
            kwfoo_rrule_hitcount[] += 1
            function kwfoo_pullback(Δy)
                kwfoo_pullback_hitcount[] += 1
                return NoTangent(), Δy
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
        function ChainRulesCore.rrule(::typeof(not_diff_kw_eg), x, i; kwargs...)
            function not_diff_kw_eg_pullback(Δ)
                return NoTangent(), ZeroTangent(), NoTangent()
            end
            return not_diff_kw_eg(x, i; kwargs...), not_diff_kw_eg_pullback
        end
        

        @test (nothing,) == Zygote.gradient(x->not_diff_kw_eg(x, 2), 10.4)
        @test (nothing,) == Zygote.gradient(x->not_diff_kw_eg(x, 2; kw=2.0), 10.4)
    end

    @testset "Type only rrule" begin
        struct StructForTestingTypeOnlyRRules{T}
            x::T
        end
        StructForTestingTypeOnlyRRules() = StructForTestingTypeOnlyRRules(1.0)
        
        function ChainRulesCore.rrule(P::Type{<:StructForTestingTypeOnlyRRules})
            # notice here we mess with the primal doing 2.0 rather than 1.0, this is for testing purposes
            # and also because apparently people actually want to do this. Weird, but 🤷
            # https://github.com/SciML/SciMLBase.jl/issues/69#issuecomment-865639754
            P(2.0),  _ -> (NoTangent(),)
        end

        @assert StructForTestingTypeOnlyRRules().x == 1.0
        aug_primal_val, _ = Zygote.pullback(x->StructForTestingTypeOnlyRRules(), 1.2)
        @test aug_primal_val.x == 2.0
    end

    @testset "@opt_out" begin
        oout_id(x) = x
        oout_id_rrule_hitcount = Ref(0)
        function ChainRulesCore.rrule(::typeof(oout_id), x::Any)
            oout_id_rrule_hitcount[] += 1
            oout_id_pullback(ȳ) = (NoTangent(), ȳ)
            return oout_id(x), oout_id_pullback
        end

        @opt_out ChainRulesCore.rrule(::typeof(oout_id), x::AbstractArray)

        # Hit one we haven't opted out
        oout_id_rrule_hitcount[] = 0
        oout_id_outer(x) = sum(oout_id(x))
        @test (1.0,) == Zygote.gradient(oout_id_outer, π)
        @test oout_id_rrule_hitcount[] == 1

        # make sure don't hit the one we have opted out
        oout_id_rrule_hitcount[] = 0
        @test ([1.0],) == Zygote.gradient(oout_id_outer, [π])
        @test oout_id_rrule_hitcount[] == 0

        # Now try opting out After we have already used it 
        @opt_out ChainRulesCore.rrule(::typeof(oout_id), x::Real)
        oout_id_rrule_hitcount[] = 0
        @test (1.0,) == Zygote.gradient(oout_id_outer, π)
        @test oout_id_rrule_hitcount[] == 0
    end

    # issue #1204
    @testset "NotImplemented" begin
        f_notimplemented(x) = x
        @scalar_rule f_notimplemented(x) @not_implemented("not implemented :(")
        @test Zygote.gradient(f_notimplemented, 0.1) === (nothing,)
        @test Zygote.gradient(x -> f_notimplemented(x[1]), 0.1) === (nothing,)
        if isdefined(Base, :only)
            @test Zygote.gradient(x -> f_notimplemented(only(x)), (0.1,)) === (nothing,)
            @test Zygote.gradient(x -> f_notimplemented(only(x)), [0.1]) === (nothing,)
        end
    end

    # https://github.com/FluxML/Zygote.jl/issues/1234
    @testset "rrule lookup ambiguities" begin
      @testset "unconfigured" begin
        f_ambig(x, y) = x + y
        ChainRulesCore.rrule(::typeof(f_ambig), x::Int, y) = x + y, _ -> (0, 0)
        ChainRulesCore.rrule(::typeof(f_ambig), x, y::Int) = x + y, _ -> (0, 0)

        @test_throws MethodError pullback(f_ambig, 1, 2)
      end
      @testset "configured" begin
        h_ambig(x, y) = x + y
        ChainRulesCore.rrule(::ZygoteRuleConfig, ::typeof(h_ambig), x, y) = x + y, _ -> (0, 0)
        ChainRulesCore.rrule(::RuleConfig, ::typeof(h_ambig), x::Int, y::Int) = x + y, _ -> (0, 0)

        @test_throws MethodError pullback(h_ambig, 1, 2)
      end
    end
end

@testset "ChainRulesCore.rrule_via_ad" begin
    @testset "basic" begin
        # Not marked as tests since perhaps ZeroTangent would be better.
        rrule_via_ad(ZygoteRuleConfig(), round, 2.2)[2](1) == (NoTangent(), 0.0)
        # But test_rrule is happy:
        test_rrule(ZygoteRuleConfig(), round, 2.2; rrule_f=rrule_via_ad)

        test_rrule(ZygoteRuleConfig(), vcat, rand(3), rand(4); rrule_f=rrule_via_ad)
        test_rrule(ZygoteRuleConfig(), getindex, rand(5), 3; rrule_f=rrule_via_ad)
    end

    @testset "kwargs" begin
        test_rrule(
            ZygoteRuleConfig(), sum, [1.0 2; 3 4];
            rrule_f=rrule_via_ad, check_inferred=false, fkwargs=(;dims=1)
        )
    end

    @testset "struct" begin
        struct Foo
            x
            y
        end
        makefoo(a, b) = Foo(a, b)
        sumfoo(foo) = foo.x + foo.y


        test_rrule(
            ZygoteRuleConfig(), sumfoo, Foo(1.2, 2.3); rrule_f=rrule_via_ad, check_inferred=false
        )
        test_rrule(
            ZygoteRuleConfig(), makefoo, 1.0, 2.0;
            rrule_f=rrule_via_ad, check_inferred=false
        )
    end

    @testset "tuples/namedtuples" begin
        my_tuple(a, b, c) = (a+b, b+c)
        my_namedtuple(a, b, c) = (a=a, b=b, c=0.0)

        test_rrule(
            ZygoteRuleConfig(), my_tuple, 1., 2., 3.; rrule_f=rrule_via_ad
        )
        test_rrule(
            ZygoteRuleConfig(), my_namedtuple, 1., 2., 3.; rrule_f=rrule_via_ad
        )
        test_rrule(
            ZygoteRuleConfig(), my_namedtuple, 1., (2.0, 2.4), 3.; rrule_f=rrule_via_ad
        )
        test_rrule(
            ZygoteRuleConfig(), sum, (1.0, 2.0, 3.0); rrule_f=rrule_via_ad, check_inferred=false
        )
        test_rrule(
            ZygoteRuleConfig(), sum, (a=1.0, b=2.0); rrule_f=rrule_via_ad, check_inferred=false
        )
        # There is at present no rrule for sum(::Tuple), so those are testing zygote directly.
    end

    @testset "arrays" begin
        nada(x, y) = 1.0
        test_rrule(ZygoteRuleConfig(), nada, rand(3), rand(2,3); rrule_f=rrule_via_ad)
        test_rrule(ZygoteRuleConfig(), +, rand(3), rand(3); rrule_f=rrule_via_ad)
        test_rrule(ZygoteRuleConfig(), *, rand(1, 3), rand(3); rrule_f=rrule_via_ad)
    end

    @testset "rules which call rrule_via_ad" begin
        # since cbrt has a rule, this will test the shortcut:
        test_rrule(ZygoteRuleConfig(), sum, cbrt, randn(5))
        test_rrule(ZygoteRuleConfig(), sum, cbrt, randn(5); rrule_f=rrule_via_ad)

        # but x -> cbrt(x) has no rule, so will be done by Zygote
        # increased tolerances because these are occasionally flaky at rtol=1e-9
        test_rrule(ZygoteRuleConfig(), sum, x -> cbrt(x), randn(5); rtol=1e-8)
        test_rrule(ZygoteRuleConfig(), sum, x -> cbrt(x), randn(5); rtol=1e-8,
                   rrule_f=rrule_via_ad)
    end

    # See https://github.com/FluxML/Zygote.jl/issues/1078
    @testset "ProjectTo{AbstractArray}(::Tangent{Any})" begin
        X = UpperHessenberg(randn(5, 5))
        dX = Tangent{Any}(element=randn(5, 5))
        @test ProjectTo(X)(dX) === dX
    end
end

@testset "FastMath support" begin
    @test gradient(2.0) do x
      @fastmath x^2.0
    end == (4.0,)

    @test gradient(2) do x
      @fastmath log(x)
    end == (1/2,)
end

@testset "zygote2differential inference" begin
    @test @inferred(Zygote.z2d(1.0, 2.0)) isa Real
    @test @inferred(Zygote.z2d([1,2,3], [4,5,6])) isa Vector
    @test @inferred(Zygote.z2d((1, 2.0, 3+4im), (5, 6.0, 7+8im))) isa Tangent{<:Tuple}

    # Below Julia 1.7, these need a @generated version to be inferred:
    @test @inferred(Zygote.z2d((re=1,), 3.0+im)) isa Tangent{ComplexF64}
    @test @inferred(Zygote.z2d((re=1, im=nothing), 3.0+im)) isa Tangent{ComplexF64}

    # collapse nothings
    @test @inferred(Zygote.z2d((nothing,), (1,))) === NoTangent()
    @test @inferred(Zygote.z2d((nothing, nothing), (1,2))) === NoTangent()

    # To test the generic case, we need a struct within a struct. 
    nested = Tangent{Base.RefValue{ComplexF64}}(; x=Tangent{ComplexF64}(; re=1, im=NoTangent()),)
    if VERSION > v"1.7-"
        @test @inferred(Zygote.z2d((; x=(; re=1)), Ref(3.0+im))) == nested
        @test @inferred(Zygote.z2d((; x=(; re=nothing)), Ref(3.0+im))) === NoTangent()
    else
        @test Zygote.z2d((; x=(; re=1)), Ref(3.0+im)) == nested
        @test Zygote.z2d((; x=(; re=nothing)), Ref(3.0+im)) === NoTangent()
    end

    x = (c = (a = randn(3,3), b = rand(3)), d = randn(5))
    z2d_compiled = Zygote.z2d(x, x)
    z2d_fallback = Zygote._z2d_struct_fallback(x, x)
    @test z2d_compiled.d === z2d_fallback.d
    @test z2d_compiled.c.a === z2d_fallback.c.a
    @test z2d_compiled.c.b === z2d_fallback.c.b
end

@testset "ChainRules translation" begin
    @test Zygote.wrap_chainrules_input(nothing) == ZeroTangent()
    @test Zygote.wrap_chainrules_input((nothing,)) == ZeroTangent()
    @test Zygote.wrap_chainrules_input([nothing]) == ZeroTangent()
    @test Zygote.wrap_chainrules_input(((1.0, 2.0), 3.0)) == Tangent{Any}(Tangent{Any}(1.0, 2.0), 3.0)
    @test Zygote.wrap_chainrules_input((; a = 1.0, b = 2.0)) == Tangent{Any}(a = 1.0, b = 2.0)
    @test Zygote.wrap_chainrules_input(Ref(1)) == 1
    @test Zygote.wrap_chainrules_input([2.0; 4.0]) == [2.0; 4.0]
    @test Zygote.wrap_chainrules_input([[2.0; 4.0], [1.0; 3.0]]) == [[2.0; 4.0], [1.0; 3.0]]
    @test Zygote.wrap_chainrules_input([nothing; 4.0]) == [0.0; 4.0] # ChainRules uses the numeric zero where possible
end
