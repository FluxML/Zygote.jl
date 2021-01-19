using Zygote, Test
using Zygote: gradient
using CUDA: has_cuda

using Zygote
@testset "gradient checkpointing" begin

    @testset "checkpointed does not change pullback value" begin
        for setup in [
                (f=identity, args = (1.0,), dy=1.0),
                (f=max, args = (1.0,2, 3), dy=1.0),
                (f=sum, args = (cos, [1.0, 2.0],), dy=1.0),
                (f=*, args = (randn(2,2),randn(2,2)), dy=randn(2,2)),
            ]
            y_ref, pb_ref = Zygote.pullback(setup.f, setup.args...)
            y_cp, pb_cp = Zygote.pullback(Zygote.checkpointed, setup.f, setup.args...)
            @test y_cp == y_ref
            @test pb_cp(setup.dy) == (nothing, pb_ref(setup.dy)...)
        end
    end

    mutable struct CountCalls
        f
        ncalls
    end
    CountCalls(f=identity) = CountCalls(f, 0)
    function (o::CountCalls)(x...)
        o.ncalls += 1
        o.f(x...)
    end

    f = CountCalls()
    g = CountCalls()
    h = CountCalls()

    y, pb = Zygote.pullback(h ∘ g ∘ f, 4.0)
    @test y === 4.0
    @test f.ncalls === g.ncalls === h.ncalls === 1
    @test pb(1.0) === (1.0,)
    @test f.ncalls === g.ncalls === h.ncalls === 1

    f = CountCalls()
    g = CountCalls()
    h = CountCalls()
    y,pb = Zygote.pullback(Zygote.checkpointed, h∘g∘f, 4.0)
    @test f.ncalls === g.ncalls === h.ncalls === 1
    @test pb(1.0) === (nothing, 1.0,)
    @test f.ncalls === g.ncalls === h.ncalls === 2
    @test pb(1.0) === (nothing, 1.0,)
    @test f.ncalls === g.ncalls === h.ncalls === 3

    f = CountCalls()
    g = CountCalls()
    h = CountCalls()
    function only_some_checkpointed(x)
        x2 = f(x)
        Zygote.checkpointed(h∘g, x2)
    end
    y,pb = Zygote.pullback(only_some_checkpointed, 4.0)
    @test f.ncalls === g.ncalls === h.ncalls === 1
    @test pb(1.0) === (1.0,)
    @test g.ncalls === h.ncalls === 2
    @test f.ncalls === 1

    @testset "nested checkpointing" begin
        f1 = CountCalls(sin)
        f2 = CountCalls(cos)
        f3 = CountCalls(max)
        function nested_checkpoints(x)
            Zygote.checkpointed() do
                a = f1(x)
                Zygote.checkpointed() do
                    b = f2(a)
                    Zygote.checkpointed() do
                        f3(a,b)
                    end
                end
            end
        end
        function nested_nocheckpoints(x)
            a = f1.f(x)
            b = f2.f(a)
            f3.f(a,b)
        end
        x = randn()
        y,pb = Zygote.pullback(nested_checkpoints, x)
        @test f1.ncalls == f2.ncalls == f3.ncalls
        dy = randn()
        pb(dy)
        @test f1.ncalls == 2
        @test f2.ncalls == 3
        @test f3.ncalls == 4
        y_ref, pb_ref = Zygote.pullback(nested_nocheckpoints, x)
        @test y_ref === y
        @test pb_ref(dy) == pb(dy)
    end
end

@testset "Interface" begin
  include("interface.jl")
end


@testset "Tools" begin
  include("tools.jl")
end

@testset "lib/number" begin
  include("lib/number.jl")
end

@testset "Features" begin
  include("features.jl")
end

@testset "Forward" begin
  include("forward/forward.jl")
end

@testset "Data Structures" begin
  include("structures.jl")
end

@testset "ChainRules" begin
  include("chainrules.jl")
end

@testset "Gradients" begin
  include("gradcheck.jl")
end

@testset "Complex" begin
  include("complex.jl")
end

@testset "Compiler" begin
  include("compiler.jl")
end

if has_cuda()
  @testset "CUDA tests" begin
    include("cuda.jl")
  end
else
  @warn "CUDA not found - Skipping CUDA Tests"
end
