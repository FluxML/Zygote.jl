using Zygote, Test
using Zygote: gradient
using CUDA: has_cuda

using Zygote
@testset "gradient checkpointing" begin
    mutable struct CountCalls
        ncalls
    end
    CountCalls() = CountCalls(0)
    function (o::CountCalls)(x)
        o.ncalls += 1
        x
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
    @test pb(1.0) === (1.0,)
    @test f.ncalls === g.ncalls === h.ncalls === 2
    @test pb(1.0) === (1.0,)
    @test f.ncalls === g.ncalls === h.ncalls === 3

    f = CountCalls()
    g = CountCalls()
    h = CountCalls()
    function doit(x)
        x2 = f(x)
        Zygote.checkpointed(h∘g, x2)
    end
    y,pb = Zygote.pullback(doit, 4.0)
    @test f.ncalls === g.ncalls === h.ncalls === 1
    @show pb(1.0)
    @test pb(1.0) === (1.0,)
    @test g.ncalls === h.ncalls === 2
    @test f.ncalls === 1
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
