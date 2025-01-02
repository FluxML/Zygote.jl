using Zygote, Test, LinearAlgebra
using Zygote: gradient, ZygoteRuleConfig

@testset "all" begin  # Overall testset ensures it keeps running after failure
  if !haskey(ENV, "GITHUB_ACTION")
    using CUDA
    if CUDA.has_cuda()
      @testset "CUDA tests" begin
        include("cuda.jl")
      end
      @info "CUDA tests have run"
    else
      @warn "CUDA not found - Skipping CUDA Tests"
    end
  end

  @testset "deprecated.jl" begin
    include("deprecated.jl")
  end

  @testset "Interface" begin
    include("interface.jl")
  end

  @testset "Tools" begin
    include("tools.jl")
  end

  @testset "Utils" begin
    include("utils.jl")
  end

  @testset "lib" begin
    include("lib/number.jl")
    include("lib/lib.jl")
    include("lib/base.jl")
    include("lib/array.jl")
  end

  @testset "Features" begin
    include("features.jl")
    @info "features.jl done"
  end

  @testset "Forward" begin
    include("forward/forward.jl")
  end

  @testset "Data Structures" begin
    include("structures.jl")
  end

  @testset "ChainRules" begin
    include("chainrules.jl")
    @info "chainrules.jl done"
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

end # @testset "all"
