using Zygote, Test
using Zygote: gradient
using CUDA: has_cuda

@testset "Interface" begin  
  include("interface.jl")
end


@testset "Tools" begin  
  include("tools.jl")
end


@testset "Features" begin
  include("features.jl")
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
