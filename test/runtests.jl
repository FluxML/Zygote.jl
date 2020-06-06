using Zygote, Test
using Zygote: gradient

@testset "Zygote" begin

@info "Testing interface" 

@testset "Interface" begin  
  include("interface.jl")
end

@info "Testing tools" 

@testset "Tools" begin  
  include("tools.jl")
end

@info "Testing compiler features"

@testset "Features" begin
  include("features.jl")
end

@info "Testing data structures"

@testset "Data Structures" begin
  include("structures.jl")
end

@info "Testing ChainRules integration"

@testset "ChainRules" begin
  include("chainrules.jl")
end

@info "Running Gradient Checks"

@testset "Gradients" begin
  include("gradcheck.jl")
end

@testset "Complex" begin
  include("complex.jl")
end

@info "Testing Inference & Debug Info"

@testset "Compiler" begin
  include("compiler.jl")
end

using CUDA
if has_cuda()
  @info "Starting GPU integration tests..."
  @testset "CUDA tests" begin
    include("cuda.jl")
  end
else
  @warn "CUDA not found - Skipping CUDA Tests"
end

end
