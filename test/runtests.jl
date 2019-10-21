using Zygote, Test
using Zygote: gradient

@testset "Zygote" begin

@info "Testing compiler features"

@testset "Features" begin
  include("features.jl")
end

@info "Testing data structures"

@testset "Data Structures" begin
  include("structures.jl")
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

@info "Starting GPU intgeration tests..."
if haskey(ENV, "CI_GITLAB_CUDA")
  @testset "CUDA tests" begin
    include("cuda.jl")
  end
else
  @info "ENV variable CI_GITLAB_CUDA not set - Skipping CUDA Tests"
end

end
