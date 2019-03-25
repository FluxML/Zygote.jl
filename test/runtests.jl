using Zygote, Test
using Zygote: gradient

@testset "Zygote" begin

# @testset "Features" begin
#   include("features.jl")
# end

@testset "Gradients" begin
  include("gradcheck.jl")
end

# @testset "Compiler" begin
#   include("compiler.jl")
# end

end
