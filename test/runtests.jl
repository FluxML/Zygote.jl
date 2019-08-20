using Zygote, Test
using Zygote: gradient

if Zygote.usetyped
  @info "Testing Zygote in type-hacks mode."
else
  @info "Testing Zygote in normal mode."
end

@testset "Zygote" begin

@testset "Features" begin
  include("features.jl")
end

@testset "Data Structures" begin
  include("structures.jl")
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

end
