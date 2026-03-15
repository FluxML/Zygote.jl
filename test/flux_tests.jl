@testitem "flux" begin

using Flux
using Zygote

@testset "normalise keyword path" begin
  x = randn(Float32, 16, 10, 1)
  g = Zygote.gradient(x -> sum(Flux.normalise(x; dims = 1:1, eps = 1f-5)), x)
  @test size(g[1]) == size(x)
end

@testset "LayerNorm gradient" begin
  ln = Flux.LayerNorm(16)
  x = randn(Float32, 16, 10, 1)
  g = Zygote.gradient(x -> sum(ln(x)), x)
  @test size(g[1]) == size(x)
end

end
