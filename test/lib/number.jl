@testset "nograds" begin
  @test gradient(floor, 1) === (0.0,)
  @test gradient(ceil, 1) === (0.0,)
  @test gradient(round, 1) === (0.0,)
  @test gradient(hash, 1) === nothing
  @test gradient(div, 1, 2) === nothing
end #testset
