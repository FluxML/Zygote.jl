@testset "nograds" begin 
  @test gradient(floor, 1) === nothing 
  @test gradient(ceil, 1) === nothing 
  @test gradient(round, 1) === nothing 
  @test gradient(hash, 1) === nothing 
  @test gradient(div, 1, 2) === nothing 
end #testset
