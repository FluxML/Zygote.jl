@testset "Buffer" begin
  w = rand(2,3)
  b = rand(2)
  c = rand(2)
  
  buff = Zygote.Buffer([])
  @test length(buff) == 0
  
  push!(buff, w)
  push!(buff, b)
  push!(buff, c)
  @test length(buff) == 3
  @test buff[1] === w 
  @test buff[2] === b 
  @test buff[3] === c 
  
  deleteat!(buff, 2)
  @test length(buff) == 2
  @test buff[1] === w 
  @test buff[2] === c 

  deleteat!(buff, 1)
  @test length(buff) == 1
  @test buff[1] === c
end