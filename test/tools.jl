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

@testset "isderiving" begin

  function f(x)
    if Zygote.isderiving(x)
      x^2
    else
      2x^2
    end
  end

  # Test higher order derivatives
  gs = gradient(4) do x
    gradient(x) do y
      f(y)
    end[1]
  end

  @test gs == (2,)

  struct Tester
    cpu_offload::Float64
  end

  function Tester(p)
      @show Zygote.isderiving(p)
      cpu_offload = Zygote.isderiving(p) ? 0.0 : 0.2
      Tester(cpu_offload)
  end

  function f(p)
    sum(Tester(p).cpu_offload .* p)
  end

  p = [1.0]
  gs = gradient(f, p)
  @test gs[1] == [0.]

end
