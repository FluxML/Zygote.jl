@testset "Params delete!" begin
  w = rand(2,3)
  b = rand(2)
  ps = Params([w,b])
  delete!(ps, b)
  @test length(ps.order) == length(ps.params) == 1
  @test first(ps.order) == first(ps.params) == w
end

@testset "copyto!" begin
  x = [0,0]
  ps = Params([x])
  copyto!(ps, [1, 2])
  @test x == [1, 2]
  
  x = [0,0]
  y = [0]
  ps = Params([x, y])
  copyto!(ps, [1, 2, 3])
  @test x == [1, 2]
  @test y == [3]

  ps = Params([[1,2]])
  x = [0, 0]
  copyto!(x, ps)
  @test x == [1, 2]
  
  ps = Params([[1,2], [3]])
  x = [0, 0, 0]
  copyto!(x, ps)
  @test x == [1, 2, 3]
end