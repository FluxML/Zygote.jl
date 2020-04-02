@testset "Params delete!" begin
  w = rand(2,3)
  b = rand(2)
  ps = Params([w,b])
  delete!(ps, b)
  @test length(ps.order) == length(ps.params) == 1
  @test first(ps.order) == first(ps.params) == w
end