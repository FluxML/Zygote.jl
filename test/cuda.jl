# Test GPU movement inside the call to `gradient`
@testset "GPU movement" begin
  r = rand(Float32, 3,3)
  @test gradient(x -> sum(cu(x)), r)[1] isa AbstractArray
end

@testset "basic bcasting" begin
  a = cu(Float32.(1:9))
  v(x, n) = x .^ n
  pow_grada = cu(Float32[7.0, 448.0, 5103.0, 28672.0, 109375.0, 326592.0, 823543.0, 1.835008e6, 3.720087e6])
  @test gradient(x -> v(x, 7) |> sum, a) == (pow_grada,)
  w(x) = broadcast(log, x)
  log_grada = cu(Float32[1.0, 0.5, 0.33333334, 0.25, 0.2, 0.16666667, 0.14285715, 0.125, 0.11111111])
  @test gradient(x -> w(x) |> sum, a) == (log_grada,)
end
