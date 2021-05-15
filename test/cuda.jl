using CUDA

# Test GPU movement inside the call to `gradient`
@testset "GPU movement" begin
  r = rand(Float32, 3,3)
  @test gradient(x -> sum(cu(x)), r)[1] isa Array{Float32, 2}
end

@testset "basic bcasting" begin
  a = Float32.(1:9)
  a_gpu = a |> cu

  v(x, n) = sum(x .^ n)
  g = gradient(x -> v(x, 7), a)[1]
  g_gpu = gradient(x -> v(x, 7), a_gpu)[1]
  @test g_gpu isa CuArray
  @test g_gpu |> collect ≈ g
  
  w(x) = sum(broadcast(log, x))
  g = gradient(x -> w(x), a)[1]
  g_gpu = gradient(x -> w(x), a_gpu)[1]
  @test g_gpu isa CuArray
  @test g_gpu |> collect ≈ g
  
end

@testset "un-broadcasting *, / with mapreduce" begin
  cu12 = cu(Float32[1,2])
  @test gradient((x,y) -> sum(x .* y), cu12, 5) == ([5, 5], 3)
  @test gradient((x,y) -> sum(x .* y), 5, cu12) == (3, [5, 5])
  @test gradient((x,y) -> sum(x .* y), cu12, [3 4 5]) == ([12, 12], [3 3 3])
  @test gradient((x,y) -> sum(x ./ y), cu12, 5) == ([0.2, 0.2], -0.12)
end

@testset "jacobian" begin
  v1 = cu(collect(1:3f0))

  res1 = jacobian(x -> x .* x', collect(1:3f0))[1]
  j1 = jacobian(x -> x .* x', v1)[1]
  @test j1 isa CuArray
  @test j1 ≈ cu(res1)

  res2 = jacobian(x -> x ./ sum(x), collect(1:3f0))[1]
  j2 = jacobian(() -> v1 ./ sum(v1), Params([v1]))
  @test j2[v1] isa CuArray
  @test j2[v1] ≈ cu(res2)
end
