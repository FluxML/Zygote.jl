using CUDA
using Zygote: Grads
using Random: randn!
CUDA.allowscalar(false)

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

@testset "gradient algebra" begin
  w, b = rand(2) |> cu, rand(2) |> cu
  x1, x2 = rand(2) |> cu, rand(2) |> cu
 
  gs1 = gradient(() -> sum(w .* x1), Params([w])) 
  gs2 = gradient(() -> sum(w .* x2), Params([w])) 

  @test .- gs1 isa Grads
  @test gs1 .- gs2 isa Grads 
  @test .+ gs1 isa Grads
  @test gs1 .+ gs2 isa Grads 
  @test 2 .* gs1 isa Grads 
  @test (2 .* gs1)[w] ≈ 2 * gs1[w]
  @test gs1 .* 2 isa Grads 
  @test gs1 ./ 2 isa Grads  
  @test (gs1 .+ gs2)[w] ≈ gs1[w] .+ gs2[w] 

  gs12 = gs1 .+ gs2
  gs1 .+= gs2
  @test gs12[w] ≈ gs1[w] 

  gs3 = gradient(() -> sum(w .* x1), Params([w, b])) # grad nothing with respect to b
  gs4 = gradient(() -> sum(w .* x2 .+ b), Params([w, b])) 

  @test .- gs3 isa Grads
  @test gs3 .- gs4 isa Grads 
  @test .+ gs3 isa Grads
  @test gs3 .+ gs4 isa Grads 
  @test 2 .* gs3 isa Grads 
  @test gs3 .* 2 isa Grads 
  @test gs3 ./ 2 isa Grads  
  @test (gs3 .+ gs4)[w] ≈ gs3[w] .+ gs4[w]
  @test (gs3 .+ gs4)[b] ≈ gs4[b] 
  
  @test gs3 .+ IdDict(w => similar(w), b => similar(b)) isa Grads
  gs3 .+= IdDict(p => randn!(similar(p)) for p in keys(gs3))
  @test gs3 isa Grads 

  @test_throws ArgumentError gs1 .+ gs4
end
