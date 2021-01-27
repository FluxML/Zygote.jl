using LinearAlgebra

@testset "jacobian(f, x, y)" begin
  j1 = jacobian((a,x) -> a.^2 .* x, [1,2,3], 1)
  @test j1[1] ≈ Diagonal([2,4,6])
  @test j1[2] ≈ [1, 4, 9]
  @test j1[2] isa Vector

  j2 = jacobian((a,t) -> sum(a .* t[1]) + t[2], [1,2,3], (4,5))
  @test j2[1] == [4 4 4]
  @test j2[1] isa Matrix
  @test j2[2] === nothing  # input other than Number, Array is ignored

  j3 = jacobian((a,d) -> prod(a, dims=d), [1 2; 3 4], 1)
  @test j3[1] ≈ [3 1 0 0; 0 0 4 2]
  @test j3[2] ≈ [0, 0]  # pullback is always Nothing, but array already allocated

  j4 = jacobian([1,2,-3,4,-5]) do xs
    map(x -> x>0 ? x^3 : 0, xs)  # pullback gives Nothing for some elements x
  end
  @test j4[1] ≈ Diagonal([3,12,0,48,0])
end

@testset "jacobian(loss, ::Params)" begin
  xs = [1 2; 3 4]
  ys = [5,7,9];
  Jxy = jacobian(() -> ys[1:2] .+ sum(xs.^2), Params([xs, ys]))
  @test Jxy[ys] ≈ [1 0 0; 0 1 0]
  @test Jxy[xs] ≈ [2 6 4 8; 2 6 4 8]
end
