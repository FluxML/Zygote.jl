using LinearAlgebra
using Zygote: hessian_dual, hessian_reverse

@testset "hessian: $hess" for hess in [hessian_dual, hessian_reverse]

  if hess == hessian_dual
    @test hess(x -> x[1]*x[2], randn(2)) ≈ [0 1; 1 0]
    @test hess(((x,y),) -> x*y, randn(2)) ≈ [0 1; 1 0]  # original docstring version
  else
    @test hess(x -> x[1]*x[2], randn(2)) ≈ [0 1; 1 0]  # can't differentiate ∇getindex
    @test hess(((x,y),) -> x*y, randn(2)) ≈ [0 1; 1 0]
  end
  @test hess(x -> sum(x.^3), [1 2; 3 4]) ≈ Diagonal([6, 18, 12, 24])
  @test hess(sin, pi/2) ≈ -1

  @test_throws Exception hess(sin, im*pi)
  @test_throws Exception hess(x -> x+im, pi)
  @test_throws Exception hess(identity, randn(2))
end

@testset "jacobian(f, args...)" begin
  @test jacobian(identity, [1,2])[1] == [1 0; 0 1]

  j1 = jacobian((a,x) -> a.^2 .* x, [1,2,3], 1)
  @test j1[1] ≈ Diagonal([2,4,6])
  @test j1[2] ≈ [1, 4, 9]
  @test j1[2] isa Vector

  j2 = jacobian((a,t) -> sum(a .* t[1]) + t[2], [1,2,3], (4,5))  # scalar output is OK
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

  j5 = jacobian((x,y) -> hcat(x[1], y), fill(pi), exp(1))  # zero-array
  @test j5[1] isa Matrix
  @test vec(j5[1]) == [1, 0]

  @test_throws ArgumentError jacobian(identity, [1,2,3+im])
  @test_throws ArgumentError jacobian(sum, [1,2,3+im])  # scalar, complex

  f6(x,y) = abs2.(x .* y)
  g6 = gradient(first∘f6, [1+im, 2], 3+4im)
  j6 = jacobian((x,y) -> abs2.(x .* y), [1+im, 2], 3+4im)
  @test j6[1][1,:] ≈ g6[1]
  @test j6[2][1] ≈ g6[2]
end

@testset "jacobian(loss, ::Params)" begin
  xs = [1 2; 3 4]
  ys = [5,7,9];
  Jxy = jacobian(() -> ys[1:2] .+ sum(xs.^2), Params([xs, ys]))
  @test Jxy[ys] ≈ [1 0 0; 0 1 0]
  @test Jxy[xs] ≈ [2 6 4 8; 2 6 4 8]
end
