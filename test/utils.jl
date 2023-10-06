using ForwardDiff
using Zygote: hessian_dual, hessian_reverse

@testset "hessian: $hess" for hess in [hessian_dual, hessian_reverse]
  @test hess(x -> x[1]*x[2], randn(2)) ≈ [0 1; 1 0]
  @test hess(((x,y),) -> x*y, randn(2)) ≈ [0 1; 1 0]  # original docstring version
  @test hess(x -> sum(x.^3), [1 2; 3 4]) ≈ Diagonal([6, 18, 12, 24])
  @test hess(sin, pi/2) ≈ -1

  @test_throws Exception hess(sin, im*pi)
  @test_throws Exception hess(x -> x+im, pi)
  @test_throws Exception hess(identity, randn(2))
end

VERSION > v"1.6-" && @testset "diagonal hessian" begin
  @test diaghessian(x -> x[1]*x[2]^2, [1, pi]) == ([0, 2],)

  if VERSION > v"1.6-"
    # Gradient of ^ may contain log(complex(...)), which interacts badly with Dual below Julia 1.6:
    # julia> log(ForwardDiff.Dual(1,0) + 0im) # ERROR: StackOverflowError:
    # https://github.com/JuliaDiff/ChainRules.jl/issues/525
    # Fixed in 1.6 by: https://github.com/JuliaLang/julia/pull/36030
    xs, y = randn(2,3), rand()
    f34(xs, y) = xs[1] * (sum(xs .^ (1:3)') + y^4)  # non-diagonal Hessian, two arguments

    dx, dy = diaghessian(f34, xs, y)
    @test size(dx) == size(xs)
    @test vec(dx) ≈ diag(hessian(x -> f34(x,y), xs))
    @test dy ≈ hessian(y -> f34(xs,y), y)
  end

  zs = randn(7,13)  # test chunk mode
  @test length(zs) > ForwardDiff.DEFAULT_CHUNK_THRESHOLD
  @test length(zs) % ForwardDiff.DEFAULT_CHUNK_THRESHOLD != 0
  f713(zs) = sum(vec(zs)' .* exp.(vec(zs)))
  @test vec(diaghessian(f713, zs)[1]) ≈ diag(hessian(f713, zs))

  @test_throws Exception diaghessian(sin, im*pi)
  @test_throws Exception diaghessian(x -> x+im, pi)
  @test_throws Exception diaghessian(identity, randn(2))
end

@testset "jacobian(f, args...)" begin
  @test jacobian(identity, [1,2])[1] == [1 0; 0 1]
  @test withjacobian(identity, [1,2]) == (val = [1,2], grad = ([1 0; 0 1],))

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
  @test j5[2] == [0, 1]

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

  z, grad = withjacobian(() -> ys[1:2] .+ sum(xs.^2), Params([xs, ys]))
  @test z == [35, 37]
  @test grad[ys] ≈ [1 0 0; 0 1 0]
end

using ForwardDiff

@testset "adjoints of ForwardDiff functions" begin
  f1(x) = ForwardDiff.gradient(x -> sum(exp.(x.+1)), x)
  x1 = randn(3,7)
  @test Zygote.jacobian(f1, x1)[1] ≈ ForwardDiff.jacobian(f1, x1)

  f2(x) = ForwardDiff.jacobian(x -> log.(x[1:3] .+ x[2:4]), x)
  x2 = rand(5) .+ 1
  @test Zygote.jacobian(f2, x2)[1] ≈ ForwardDiff.jacobian(f2, x2)

  f3(x) = sum(ForwardDiff.hessian(x -> sum(x .^2 .* x'), x)[1:4:end])
  x3 = rand(3)
  @test Zygote.gradient(f3, x3)[1] ≈ ForwardDiff.gradient(f3, x3)

  @test gradient(x -> ForwardDiff.derivative(x -> x^4, x), 7) == (4 * 3 * 7^2,)

  f4(x) = ForwardDiff.derivative(x -> [x,x^2,x^3], x)
  @test Zygote.jacobian(f4, pi)[1] ≈ ForwardDiff.derivative(f4, pi)

  # Tests from https://github.com/FluxML/Zygote.jl/issues/769
  f(x) = [2x[1]^2 + x[1],x[2]^2 * x[1]]
  g1(x) = sum(ForwardDiff.jacobian(f,x))
  out,back = Zygote.pullback(g1,[2.0,3.2])
  stakehouse = back(1.0)[1]
  @test typeof(stakehouse) <: Vector
  @test size(stakehouse) == (2,)
  @test stakehouse ≈ ForwardDiff.gradient(g1,[2.0,3.2])

  g2(x) = prod(ForwardDiff.jacobian(f,x))
  out,back = Zygote.pullback(g2,[2.0,3.2])
  @test_skip back(1.0)[1] == ForwardDiff.gradient(g2,[2.0,3.2])  # contains NaN, @adjoint prod isn't careful

  g3(x) = sum(abs2,ForwardDiff.jacobian(f,x))
  out,back = Zygote.pullback(g3,[2.0,3.2])
  @test back(1.0)[1] == ForwardDiff.gradient(g3,[2.0,3.2])
  
  # From https://github.com/FluxML/Zygote.jl/issues/1218
  f1218(x::AbstractVector,y::AbstractVector) = sum(x)*sum(y)
  gradf1218(x,y) = ForwardDiff.gradient(x->f1218(x,y), x)[1]
  x = [0.1]
  y = rand(5)
  @test ForwardDiff.gradient(y->gradf1218(x,y), y) == ones(5)
  # this returns (nothing,) -- now prints a warning
  @test_broken Zygote.gradient(y->gradf1218(x,y), y) == ones(5)
end
