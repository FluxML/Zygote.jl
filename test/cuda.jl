using CUDA
using Zygote: Grads
using Random: randn!
import FiniteDifferences
CUDA.allowscalar(false)

function gradcheck_gpu(f, xs...)
    grad_zygote = gradient(f, xs...)
    m = FiniteDifferences.central_fdm(5,1)
    grad_finite_difference = FiniteDifferences.grad(m, f, collect.(xs)...)
    return all(isapprox.(collect.(grad_zygote), grad_finite_difference))
end


# Test GPU movement inside the call to `gradient`
@testset "GPU movement" begin
  r = rand(Float32, 3,3)
  @test gradient(x -> sum(cu(x)), r)[1] isa Matrix{Float32}
  @test gradient(x -> sum(x->log(x), cu(x)), r)[1] isa Matrix
  @test gradient((x,cy) -> sum(cu(x) * cy) + sum(cy'), r, cu(r))[2] isa CUDA.CuArray
  @test_skip gradient((x,cy) -> sum(cu(x[:,1])' * cy), r, cu(r))[2] isa CUDA.CuArray # generic_matmatmul!

  # Other direction:
  @test_skip gradient(x -> sum(Array(x)), cu(r))[1] isa CUDA.CuArray
  @test_skip gradient((x,cy) -> sum(x * Array(cy)) + sum(cy'), r, cu(r))[2] isa CUDA.CuArray
end

@testset "broadcasting" begin
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

  # https://github.com/FluxML/Zygote.jl/issues/1027                  # status on Zygote v0.6.14, CUDA v3.3.0 in comments:
  @test gradient(x -> sum(x .!= 0), a_gpu) == (nothing,)             # was MethodError: no method matching iterate(::Nothing)
  @test gradient(x -> sum(x .> 3), a_gpu) == (nothing,)
  g3 = gradient(x -> sum(x .^ 3) / count(x .> 3), a)[1]              # was Can't differentiate gc_preserve_end expression
  @test_skip cu(g3) ≈ gradient(x -> sum(x .^ 3) / sum(x .> 3), a_gpu)[1]  # was KernelException -- not fixed by PR #1018
  @test cu(g3) ≈ gradient(x -> sum(x .^ 3) / count(x .> 3), a_gpu)[1]

  # Projection: eltype preservation:
  @test gradient(x -> 2.3 * sum(x.^4), a_gpu)[1] isa CuArray{Float32}
  @test_skip gradient(x -> sum(x .* 5.6), a_gpu)[1] isa CUDA.CuArray{Float32} # dot(x::CuArray{Float64}, y::CuArray{Float32}) fallback
  # structure restoration:
  @test gradient(x -> sum(sqrt.(x)), a_gpu')[1] isa Adjoint  # previously a matrix
  @test gradient(x -> sum(exp.(x)), Diagonal(a_gpu))[1] isa Diagonal
  # non-differentiables
  @test gradient((x,y) -> sum(x.^2 .+ y'), a_gpu, a_gpu .> 0)[2] === nothing
end

@testset "sum(f, x)" begin
  a = Float32[-1.5, -9.0, 2.4, -1.3, 0.01]
  a_gpu = a |> cu

  f(x) = sum(abs, x)
  g = gradient(f, a)[1]
  g_gpu = gradient(f, a_gpu)[1]
  @test g_gpu isa CuArray
  @test g_gpu |> collect ≈ g

  f2(x) = sum(abs2, x)  # sum(abs2, x) has its own rrule
  g2 = gradient(f2, a)[1]
  g2_gpu = gradient(f2, a_gpu)[1]
  @test g2_gpu isa CuArray
  @test g2_gpu |> collect ≈ g2

  f3(x) = sum(y->y^3, x')  # anonymous function
  g3 = gradient(f3, a')[1]
  g3_gpu = gradient(f3, a_gpu')[1]
  @test g3_gpu isa Adjoint{Float32, <:CuArray{Float32, 1}}  # preserves structure
  @test g3_gpu |> collect ≈ g3
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

@testset "vcat scalar indexing" begin
  r = cu(rand(Float32, 3))
  grads = (cu(ones(Float32, 3)), 1.f0)
  @test gradient((x,y) -> sum(vcat(x,y)), r, 5) == grads

  @test gradient((x,y) -> sum(vcat(x,y)), r, Float64(5))[1] isa CUDA.CuArray{Float32}
  @test gradient((x,y) -> sum(vcat(x,y)), r, Float64(5))[2] isa Float64  # projection

  @test_skip gradient((x,y) -> sum(vcat(x,y)), 5f0, r)[2] isa CUDA.CuArray{Float32}  # wrong order
  @test_skip gradient((x,y) -> sum(vcat(x,y)), 1f0, r, 2f0, r)[2] isa CUDA.CuArray{Float32}
end


@testset "CUDA complex broadcasting" begin
    # Issue 961 and 1121 and 1215
    x = 2*rand(Float32, 10) .- 1f0
    y = 2*rand(ComplexF32, 10) .- 1f0

    xgpu =cu(x)
    ygpu =cu(y)

    g1 = Zygote.gradient(x->sum(abs2, x), ygpu)[1]
    g2 = Zygote.gradient(x->sum(abs2.(x)), ygpu)[1]
    g3 = Zygote.gradient(x->sum(abs2, x), y)[1]
    @test g1 isa CUDA.CuArray{ComplexF32}
    @test g2 isa CUDA.CuArray{ComplexF32}
    @test collect(g1) ≈ collect(g2)
    @test collect(g1) ≈ g3


    r3 = cu(Float32.(inv.(2:4)))
    c3 = cu(ComplexF32.(inv.(5:7) .+ im ./ (8:10)))


    # These check _broadcast_forward(::Type{<:Dual}, ...)
    @test gradcheck_gpu((x,y)->sum(abs2, x.^2  .+ y), xgpu, ygpu)
    @test gradcheck_gpu((x,y)->sum(abs, exp.(x) .+ imag.(y)), xgpu, ygpu)

    # These check _broadcast_forward_complex(::Type{<:Complex}, ...)
    @test gradcheck_gpu((x,y)->sum(abs2, cos.(x) .+ sin.(y)), xgpu, ygpu)
    @test gradcheck_gpu((x,y)->sum(abs, cos.(x).*sin.(y)), xgpu, ygpu)
    @test gradcheck_gpu((x,y)->sum(abs, cos.(x) .+ sin.(conj.(y))), xgpu, ygpu)
    @test gradcheck_gpu((x,y)->sum(abs, cos.(x) .+ sin.(conj.(y))), xgpu, ygpu)
    @test gradcheck_gpu((r,c) -> sum(abs2, sin.(conj.(c)./transpose(r) .- im) .- imag.(c .+ tanh.(r./c'))), r3, c3)

    # Painful test!
    @test gradcheck_gpu(c -> sum(abs2, imag.(sqrt.(c .+ im))), c3)
    @test gradcheck_gpu(r -> sum(abs2, log.(1 .+ im .* r)./2), r3)


    # These check _broadcast_forward(::Type{<:Complex}, ...)
    @test gradcheck_gpu(x->sum(real, cis.(x)), xgpu)
    @test gradcheck_gpu(x->sum(real, cispi.(x)), xgpu)

    # These check _broadcast_forward_complex(::Type{<:Dual}, ...)
    @test gradcheck_gpu(x->sum(imag, x.^2 .+ abs.(sinh.(conj.(x)))), ygpu)


end
