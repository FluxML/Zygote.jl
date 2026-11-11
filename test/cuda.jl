using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSC
using FillArrays: Fill
using ChainRulesCore: ProjectTo
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

  # Other direction: https://github.com/FluxML/Zygote.jl/issues/1305 — `Array`/
  # `collect` move a GPU array to the host, so their pullback must move the
  # cotangent back to the device instead of leaving it a CPU array.
  @test gradient(x -> sum(Array(x)), cu(r))[1] isa CUDA.CuArray
  @test gradient(x -> sum(collect(x)), cu(r))[1] isa CUDA.CuArray
  @test gradient((x,cy) -> sum(x * Array(cy)) + sum(cy'), r, cu(r))[2] isa CUDA.CuArray
  @test collect(gradient(x -> sum(abs2, Array(x)), cu(r))[1]) ≈ gradient(x -> sum(abs2, Array(x)), r)[1]
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
  # https://github.com/FluxML/Zygote.jl/issues/1597 : Bool broadcasts must not go
  # through ForwardDiff Duals on GPU (predicate ties get broken using partials).
  @test cu(g3) ≈ gradient(x -> sum(x .^ 3) / count(x .> 3), a_gpu)[1]
  @test gradient(x -> sum(x .^ 3) / sum(x .> 3), a_gpu)[1] ≈ cu(g3)

  # Projection: eltype preservation:
  @test gradient(x -> 2.3 * sum(x.^4), a_gpu)[1] isa CuArray{Float32}
  @test_skip gradient(x -> sum(x .* 5.6), a_gpu)[1] isa CUDA.CuArray{Float32} # dot(x::CuArray{Float64}, y::CuArray{Float32}) fallback
  # structure restoration:
  @test gradient(x -> sum(sqrt.(x)), a_gpu')[1] isa Adjoint  # previously a matrix
  @test gradient(x -> sum(exp.(x)), Diagonal(a_gpu))[1] isa Diagonal
  # non-differentiables
  @test gradient((x,y) -> sum(x.^2 .+ y'), a_gpu, a_gpu .> 0)[2] === nothing

  # https://github.com/FluxML/Zygote.jl/issues/1696 : a runtime exponent has its own rule.
  # Through `Dual`s, the exponent's partial `log(x)` made the gradient `NaN` for `x < 0`.
  let xn = Float32[-2, -1, 0.5, 3], xp = Float32[0.5, 1, 2, 3]
    cpu(x) = x isa CuArray ? collect(x) : x
    for (x, p) in ((xn, 2), (xn, 3f0), (xn, 0), (xp, 0.5f0), (xp, Float32[0.5, 1, -1, 2]))
      g = gradient((x, p) -> sum(x .^ p), x, p)
      g_gpu = gradient((x, p) -> sum(x .^ p), cu(x), p isa Array ? cu(p) : p)
      @test g_gpu[1] isa CuArray{Float32}
      @test collect(g_gpu[1]) ≈ g[1]
      @test cpu(g_gpu[2]) ≈ g[2]
    end
    @test gradient(x -> sum(x .^ 0.5), cu(xp)')[1] isa Adjoint{Float32, <:CuArray}
  end

  # https://github.com/FluxML/Zygote.jl/issues/1424 : a broadcast whose output
  # element is a Tuple (here `tuple.(x, y)`) must not leak ForwardDiff `Dual`s
  # into the primal, and its pullback must not be `nothing`.
  let x = cu(Float32[1, 2, 3]), y = cu(Float32[4, 5, 6])
    f(x, y) = broadcast(tuple, x, y)
    o, back = pullback(f, x, y)
    @test eltype(o) == Tuple{Float32,Float32}
    @test collect(o) == collect(broadcast(tuple, x, y))
    gx, gy = back(o)
    oc, backc = pullback(f, collect(x), collect(y))
    gxc, gyc = backc(oc)
    @test gx isa CuArray && gy isa CuArray
    @test collect(gx) ≈ gxc
    @test collect(gy) ≈ gyc
  end
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

  # https://github.com/FluxML/Zygote.jl/issues/1529 : `sum(abs, x)` on the GPU
  # goes through the forward-mode broadcast; at a zero input ForwardDiff used to
  # give 1 (real) / NaN (complex via `hypot`) instead of the CPU's 0.
  @test collect(gradient(f, cu(Float32[0.0]))[1]) == Float32[0.0]
  @test collect(gradient(f, cu(ComplexF32[0.0 + 0.0im]))[1]) == ComplexF32[0.0 + 0.0im]
  ca = ComplexF32[3.0 + 4.0im, 0.0 + 0.0im, -1.0 + 0.0im]
  @test collect(gradient(f, cu(ca))[1]) ≈ gradient(f, ca)[1]

  # https://github.com/FluxML/Zygote.jl/issues/1498 : `sum(f, x)` over a GPU
  # array *wrapper* (here a logical `view`) used to scalar-index, because the
  # fast forward-mode rule only matched bare `AbstractGPUArray`s.
  let X = randn(Float32, 16, 16), iso = rand(Bool, 16, 16)
    h(x, m) = sum(z -> abs2(1 - z), view(x, m))
    g_gpu = gradient(x -> h(x, cu(iso)), cu(X))[1]
    @test g_gpu isa CuArray
    @test collect(g_gpu) ≈ gradient(x -> h(x, iso), X)[1]
  end
end

# https://github.com/FluxML/Zygote.jl/issues/1701 : `∇map` stored per-element
# pullbacks in a device array, which failed GPU compilation.
@testset "map" begin
  a = Float32[-1.5, -9.0, 2.4, -1.3, 0.01]
  b = Float32[0.3, 2.0, -1.1, 4.2, 1.5]
  a_gpu, b_gpu = cu(a), cu(b)

  g_gpu = gradient(x -> sum(map(abs2, x)), a_gpu)[1]
  @test g_gpu isa CuArray
  @test collect(g_gpu) ≈ gradient(x -> sum(map(abs2, x)), a)[1]

  f2(x, y) = sum(map((xi, yi) -> sin(xi) * yi, x, y))
  g2_gpu = gradient(f2, a_gpu, b_gpu)
  @test all(g -> g isa CuArray, g2_gpu)
  @test all(map((g, g_cpu) -> collect(g) ≈ g_cpu, g2_gpu, gradient(f2, a, b)))

  @test collect(gradient(x -> sum(map(*, x, 1:5)), a_gpu)[1]) ≈ 1:5

  M = randn(Float32, 3, 4)
  gM_gpu = gradient(x -> sum(map(tanh, x)), cu(M))[1]
  @test gM_gpu isa CuMatrix
  @test collect(gM_gpu) ≈ gradient(x -> sum(map(tanh, x)), M)[1]
  @test collect(gradient(x -> sum(map(tanh, x)), cu(M)')[1]) ≈ gradient(x -> sum(map(tanh, x)), M')[1]

  c = ComplexF32[1 + 2im, 3 - 1im]
  @test collect(gradient(x -> sum(abs2, map(z -> z^2, x)), cu(c))[1]) ≈ gradient(x -> sum(abs2, map(z -> z^2, x)), c)[1]

  @test gradient(x -> sum(map(xi -> xi > 0, x)), a_gpu) == (nothing,)

  # 0-dim, whose broadcast returns a scalar #1725
  x0 = fill!(similar(a_gpu, ()), 2f0)
  g0 = gradient(x -> sum(map(abs2, x)), x0)[1]
  @test g0 isa CuArray{Float32, 0}
  @test Array(g0) == fill(4f0)
end

# Forward-mode GPU broadcasts used to return `nothing` for variables captured by
# the broadcast function. #1701
@testset "captured variables" begin
  a = Float32[-1.5, -9.0, 2.4, -1.3, 0.01]
  a_gpu = cu(a)

  for h in ((x, w) -> sum(map(xi -> xi * w, x)),
            (x, w) -> sum((xi -> sin(xi * w)).(x)),
            (x, w) -> sum(xi -> abs2(xi - w), x),
            (x, w) -> sum(Base.Fix2(*, w).(x)))
    g = gradient(w -> h(a_gpu, w), 2f0)[1]
    @test g isa Float32
    @test g ≈ gradient(w -> h(a, w), 2f0)[1]
  end

  f3(x, w, b) = sum(map(xi -> tanh(xi * w + b), x))
  g3_gpu = gradient(f3, a_gpu, 2f0, 0.5f0)
  g3 = gradient(f3, a, 2f0, 0.5f0)
  @test g3_gpu[1] isa CuArray
  @test collect(g3_gpu[1]) ≈ g3[1]
  @test g3_gpu[2] ≈ g3[2]
  @test g3_gpu[3] ≈ g3[3]

  # ChainRules' derivative of `xi^p` in `p` for `xi < 0`, not ForwardDiff's `NaN` (#1725)
  g_pow = gradient((x, p) -> sum(map(xi -> xi^p, x)), a_gpu, 2f0)
  @test collect(g_pow[1]) ≈ 2 .* a
  @test g_pow[2] ≈ gradient((x, p) -> sum(map(xi -> xi^p, x)), a, 2f0)[2]
  q = Float32[1, 2, 1, 0, 3]
  fq(x, q) = sum(map((xi, qi) -> xi^(qi + 1), x, q))
  @test all(map((g, g_cpu) -> collect(g) ≈ g_cpu, gradient(fq, a_gpu, cu(q)), gradient(fq, a, q)))

  # integer fields are not differentiated
  f4(x, w) = sum(let n = 2; map(xi -> xi^n * w, x) end)
  @test gradient(w -> f4(a_gpu, w), 2f0)[1] ≈ gradient(w -> f4(a, w), 2f0)[1]

  c = ComplexF32[1 + 2im, 3 - 1im]
  f5(x, w) = sum(abs2, map(z -> z * w, x))
  @test gradient(w -> f5(cu(c), w), 1f0 + 2f0im)[1] ≈ gradient(w -> f5(c, w), 1f0 + 2f0im)[1]
  @test gradient(w -> f5(cu(c), w), 0.5f0)[1] ≈ gradient(w -> f5(c, w), 0.5f0)[1]
  @test gradient(w -> f5(cu(a), w), 1f0 + 2f0im)[1] ≈ gradient(w -> f5(a, w), 1f0 + 2f0im)[1]

  # `Tuple` output, with one component independent of `w`
  o, back = pullback(w -> map(xi -> (xi * w, xi), a_gpu), 2f0)
  @test back(o)[1] ≈ pullback(w -> map(xi -> (xi * w, xi), a), 2f0)[2](collect(o))[1]

  @test gradient(w -> sum(map(xi -> xi > w, a_gpu)), 2f0) == (nothing,)

  f6(x) = sum(let w = 3f0; map(xi -> sin(xi) * w, x) end)
  @test collect(Zygote.hessian(f6, a_gpu)) ≈ Zygote.hessian(f6, a)
end

# Floats nested in the function's fields, and captured `Dual`s of nested forward mode. #1725
struct AffineC; a::Float32; b::Float32; end
(m::AffineC)(x) = m.a * x + m.b
approx_tangent(x, y) = x === y || (x isa Number && y isa Number && x ≈ y) ||
  (x isa Union{Tuple, NamedTuple} && y isa typeof(x).name.wrapper && all(map(approx_tangent, Tuple(x), Tuple(y))))

@testset "captured variables, nested" begin
  a = Float32[-1.5, -9.0, 2.4, -1.3, 0.01]
  a_gpu = cu(a)

  g = let w = 2f0; yi -> yi * w end
  for (h, p) in (((x, p) -> sum(map(xi -> xi * p.a + p.b, x)), (a = 2f0, b = 0.5f0)),
                 ((x, t) -> sum((xi -> xi * t[1] + t[2]).(x)), (2f0, 0.5f0)),
                 ((x, g) -> sum(map(xi -> g(xi) + 1, x)), g),
                 ((x, g) -> sum(map(g ∘ sin, x)), g))
    @test approx_tangent(gradient(p -> h(a_gpu, p), p)[1], gradient(p -> h(a, p), p)[1])
  end

  # `hessian` with respect to a captured scalar, also used outside the `map`
  for h in ((x, w) -> (w1 = w[1]; sum(map(xi -> sin(xi * w1), x))),
            (x, w) -> (w1 = w[1]; sum(map(xi -> sin(xi * w1), x)) * w1))
    @test Zygote.hessian(w -> h(a_gpu, w), [2f0]) ≈ Zygote.hessian(w -> h(a, w), [2f0])
  end
  v = Float32[0.3, -0.5, 0.8]
  h2(v) = (w = sum(v); sum(map(xi -> sin(xi * w), v)))
  @test collect(Zygote.hessian(h2, cu(v))) ≈ Zygote.hessian(h2, v)

  # can't hold or take `Dual`s: warns
  @test (@test_logs (:warn, r"f\.a, f\.b") gradient(m -> sum(map(m, a_gpu)), AffineC(2f0, 0.5f0))) == (nothing,)
  @test (@test_logs (:warn, r"type assertion") gradient(w -> sum(map(xi -> xi * (w::Float32), a_gpu)), 2f0)) == (nothing,)
end

# The CPU rule for `accumulate` keeps a pullback per step. #1684
@testset "accumulate" begin
  op(a, b) = a + b + a * b / 2  # associative, as CUDA's `accumulate` assumes
  v, M, p = rand(Float32, 7), rand(Float32, 3, 4), 0.7f0
  for (h, args) in [
      ((x, p) -> sum(abs2, accumulate(op, x; init=p)), (v, p)),
      (x -> sum(abs2, accumulate(op, x)), (v,)),
      ((x, p) -> sum(accumulate(*, x; init=p)), (v, 0.7)),
      ((x, p) -> sum(accumulate(op, x; init=p)), (rand(Float32, 10_000) ./ 1000, p)),
      ((x, p) -> sum(abs2, accumulate(op, x; init=p)), (M, p)),
      (x -> sum(abs2, accumulate(op, x)), (M,)),
      ((x, p) -> sum(abs2, accumulate(op, x; dims=1, init=p)), (M, p)),
      ((x, p) -> sum(abs2, accumulate(op, x; init=p, dims=2)), (M, p)),
      (x -> sum(abs2, accumulate(op, x; dims=2)), (M,)),
      ((x, p) -> sum(abs2, accumulate(op, x; dims=3, init=p)), (M, p)),
      ((x, p) -> sum(abs2, accumulate(op, x; dims=2, init=p)), (rand(Float32, 2, 3, 4), p)),
      ((x, α) -> sum(abs2, accumulate((a, b) -> a + b + α * a * b, x; dims=2, init=α)), (M, 0.9f0)),
    ]
    g = gradient(h, map(a -> a isa Array ? cu(a) : a, args)...)
    @test g[1] isa CuArray
    @test all(map((g, g_cpu) -> g === g_cpu || (g isa CuArray ? collect(g) : g) ≈ g_cpu, g, gradient(h, args...)))
  end
end

# GPU broadcasts recompute the `Dual`s in the pullback instead of storing them. #1696
@testset "broadcast_forward recompute" begin
  σ(x) = 1 / (1 + exp(-x))
  lstm(f, c, i, g) = σ(f) * c + σ(i) * tanh(g)
  F, C = randn(Float32, 4, 3), randn(Float32, 4, 3)
  i, g = randn(Float32, 4), 0.7f0
  h(F, C, i, g) = sum(lstm.(F, C, i, g))
  @test all(map((a, b) -> (a isa CuArray ? collect(a) : a) ≈ b, gradient(h, cu(F), cu(C), cu(i), g), gradient(h, F, C, i, g)))

  y = first(pullback(x -> σ.(x), cu(F)))
  @test y isa CuMatrix{Float32}
  @test collect(y) ≈ σ.(F)

  mul(a, b) = a * b
  m = rand(Bool, 4, 3)
  gx, gm = gradient((x, m) -> sum(ifelse.(m, σ.(x), 0f0)), cu(F), cu(m))
  @test collect(gx) ≈ gradient(x -> sum(ifelse.(m, σ.(x), 0f0)), F)[1]
  @test gm === nothing
  n = rand(1:3, 4, 3)
  @test collect(gradient(x -> sum(mul.(x, cu(n))), cu(F))[1]) ≈ n
  @test gradient(s -> sum(mul.(cu(F), s)), 2f0)[1] ≈ sum(F)
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

@testset "hessian" begin
  # https://github.com/FluxML/Zygote.jl/issues/1348 , #1312 : `hessian`
  # (forward-over-reverse) seeded ForwardDiff `Dual`s and extracted their
  # partials by scalar-indexing, which is disallowed on the GPU.
  x = cu(rand(Float32, 3))
  H = Zygote.hessian(z -> sum(sin.(z)), x)
  @test collect(H) ≈ Zygote.hessian(z -> sum(sin.(z)), collect(x))

  @test collect(Zygote.hessian(z -> norm(sin.(z)), x)) ≈
        Zygote.hessian(z -> norm(sin.(z)), collect(x)) rtol=1e-3

  # masked-mean loss through a dense `A*v` (#1312); the sparse `A*v` variant
  # still hits a cuSPARSE limitation (sparse matrix × ForwardDiff `Dual` vector).
  A = cu(rand(Float32, 10, 4)); mask = cu(round.(rand(Float32, 10)))
  v0 = cu(rand(Float32, 4))
  gpuloss(v) = (sum((A*v) .* mask) / sum(mask) - 1f0)^2
  cpuloss(v) = (sum((collect(A)*v) .* collect(mask)) / sum(collect(mask)) - 1f0)^2
  @test collect(Zygote.hessian(gpuloss, v0)) ≈ Zygote.hessian(cpuloss, collect(v0)) rtol=1e-2
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

  # splatting a vector of arrays (#1687)
  for cat in (vcat, hcat)
    vs = [cu(rand(Float32, 3)) for _ in 1:4]
    g = gradient(vs -> sum(abs2, cat(vs...)), vs)[1]
    @test g isa Vector{<:CUDA.CuArray{Float32}}
    @test Array.(g) ≈ 2 .* Array.(vs)
  end
end

@testset "repeat(x; inner, outer)" begin
  # https://github.com/FluxML/Zygote.jl/issues/1689 : used to scalar-index
  for (x, kw) in [(rand(Float32, 5), (inner=2, outer=3)),
                  (rand(Float32, 5), (inner=(2,1), outer=(1,3))),
                  (rand(Float32, 4, 3), (inner=(2,1,2), outer=(1,3,2))),
                  (rand(Float32, 4, 3, 2), (inner=(2,2,1), outer=(1,1,3)))]
    f(x) = sum(abs2, repeat(x; kw...))
    g_gpu = gradient(f, cu(x))[1]
    @test g_gpu isa CuArray
    @test collect(g_gpu) ≈ gradient(f, x)[1]
  end
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

@testset "sort" begin
  # https://github.com/FluxML/Zygote.jl/issues/1499 : `sort` with the `dims`
  # keyword on a CuArray used to error (`llvmcall` requires the compiler) because
  # Zygote differentiated through the GPU sort kernel. The pullback now scatters
  # through the sorting permutation instead.
  x = cu(rand(Float32, 4))
  @test gradient(x -> sum(sort(x)), x)[1] isa CuArray
  @test collect(gradient(x -> sum(abs2, sort(x)), x)[1]) ≈ gradient(x -> sum(abs2, sort(x)), collect(x))[1]

  X = rand(Float32, 3, 4)
  @test gradient(x -> sum(sort(x, dims=1)), cu(X))[1] isa CuArray
  for dims in (1, 2)
    @test collect(gradient(x -> sum(abs2, sort(x; dims)), cu(X))[1]) ≈
          gradient(x -> sum(abs2, sort(x; dims)), X)[1]
  end
end

@testset "accum structured + GPU" begin
  # https://github.com/FluxML/Zygote.jl/issues/1512 : the `tr` adjoint returns a
  # host-backed `Diagonal{<:Fill}` cotangent; accumulating it against the dense
  # GPU cotangent from `sum(abs2, x)` used to broadcast on the host and
  # scalar-index the GPU array.
  for x0 in (CUDA.zeros(Float32, 2, 2), cu(rand(Float32, 3, 3)))
    g = gradient(x -> sum(abs2, x) - tr(x), x0)[1]
    @test g isa CuArray
    @test collect(g) ≈ gradient(x -> sum(abs2, x) - tr(x), collect(x0))[1]
  end

  # the diagonal is added on the device: no n×n host matrix is built
  # (https://github.com/FluxML/Zygote.jl/issues/1700)
  n = 512
  for T in (Float32, Float64, ComplexF32)
    x = cu(rand(T, n, n))
    D = Diagonal(Fill(T(2), n))
    y = Zygote._gpu_like(x, D)
    @test y isa CuArray{T,2}
    @test collect(y) == collect(D)
    @test collect(Zygote.accum(x, D)) == collect(x) + collect(D) == collect(Zygote.accum(D, x))
  end
  x = cu(rand(Float32, n, n))
  D = Diagonal(Fill(2f0, n))
  Zygote._gpu_like(x, D)
  @test @allocated(Zygote._gpu_like(x, D)) < n^2
  @test eltype(Zygote._gpu_like(x, Diagonal(Fill(2.0, n)))) == Float64

  g = gradient(x -> sum(abs2, x) - tr(x), x)[1]
  @test g isa CuArray
  @test collect(g) ≈ gradient(x -> sum(abs2, x) - tr(x), collect(x))[1]
end

@testset "accum wrapped GPU arrays" begin
  # https://github.com/FluxML/Zygote.jl/issues/1676 : a `Transpose`/`Adjoint` of a
  # GPU array was copied to the host and back before accumulating.
  x = cu(rand(Float32, 3, 3))
  for y in (transpose(x), x', PermutedDimsArray(x, (2, 1)))
    @test Zygote._gpu_like(x, y) === y
    @test Zygote.accum(x, y) isa CuArray
    @test collect(Zygote.accum(x, y)) ≈ collect(Zygote.accum(y, x)) ≈ collect(x) + collect(y)
  end

  W, a, b = cu(rand(Float32, 3, 4)), cu(rand(Float32, 4, 2)), cu(rand(Float32, 3, 2))
  f(W, a, b) = sum(abs2, W * a) + sum(abs2, transpose(W) * b)
  g = gradient(W -> f(W, a, b), W)[1]
  @test g isa CuArray
  @test collect(g) ≈ gradient(W -> f(W, collect(a), collect(b)), collect(W))[1]

  # wrapped GPU cotangent accumulated with the host `Diagonal{<:Fill}` from `tr`
  g = gradient(x -> sum(abs2, x') - tr(x), x)[1]
  @test g isa CuArray
  @test collect(g) ≈ gradient(x -> sum(abs2, x') - tr(x), collect(x))[1]
end

@testset "sparse projection" begin
  # https://github.com/FluxML/Zygote.jl/issues/1313 : the gradient w.r.t. a
  # `CuSparseMatrixCSC` must stay sparse (with the primal's pattern) and match
  # the CPU sparse path, instead of coming back dense.
  M = sprand(10, 10, 0.3)
  x = randn(10)
  g_cpu = gradient(M -> sum((M*x).^2), M)[1]
  @test g_cpu isa SparseMatrixCSC

  Mg = CuSparseMatrixCSC(Float32.(M))
  xg = cu(Float32.(x))
  g_gpu = gradient(M -> sum((M*xg).^2), Mg)[1]
  @test g_gpu isa CuSparseMatrixCSC{Float32}
  @test collect(g_gpu.colPtr) == M.colptr
  @test collect(g_gpu.rowVal) == M.rowval
  @test collect(g_gpu.nzVal) ≈ Float32.(g_cpu.nzval)

  # the result must support sparsity-preserving updates (the original bug was an
  # exception in `update!` because the gradient was dense)
  @test Mg - 0.1f0 * g_gpu isa CuSparseMatrixCSC

  # https://github.com/FluxML/Zygote.jl/issues/1700 : the column index of the
  # stored entries is built on the device when the projection is applied, not
  # (with a host loop) when it is constructed. Non-square, with empty columns.
  A = sprand(Float32, 200, 300, 0.1)
  A[:, 2] .= 0
  A[:, end] .= 0
  dropzeros!(A)
  Ag = CuSparseMatrixCSC(A)
  P = ProjectTo(Ag)
  @test @allocated(ProjectTo(Ag)) < nnz(A)

  D = randn(Float32, size(A))
  @test collect(P(cu(D)).nzVal) ==
        [D[A.rowval[k], c] for c in 1:size(A, 2) for k in A.colptr[c]:(A.colptr[c+1]-1)]
  @test_throws DimensionMismatch P(cu(D[1:end-1, :]))

  v = randn(Float32, 300)
  vg = cu(v)
  gA = gradient(M -> sum(abs2, M * vg), Ag)[1]
  @test gA isa CuSparseMatrixCSC{Float32}
  @test collect(gA.nzVal) ≈ gradient(M -> sum(abs2, M * v), A)[1].nzval
  gv = gradient(x -> sum(abs2, Ag * x), vg)[1]
  @test gv isa CuArray
  @test collect(gv) ≈ gradient(x -> sum(abs2, A * x), v)[1]

  # the linear index must not overflow `Int32` when `m*n > typemax(Int32)`
  m = n = 50_000
  big = CuSparseMatrixCSC(sparse(Int32[3, 49_999, 50_000], Int32[1, 49_998, 50_000], Float32[1, 2, 3], m, n))
  Pbig = ProjectTo(big)
  lin = Base.get_extension(Zygote, :ZygoteCUDAExt)._linear_index.(
    Ref(Pbig.colPtr), Pbig.rowVal, eachindex(Pbig.rowVal), m)
  @test collect(lin) == [3, 49_999 + m * 49_997, 50_000 + m * 49_999]

  # https://github.com/FluxML/Zygote.jl/issues/1700 : a sparse cotangent is
  # looked up in the primal's pattern instead of being densified.
  stored(A, D) = [D[i, j] for (i, j, _) in zip(findnz(A)...)]
  for T in (Float32, Float64, ComplexF32), (m, n) in ((200, 300), (300, 200))
    A = sprand(T, m, n, 0.1)
    A[:, 2] .= 0
    A[:, end] .= 0
    dropzeros!(A)
    D = sprand(T, m, n, 0.15)
    D[:, 5] .= 0
    dropzeros!(D)
    Ag = CuSparseMatrixCSC(A)
    P = ProjectTo(Ag)
    for dx in (D, 2 * A, spzeros(T, m, n)) # differing, same, and empty pattern
      g = P(CuSparseMatrixCSC(dx))
      @test g isa CuSparseMatrixCSC{T}
      @test collect(g.colPtr) == A.colptr
      @test collect(g.rowVal) == A.rowval
      @test collect(g.nzVal) == stored(A, dx)
    end
    @test isempty(ProjectTo(CuSparseMatrixCSC(spzeros(T, m, n)))(CuSparseMatrixCSC(D)).nzVal)
  end

  A = sprand(Float32, 50, 40, 0.2)
  Ag = CuSparseMatrixCSC(A)
  P = ProjectTo(Ag)
  D = sprand(Float64, 50, 40, 0.2)
  @test P(CuSparseMatrixCSC(D)) isa CuSparseMatrixCSC{Float32}
  @test collect(P(CuSparseMatrixCSC(D)).nzVal) == Float32.(stored(A, D))
  nz = randn(Float32, nnz(A)) # same arrays as the primal
  @test collect(P(CuSparseMatrixCSC(Ag.colPtr, Ag.rowVal, cu(nz), size(A))).nzVal) == nz
  @test_throws DimensionMismatch P(CuSparseMatrixCSC(D[1:end-1, :]))

  # row indices not sorted within the columns: falls back to densifying
  D = sprand(Float32, 50, 40, 0.5)
  rowval, nzval = copy(D.rowval), copy(D.nzval)
  for c in 1:size(D, 2)
    k = D.colptr[c]:D.colptr[c+1]-1
    rowval[k], nzval[k] = reverse(rowval[k]), reverse(nzval[k])
  end
  Du = CuSparseMatrixCSC(cu(Int32.(D.colptr)), cu(Int32.(rowval)), cu(nzval), size(D))
  @test collect(P(Du).nzVal) == stored(A, D)

  # no O(m*n) allocation
  m = n = 4096
  A, D = sprand(Float32, m, n, 0.002), sprand(Float32, m, n, 0.002)
  P, Dg = ProjectTo(CuSparseMatrixCSC(A)), CuSparseMatrixCSC(D)
  P(Dg)
  @test CUDA.@allocated(P(Dg)) < sizeof(Float32) * m * n / 10
  @test collect(P(Dg).nzVal) == stored(A, D)

  # sparse cotangents with a pattern of their own reach the projection in `gradient`
  A, B = sprand(Float32, 60, 50, 0.1), sprand(Float32, 60, 50, 0.1)
  Ag, Bg = CuSparseMatrixCSC(A), CuSparseMatrixCSC(B)
  gA = gradient(A -> sum(abs2, A + Bg), Ag)[1]
  @test gA isa CuSparseMatrixCSC{Float32}
  @test collect(gA.nzVal) ≈ gradient(A -> sum(abs2, A + B), A)[1].nzval
end
