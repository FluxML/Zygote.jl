using Zygote: Grads

# ignore / dropgrad / @nograd
@test_deprecated dropgrad(1)
@test_deprecated ignore(1)
@test_deprecated Zygote.@ignore x=1

@test gradient(x -> Zygote.ignore(() -> x*x), 1) == (nothing,)
@test gradient(x -> Zygote.@ignore(x*x), 1) == (nothing,)
@test gradient(1) do x
  y = Zygote.@ignore x
  x * y
end == (1,)

# ---------------------------------------------------------------------------
# Implicit parameters (the `Params`/`Grads` interface) are deprecated.
# See https://github.com/FluxML/Zygote.jl/issues/1561.
# The deprecation funnels through `pullback(f, ::Params)`, so all of the public
# entry points (`gradient`, `withgradient`, `jacobian`, `withjacobian`,
# `pullback` with a `Params` argument) trigger it. Functionality is preserved
# for now; these tests live here so they can be removed together with the
# interface. The underlying machinery (used for mutable structs etc.) is tested
# elsewhere and is *not* deprecated.
# ---------------------------------------------------------------------------

@testset "implicit params deprecation warning" begin
  w = [3.0, 1.0]; x = [2.0, 4.0]
  @test_deprecated pullback(() -> sum(w .* x), Params([w]))

  gs = @test_deprecated gradient(() -> sum(w .* x), Params([w]))
  @test gs[w] == x

  res = @test_deprecated withgradient(() -> sum(w .* x), Params([w]))
  @test res.val == sum(w .* x)
  @test res.grad[w] == x

  js = @test_deprecated jacobian(() -> w .* x, Params([w]))
  @test js[w] == [x[1] 0.0; 0.0 x[2]]
end

@testset "Params" begin
  @testset "in" begin
    w = rand(2,3)
    b = rand(2)
    ps = Params([w])
    @test w ∈ ps
    @test b ∉ ps
  end

  @testset "delete!" begin
    w = rand(2,3)
    b = rand(2)
    ps = Params([w,b])
    delete!(ps, b)
    @test length(ps.order) == length(ps.params) == 1
    @test first(ps.order) == first(ps.params) == w
  end

  @testset "copy!" begin
    x = [0,0]
    ps = Params([x])
    copy!(ps, [1, 2])
    @test x == [1, 2]

    x = [0,0]
    y = [0]
    ps = Params([x, y])
    copy!(ps, [1, 2, 3])
    @test x == [1, 2]
    @test y == [3]

    ps = Params([[1,2]])
    x = [0, 0]
    copy!(x, ps)
    @test x == [1, 2]

    ps = Params([[1,2], [3]])
    x = [0, 0, 0]
    copy!(x, ps)
    @test x == [1, 2, 3]
  end

  @testset "broadcast" begin
    x, y = [1,2], [1]
    ps = Params([x, y])
    @test length.(ps) == length.([x, y]) # 617
    @test size.(ps, 1) == [2, 1]
    @test all(Params([[1,1]]) .== Params([[1,1]]))

    @test_throws ArgumentError gradient(() -> sum(sum.(ps)), ps)
  end

  @testset "indexing" begin
    x, y = [1,2], [1]
    ps = Params([x, y])
    @test ps[1] === x
    @test ps[2] === y
    @test ps[1:2] == [x, y]
  end

  @testset "comparison" begin
    x, y = [1,2], [1]
    ps1 = Params([x, y])
    ps2 = Params([x, y])
    ps3 = Params([y, x])
    @test ps1 == ps2
    @test ps1 != ps3  # comparison is order dependent
  end

  @testset "set interface" begin
    x, y, z = [1,2], [1], [3,4]
    ps1 = Params([x, y])
    ps2 = Params([z])
    ps3 = Params([y, z])

    ps = union(ps1, ps2)
    @test ps isa Params
    @test issetequal(ps, Set([x,y,z]))
    ps = union(ps1, ps3)
    @test ps isa Params
    @test issetequal(ps, Set([x,y,z]))

    ps = intersect(ps1, ps2)
    @test ps isa Params
    @test issetequal(ps, Set())

    ps = intersect(ps1, ps3)
    @test ps isa Params
    @test issetequal(ps, Set([y]))
  end

  @testset "constructor with empty args" begin
    @test length(Params()) == 0
    @test length(Params(())) == 0
    @test length(Params([])) == 0
  end
end

@testset "Grads" begin
  @testset "algebra" begin
    w, b = rand(2), rand(2)
    x1, x2 = rand(2), rand(2)

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
    gs3 .+= IdDict(p => randn(size(p)) for p in keys(gs3))
    @test gs3 isa Grads

    @test_throws ArgumentError gs1 .+ gs4
  end

  @testset "copy" begin
    w, b = rand(2), rand(2)
    x1, x2 = rand(2), rand(2)

    _, back = pullback(() -> sum(w .* x1), Params([w]))

    g1 = back(1)
    g1_w = g1[w]
    g2 = back(nothing)
    @test isnothing(g1[w])
    @test isnothing(g2[w])

    g3 = back(1) |> copy
    g4 = back(nothing)
    @test !isnothing(g3[w])
    @test g3[w] == g1_w
    @test isnothing(g4[w])

    g3_copy = copy(g3)
    @test collect(g3_copy) == collect(g3)
  end

  @testset "map and broadcast" begin
    w = rand(2)
    x1 = rand(2)
    x2 = rand(2)

    gs1 = gradient(() -> sum(w .* x1), Params([w]))
    gs2 = gradient(() -> sum(w .* x2), Params([w]))

    @test map(x -> zeros(2), gs1) isa Grads

    gs11 = map(x -> clamp.(x, -1e-5, 1e-5), gs1)
    @test gs11 isa Grads
    @test all(abs.(gs11[w]) .<= 1e-5)

    @test (x -> zeros(2)).(gs1) isa Grads
  end

  @testset "dictionary interface" begin
    w, b, x = rand(2), rand(2), rand(2)
    ps = Params([w, b])
    gs = gradient(() -> sum(tanh.(w .* x .+ b)), ps)

    @test issetequal(keys(gs), ps)
    @test length(values(gs)) == 2
    @test length(pairs(gs)) == 2
    k, v = first(pairs(gs))
    @test k === first(ps)
    @test v === gs[first(ps)]
  end

  @testset "merge" begin
    w1, b1, x1 = rand(2), rand(2), rand(2)
    ps1 = Params([w1, b1])
    gs1 = gradient(() -> sum(tanh.(w1 .* x1 .+ b1)), ps1)

    w2, b2, x2 = rand(2), rand(2), rand(2)
    ps2 = Params([w2, b2])
    gs2 = gradient(() -> sum(tanh.(w2 .* x2 .+ b2)), ps2)

    w3, b3, x3 = rand(2), rand(2), rand(2)
    ps3 = Params([w3, b3])
    gs3 = gradient(() -> sum(tanh.(w3 .* x3 .+ b3)), ps3)

    # merging with a single other Grads object
    keys1 = keys(gs1)
    values1 = values(gs1)
    gs_merged = merge!(gs1, gs2)
    @test issetequal(keys(gs_merged),  union(keys1, keys(gs2)))
    @test issetequal(values(gs_merged), union(values1, values(gs2)))
    @test length(pairs(gs_merged)) == 4

    # merging with multiple other Grads objects
    gs_merged = merge!(gs1, gs2, gs3)
    @test issetequal(keys(gs_merged), union(keys1, keys(gs2), keys(gs3)))
    @test issetequal(values(gs_merged), union(values1, values(gs2), values(gs3)))
    @test length(pairs(gs_merged)) == 6
  end

  @testset "iteration" begin
    w, b, x = rand(2), rand(2), rand(2)
    ps = Params([w, b])
    gs = gradient(() -> sum(tanh.(w .* x .+ b)), ps)

    # value-based iteration
    foreach(x -> clamp!(x, -1e-5, 1e-5), gs)
    @test all(abs.(gs[w]) .<= 1e-5)
    @test all(abs.(gs[b]) .<= 1e-5)
  end

  @testset "Params nesting" begin
    struct Dense{F,T,S}
      W::T
      b::S
      σ::F
    end

    (d::Dense)(x) = d.σ.(d.W * x .+ d.b)
    d = Dense(ones(Float32, 3,3), zeros(Float32, 3), identity)
    ps = Zygote.Params([d.W, d.b])
    r = ones(Float32, 3,3)

    gs = gradient(ps) do
      p, pb = pullback(ps) do
        sum(d(r))
      end
      g = pb(p)
      sum(g[d.W]) # + sum(g[d.b])
    end

    @test gs[d.W] ≈ fill(81f0, (3,3))

    # Test L2
    l2g = gradient(ps) do
      sum(sum(x .^ 2) for x in ps)
    end
    @test l2g[d.W] ≈ fill(2.f0, size(d.W))
    @test l2g[d.b] ≈ fill(0.f0, size(d.b))

    # Can be safely removed - creating Params within
    # gradient calls may break between releases.
    sgs = gradient(ps) do
      sum(sum(x) for x in Zygote.Params([d.W, d.b]))
    end
    @test sgs[d.W] ≈ fill(1.f0, size(d.W))
    @test sgs[d.b] ≈ fill(1.f0, size(d.b))
  end
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

@testset "Dict getindex with implicit params" begin
  d = Dict{String, Vector{Float64}}("key"=>ones(4))
  fn() = d["key"][2]
  result1 = gradient(fn, Params([d["key"]]))[d["key"]]

  x = d["key"]
  fn2() = x[2]
  result2 = gradient(fn2, Params([x]))[x]

  @test result1 == result2
end

@testset "implicit params with structs and closures" begin
  struct Layer{T}
    W::T
  end
  (f::Layer)(x) = f.W * x

  W = [1 0; 0 1]
  x = [1, 2]

  y, back = pullback(() -> W * x, Params([W]))
  @test y == [1, 2]
  @test back([1, 1])[W] == [1 2; 1 2]

  layer = Layer(W)

  y, back = pullback(() -> layer(x), Params([W]))
  @test y == [1, 2]
  @test back([1, 1])[W] == [1 2; 1 2]

  @test gradient(() -> sum(W * x), Params([W]))[W] == [1 2; 1 2]
  y, gr = withgradient(() -> sum(W * x), Params([W]))
  @test y == 3
  @test gr[W] == [1 2; 1 2]

  let
    p = [1]
    θ = Zygote.Params([p])
    θ̄ = gradient(θ) do
      p′ = (p,)[1]
      p′[1]
    end
    @test θ̄[p][1] == 1
  end
end

@testset "#300" begin
  t = (rand(2, 2), rand(2, 2))
  ps = Params(t)
  gs = gradient(()->sum(t[1]), ps)
  @test gs[t[1]] == ones(2, 2)
end

@testset "implicit params: mutable accum_param" begin
  mutable struct MutAccum{T}; x::T; end
  struct ImmAccum{T}; x::T; end

  y1 = [3.0]
  y2 = (MutAccum(y1),)
  y3 = (ImmAccum(y1),)
  @test gradient(() -> sum(y2[1].x)^2, Params([y1]))[y1] == [6.0]
  @test gradient(() -> sum(y3[1].x)^2, Params([y1]))[y1] == [6.0]

  @test gradient(() -> sum(y2[1].x .+ y2[1].x)^3, Params([y1]))[y1] == [216.0]
  @test gradient(() -> sum(y3[1].x .+ y3[1].x)^3, Params([y1]))[y1] == [216.0]

  i1 = 1
  @test gradient(() -> sum(y2[i1].x .+ y2[1].x)^3, Params([y1]))[y1] == [216.0]
  @test gradient(() -> sum(y3[i1].x .+ y3[1].x)^3, Params([y1]))[y1] == [216.0]
end

@testset "implicit params: broadcasted($op, Array, Bool)" for op in (+,-,*)
  @testset "with $bool and sizes $s" for s in ((4,), (2,3)), bool in (true,false)
    r = rand(Int8, s) .+ 0.0
    z = fill(bool, s) .+ 0.0
    gfun(args...) = gradient(() -> sum(op.(args...)), Params(filter(a->a isa Array, collect(args))))

    g = gfun(r, z)
    gres = gfun(r, bool)
    @test gres[r] == g[r]

    g = gfun(z, r)
    gres = gfun(bool, r)
    @test gres[r] == g[r]
  end
end

@testset "implicit params: Dict iteration" begin
  # https://github.com/FluxML/Zygote.jl/issues/1065
  function sumkv(d)
    s = zero(d["c"])
    for (k, v) in d
      s += v
      k == :b && (s += v)
    end
    return sum(s)
  end

  function sumvals(d)
    s = zero(d["c"])
    for v in values(d)
      s += v
    end
    return sum(s)
  end

  d_arr = Dict(:a => [3], :b => [4], "c" => [5])
  ps = d_arr |> values |> collect |> Params

  grads = gradient(() -> sumkv(d_arr), ps)
  @test (grads[d_arr[:a]], grads[d_arr[:b]], grads[d_arr["c"]]) == ([1], [2], [1])

  grads = gradient(() -> sumvals(d_arr), ps)
  @test (grads[d_arr[:a]], grads[d_arr[:b]], grads[d_arr["c"]]) == ([1], [1], [1])
end
