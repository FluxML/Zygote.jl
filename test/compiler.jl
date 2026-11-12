using LinearAlgebra
using ForwardDiff
using Zygote: pullback, @adjoint, Context

macro test_inferred(ex)
  :(let res = nothing
    @test begin
      res = @inferred $ex
      true
    end
    res
  end) |> esc
end

function trace_contains(st, func, file, line)
  any(st) do fr
    func in (nothing, fr.func) && endswith(String(fr.file), file) && fr.line == line
  end
end

bad(x) = x
const bad_def_line = (@__LINE__) + 1
@adjoint bad(x) = x, Δ -> error("bad")

const bad_call_line = (@__LINE__) + 3
function badly(x)
  x = x + 1
  x = bad(x)
  return x
end

y, back = pullback(badly, 2)
@test y == 3
@test_throws Exception back(1)
bt = try back(1) catch e stacktrace(catch_backtrace()) end

@test trace_contains(bt, nothing, "compiler.jl", bad_def_line)
@test trace_contains(bt, nothing, "compiler.jl", bad_call_line)
# Julia 1.14 builds frames from the `DebugInfo` tree, which names this frame after the
# `Pullback` method rather than the primal.
@test trace_contains(bt, :badly, "compiler.jl", bad_call_line) broken=VERSION >= v"1.14-"

# Type inference checks

y, back = @test_inferred pullback(*, 2, 3)
@test_inferred(back(1))

_sincos(x) = sin(cos(x))

y, back = @test_inferred pullback(_sincos, 0.5)
@test_inferred back(1)

f(x) = 3x^2 + 2x + 1

y, back = @test_inferred pullback(f, 5)
@test y == 86
@test_inferred(back(1))

y, back = @test_inferred pullback(Core._apply, +, (1, 2, 3))
@test_inferred back(1)

# #1697: pullbacks of code with branches are concretely typed, unless the type gets large
branchy(x) = x > 0 ? sin(x) * x : cos(x)
y, back = @test_inferred Zygote._pullback(Context(), branchy, 0.5)
@test isconcretetype(fieldtype(typeof(back), :t))
@test back(1.0)[2] ≈ cos(0.5) * 0.5 + sin(0.5)
@test Zygote.branchy_pullback(Tuple{typeof(sin)}, (1, 2.0), Val(false)) isa Zygote.Pullback{Tuple{typeof(sin)}, Tuple{Int,Float64}}
@test Zygote.branchy_pullback(Tuple{typeof(sin)}, ntuple(_ -> (1, 2.0), 30), Val(false)) isa Zygote.Pullback{Tuple{typeof(sin)}, Any}
@test Zygote.branchy_pullback(Tuple{typeof(sin)}, ntuple(_ -> (1, 2.0), 30), Val(true)) isa Zygote.Pullback{Tuple{typeof(sin)}, Tuple}

# loops record into stacks typed after their values, which the backward pass picks up
function looppow(x, n)
  r = one(x)
  for _ in 1:n
    r *= x
  end
  return r
end
y, back = Zygote._pullback(Context(), looppow, 2.0, 3)
@test Zygote.StackRef in fieldtypes(fieldtype(typeof(back), :t))
tback = Zygote.typed_pullback(back)
@test isconcretetype(fieldtype(typeof(tback), :t))
@test @inferred(Tuple{Nothing,Union{Nothing,Float64},Nothing}, tback(1.0)) == (nothing, 12.0, nothing)
@test back(1.0) == (nothing, 12.0, nothing)
@test gradient(looppow, 2.0, 0) == (nothing, nothing)
@test gradient(x -> gradient(x -> looppow(x, 3), x)[1], 2.0) == (12.0,)
bigloop(x, n) = (a = sin(x) * cos(x) * tan(x) * exp(x) * log(x) * sqrt(x); looppow(a, n))
y, back = Zygote._pullback(Context(), bigloop, 2.0, 2)
@test back(1.0)[2] ≈ ForwardDiff.derivative(x -> bigloop(x, 2), 2.0)

# a stack whose values change type falls back to `Vector{Any}`
function mixedloop(xs)
  s = 0.0
  for x in xs
    s += x isa Int ? x : sin(x)
  end
  return s
end
@test gradient(mixedloop, Any[1, 2.0, 3, 4f0])[1] == [1, cos(2.0), 1, cos(4f0)]

# TODO fix bcast inference
# bcast(x) = x .* 5
# y, back = @test_inferred pullback(bcast, [1,2,3])
# @test_inferred back([1,1,1])

foo = let a = 4
  x -> x*a
end

@test_inferred gradient(f -> f(5), foo)

getx(x) = x.x
y, back = @test_inferred pullback(getx, (x=1,y=2.0))
@test_inferred back(1)

y, back = @test_inferred pullback(x->x[1], (5,:a))
@test_inferred back(1)

y, back = @test_inferred pullback(((a,b),) -> a, (5, 10))
@test_inferred back(1)

# testcase for issue #808
# testing that methods(Base.show) does not throw. Having something more specific would be too fragile
show_err = try
  buf = IOBuffer()
  Base.show(buf, methods(Base.show))
  nothing
catch ex
  ex
end
@test show_err === nothing

struct Funky
    x
    y
end

@testset "issue #851" begin
  f = Funky(1, 1);
  function Base.getproperty(f::Funky, i::Symbol)
      return 2
  end
  @test getproperty(f, :x) == 2
  @test getfield(f, :x) == 1

  y, pb = Zygote._pullback(getproperty, f, :x)
  @test y == 2
  @test pb(1) == (nothing, nothing, nothing)
  y, pb = Zygote._pullback((f, x) -> getproperty(f, x), f, :x)
  @test y == 2
  @test pb(1) == (nothing, nothing, nothing)
  y, pb = Zygote._pullback(getfield, f, :x)
  @test y == 1
  @test pb(1) == (nothing, (x = 1, y = nothing), nothing)
end

@testset "issue #922" begin
  # checks whether getproperty gets accumulated correctly
  # instead of defining a test function as in the issue, compare the two pullbacks
  function two_svds(X::StridedMatrix{<:Union{Real, Complex}})
    return svd(X).U * svd(X).V'
  end

  function one_svd(X::StridedMatrix{<:Union{Real, Complex}})
    F = svd(X)
    return F.U * F.V'
  end

  Δoutput = randn(3,2)
  X = randn(3,2)

  d_two = Zygote.pullback(two_svds, X)[2](Δoutput)
  d_one = Zygote.pullback(one_svd, X)[2](Δoutput)
  @test d_one == d_two
end

# this test fails if adjoint for literal_getproperty is added
# https://github.com/FluxML/Zygote.jl/issues/922#issuecomment-804128905
@testset "overloaded getproperty" begin
  struct MyStruct
      a
      b
  end
  Base.getproperty(ms::MyStruct, s::Symbol) = s === :c ? ms.a + ms.b : getfield(ms, s)
  sumall(ms::MyStruct) = ms.a + ms.b + ms.c

  ms = MyStruct(1, 2)
  @test Zygote.gradient(sumall, ms) == ((a = 2, b = 2),)
end

using ChainRulesCore

function _Gaussian(suffix::Symbol)
    name = gensym(Symbol(:Gaussian_, suffix))
    return @eval begin
        struct $name{Tm, TP}
            m::Tm
            P::TP
        end
        $name
    end
end

module MyMod
  const C = 1
  func(a, b) = a * b
end

@eval usesmod(x) = Base.getproperty($MyMod, :func)(x, Base.getproperty($MyMod, :C))

@testset "inference for `getproperty`" begin
    Gaussian = _Gaussian(:getproperty)
    g = Gaussian(randn(3), randn(3, 3))
    y_explicit, back_explicit = @inferred pullback(x -> x.m, g)
    y_implicit, back_implicit = @inferred pullback(x -> x.m, Context{true}(nothing), g)
    @test y_explicit == y_implicit == getfield(g, :m)

    ∇args = ((m = [1.0, 0.0, 0.0], P = nothing),)
      # This type instability is due to the handling of non-bitstypes in `accum_param`
      @test Base.return_types(back_implicit, Tuple{Vector{Float64}}) == Any[Union{Tuple{Nothing}, typeof(∇args)}]
      # But the same should infer if implicit parameters are disabled
      @test Base.return_types(back_explicit, Tuple{Vector{Float64}}) == Any[typeof(∇args)]

    @test back_explicit([1., 0, 0]) == back_implicit([1., 0, 0]) == ∇args

    Base.getproperty(g::Gaussian, s::Symbol) = 2getfield(g, s)
    y, back = pullback(x -> x.m, g)
    @test y == 2getfield(g, :m)
    @test back([1., 0, 0]) == ((m = [2.0, 0.0, 0.0], P = nothing),)


    Gaussian = _Gaussian(:pullback)
    g = Gaussian(randn(3), randn(3, 3))
    y, back = @inferred pullback(x -> x.m, g)

    Zygote._pullback(::Zygote.AContext, ::typeof(getproperty), g::Gaussian, s::Symbol) = 3getfield(g, s), Δ -> (nothing, (; ((:m, :P) .=> nothing)..., s => 3Δ), nothing)
    y, back = pullback(x -> x.m, g)
    @test y == 3getfield(g, :m)
    @test back([1., 0, 0]) == ((m = [3.0, 0.0, 0.0], P = nothing),)


    Gaussian = _Gaussian(:rrule)
    g = Gaussian(randn(3), randn(3, 3))
    y, back = @inferred pullback(x -> x.m, g)

    ChainRulesCore.rrule(::typeof(getproperty), g::Gaussian, s::Symbol) = 4getfield(g, s), Δ -> (NoTangent(), Tangent{typeof(g)}(; s => 4Δ), NoTangent())
    y, back = pullback(x -> x.m, g)
    @test y == 4getfield(g, :m)
    @test back([1., 0, 0]) == ((m = [4.0, 0.0, 0.0], P = nothing),)


    Gaussian = _Gaussian(:bitstype)
    g = Gaussian(randn(), randn())
    y, back = @inferred pullback(x -> x.m, g)
    @test y == getfield(g, :m)
    @test @inferred(back(1.0)) == ((m = 1.0, P = nothing),)


    # Const properties on modules should be lowered as-is (not differentiated)
    @test gradient(usesmod, 1)[1] == 1.0 broken=VERSION >= v"1.12"
    @test @inferred(gradient(usesmod, 1))[1] == 1.0 broken=VERSION >= v"1.12"
end

@testset "`getproperty` generator is not compiled per field name" begin
    Gaussian = _Gaussian(:specialization)
    @test gradient(g -> g.m * g.P, Gaussian(2.0, 3.0))[1] == (m = 3.0, P = 2.0)
    m = only(methods(Zygote._generate_literal_getproperty))
    @test !any(Base.specializations(m)) do mi
        Base.unwrap_unionall(mi.specTypes).parameters[end] <: Type{<:Val}
    end
end

@testset "non-differentiable primitives error clearly" begin
    # Regression: the callable pullback's non-differentiable branch referenced a
    # stale argument name `j` (the generator stub names it `methodinstance`), so it
    # leaked `UndefVarError: j` instead of a clear message. Introduced with the
    # world-age generated-function rework; surfaced by #194, #252, #467, #619, #1156.
    _, pb = Zygote._pullback(getglobal, Base.MathConstants, :eulergamma)
    err = try; pb(1.0); nothing; catch e; e; end
    @test err isa ErrorException
    @test occursin("Non-differentiable function", err.msg)
    @test !occursin("UndefVarError", err.msg)
end

# issue 897
@test gradient(x -> sum(norm, collect(eachcol(x))), ones(3, 400))[1] ≈ fill(0.5773502691896258, 3, 400)

# Tests adapted from https://github.com/dfdx/Umlaut.jl/pull/35
@eval _has_boundscheck(x) = ifelse($(Expr(:boundscheck)), 2x, x)

@testset "Meta Expr handling" begin
  y, (dx,) = withgradient(_has_boundscheck, 1)
  @test y == 2
  @test dx == 2
end

# issue 1118 & 1380
function f_1380(x)
    if rand(Bool)
        return x
    else
        return 2x
    end

    # unreachable
    return nothing
end

@testset "unreachable block" begin
    y, back = Zygote.pullback(f_1380, 1.)
    # There should not be a compiler error
    local g
    @test_nowarn g = back(1.)
    @test only(g) ∈ (1., 2.)
end

function throws_and_catches_if_x_negative(x,y)
    z = x + y
    try
        if x < 0.
            throw(DomainError("x is negative"))
        end
        z = 2z + x + y
    catch err
        @error "something went wrong" exception=(err,catch_backtrace())
    end
    return 3z
end

function try_catch_finally(cond, x)

    try
        x = 2x
        cond && throw(DomainError())
    catch
        x = 2x
    finally
        x = 3x
    end

    x
end

function try_catch_else(cond, x)
    x = 2x

    try
        x = 2x
        cond && throw(nothing)
    catch
        x = 3x
    else
        x = 2x
    end

    x
end

@testset "try/catch" begin
    @testset "happy path (nothrow)" begin
        res, (dx,dy) = withgradient(throws_and_catches_if_x_negative, 1., 2.)
        @test res == 3 * (2 * (1. + 2.) + 1. + 2.)
        @test dx == 3. * (2. + 1.)
        @test dy == 3. * (2. + 1.)
    end

    @testset "try/catch/finally" begin
        res, (_, dx,) = withgradient(try_catch_finally, false, 1.)
        @test res == 6.
        @test dx == 6.

        res, pull = pullback(try_catch_finally, true, 1.)
        @test res == 12.
        @test_throws ErrorException pull(1.)
        err = try pull(1.) catch ex; ex end
        @test occursin("Can't differentiate function execution in catch block", string(err))
    end

        @testset "try/catch/else" begin
            @test Zygote.gradient(try_catch_else, false, 1.0) == (nothing, 8.0)
            @test_throws ErrorException Zygote.gradient(try_catch_else, true, 1.0)
        end

    function foo_try(f)
      y = 1
      try
        y = f()
      catch
        y
      end
      y
    end

    g, = gradient(x -> foo_try(() -> x), 1) # 1
    @test g == 1.

    vy, pull = pullback(foo_try, () -> 0//0) # bypass because of expr
    @test vy === 1
    @test_throws ErrorException pull(1.)

    err = try pull(1.) catch ex; ex end
    @test occursin("Can't differentiate function execution in catch block", string(err))
end

@testset "refresh" begin
  # https://github.com/FluxML/Zygote.jl/issues/1599
  # `Zygote.refresh()` must pick up an `rrule` defined *after* a gradient has
  # already been computed (and cached) for the same signature.
  function refresh1599(x::AbstractArray)
    y = similar(x)
    for i in eachindex(x)
      y[i] = x[i] > 0 ? x[i]^2 : zero(eltype(x))
    end
    sum(y)
  end

  x = [1.7, 2.1, -1.5, -0.1, 3.1]
  # Mutation is not differentiable, so this fails and caches the failing pullback.
  @test_throws Exception Zygote.gradient(refresh1599, x)

  function ChainRulesCore.rrule(::typeof(refresh1599), x)
    J = [xᵢ > 0 ? 2xᵢ : zero(xᵢ) for xᵢ in x]
    refresh1599_pullback(v) = NoTangent(), J * unthunk(v)
    refresh1599(x), refresh1599_pullback
  end

  Zygote.refresh()

  @test Zygote.gradient(refresh1599, x)[1] == [3.4, 4.2, 0.0, 0.0, 6.2]
end

# https://github.com/FluxML/Zygote.jl/issues/1680
# On Julia 1.12+ the generators' own world doesn't see bindings defined after Zygote.
later_helper_1680(x) = 2x
calls_later_helper_1680(x) = later_helper_1680(x)

@testset "constant globals are not tracked" begin
  _, back = Zygote._pullback(Context(), calls_later_helper_1680, 1.0)
  @test back.t isa Tuple{Zygote.Pullback{Tuple{typeof(later_helper_1680), Float64}}}
end

@testset "gradients through code compiled during precompilation" begin
  mktempdir() do dir
    mkpath(joinpath(dir, "ZygotePrecompile1680", "src"))
    write(joinpath(dir, "ZygotePrecompile1680", "src", "ZygotePrecompile1680.jl"), """
      module ZygotePrecompile1680
      using Zygote
      helper(x) = x
      f(x) = 2 .* helper(x)
      Zygote.gradient(x -> sum(f(x)), [1.0, 2.0])
      end
      """)
    script = """
      pushfirst!(DEPOT_PATH, $(repr(joinpath(dir, "depot"))))
      push!(LOAD_PATH, $(repr(dir)))
      using ZygotePrecompile1680, Zygote
      print(Zygote.gradient(x -> sum(abs2, ZygotePrecompile1680.f(x)), [1.0, 2.0]))
      """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
    @test readchomp(cmd) == "([8.0, 16.0],)"
  end
end

@testset "precompile workload covers keyword arguments and tuple loops" begin
  script = """
    using Zygote
    m = only(filter(methods(Zygote._pullback)) do m
      m.sig == Tuple{typeof(Zygote._pullback), Zygote.AContext, Any, Vararg{Any}}
    end)
    sigs = [Base.unwrap_unionall(mi.specTypes).parameters for mi in Base.specializations(m)]
    print(any(p -> p[3] === typeof(Core.kwcall), sigs), " ",
          any(p -> p[3] === typeof(iterate) && length(p) > 3 && p[4] <: Tuple, sigs))
    """
  cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
  @test readchomp(cmd) == "true true"
end

function branchy_decomposition(x, n)
  r = one(x)
  while n > 0
    n -= 1
    r *= x > 0 ? x : -x
  end
  return r
end

@testset "decomposition cache" begin
  T = Tuple{typeof(branchy_decomposition), Float64, Int}
  # closures in this body would otherwise move the world between calls
  w = Base.get_world_counter()
  decompose(side, world = w) = Zygote._generate_pullback_via_decomposition(T, world, side)
  sides = (:forward, :backward)

  Zygote.clear_decompositions!()
  uncached = [string(decompose(side)[2]) for side in sides]
  @test Zygote.cached_decomposition(T, w) !== nothing
  @test [string(decompose(side)[2]) for side in sides] == uncached

  @testset "callers get their own copy" begin
    m1, _ = decompose(:forward)
    m2, _ = decompose(:forward)
    @test m1.code !== m2.code
    for side in sides
      _, ir = decompose(side)
      for (_, st) in ir
        Meta.isexpr(st.expr, :call) && empty!(st.expr.args)
      end
      push!(ir, :(rewritten))
    end
    @test [string(decompose(side)[2]) for side in sides] == uncached
  end

  @testset "worlds" begin
    w0 = w
    Core.eval(@__MODULE__, :(bump_decomposition_world() = 1))
    w1 = Base.get_world_counter()
    @test w1 > w0

    Zygote.clear_decompositions!()
    decompose(:forward, w1)
    @test Zygote.cached_decomposition(T, w1) !== nothing
    @test Zygote.cached_decomposition(T, w0) === nothing
    @test Zygote.cached_decomposition(T, w1 + 1) === nothing

    # an older world neither reads nor replaces what is cached for a newer one
    @test string(decompose(:backward, w0)[2]) == uncached[2]
    @test Zygote.decompositions_world[] == w1
    @test Zygote.cached_decomposition(T, w1) !== nothing

    # a newer world drops the older entries
    Core.eval(@__MODULE__, :(bump_decomposition_world() = 2))
    w2 = Base.get_world_counter()
    decompose(:forward, w2)
    @test Zygote.decompositions_world[] == w2
    @test length(Zygote.decompositions) == 1
  end

  @testset "concurrent generators" begin
    Zygote.clear_decompositions!()
    tasks = [Threads.@spawn string(decompose(side)[2]) for side in sides for _ in 1:8]
    @test fetch.(tasks) == repeat(uncached, inner = 8)
  end

  @testset "gradients across contexts" begin
    for _ in 1:2
      @test gradient(branchy_decomposition, 2.0, 3) == (12.0, nothing)
      @test gradient(branchy_decomposition, -2.0, 3) == (-12.0, nothing)
      w = [2.0]
      @test gradient(() -> branchy_decomposition(w[1], 3), Params([w]))[w] == [12.0]
    end
  end
end

@testset "predecessor_ids" begin
  IRTools = Zygote.IRTools
  reference(ir) = [[c.id for c in IRTools.predecessors(b)] for b in IRTools.blocks(ir)]

  function elseifs(x, y)
    if x > 1
      return y
    elseif x > 0 && y > 0
      y += 1
    elseif y < -1
      y -= 1
    end
    for i in 1:3
      i == 2 && continue
      y += i
      y > 10 && break
    end
    return y
  end
  for (f, Ts...) in ((branchy_decomposition, Float64, Int), (elseifs, Float64, Float64),
                     (sum, Vector{Float64}), (sort, Vector{Int}))
    ir = IRTools.IR(typeof(f), Ts...)
    @test Zygote.predecessor_ids(ir) == reference(ir)
  end

  # fall-through edges and several branches from one block to the same target
  ir = IRTools.IR()
  x = IRTools.argument!(ir)
  b2 = IRTools.block!(ir)
  b3 = IRTools.block!(ir)
  b4 = IRTools.block!(ir)
  IRTools.branch!(IRTools.block(ir, 1), 3, unless = x)
  IRTools.branch!(IRTools.block(ir, 1), 3, unless = x)
  IRTools.branch!(b2, 4, unless = x)
  IRTools.return!(b3, x)
  IRTools.return!(b4, x)
  @test reference(ir) == [[], [1], [1, 2], [2]]
  @test Zygote.predecessor_ids(ir) == reference(ir)
end

@eval function many_blocks(x)
  $(Expr(:block, [:(x > $(i / 100) && (x = x * 1.01 + 0.001)) for i in 1:100]...))
  return x
end

@testset "gradients through many basic blocks" begin
  for x in (0.0, 0.5, 1.2, 3.0)
    @test gradient(many_blocks, x)[1] ≈ ForwardDiff.derivative(many_blocks, x)
  end
end

function reassign(x, y, c)
  a = x
  if c
    a = y
  end
  sum(a) + sum(x)
end

function swaps(x, y, n)
  a, b = x, y
  for _ in 1:n
    a, b = b, a
  end
  sum(a) + 2sum(b) + sum(x)
end

@testset "sums of cotangents only on the branch taken (#1746)" begin
  @test gradient(reassign, ones(2), ones(3), true) == (ones(2), ones(3), nothing)
  @test gradient(reassign, ones(2), ones(3), false) == (2ones(2), nothing, nothing)
  @test gradient(swaps, ones(2), ones(3), 2) == (2ones(2), 2ones(3), nothing)
  @test gradient(swaps, ones(2), ones(3), 3) == (3ones(2), ones(3), nothing)
end
