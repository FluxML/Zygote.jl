using Zygote, Test
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

trace_contains(st, func, file, line) = any(st) do fr
  func in (nothing, fr.func) && endswith(String(fr.file), file) &&
    fr.line == line
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
if VERSION <= v"1.6-" || VERSION >= v"1.10-"
  @test trace_contains(bt, :badly, "compiler.jl", bad_call_line)
else
  @test_broken trace_contains(bt, :badly, "compiler.jl", bad_call_line)
end

# Type inference checks

Zygote.refresh()

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
usesmod2(x) = Base.getproperty(MyMod, :func)(x, Base.getproperty(MyMod, :C))

@testset "inference for `getproperty`" begin
    Gaussian = _Gaussian(:getproperty)
    g = Gaussian(randn(3), randn(3, 3))
    y_explicit, back_explicit = @inferred pullback(x -> x.m, g)
    y_implicit, back_implicit = @inferred pullback(x -> x.m, Context{true}(nothing), g)
    @test y_explicit == y_implicit == getfield(g, :m)

    ∇args = ((m = [1.0, 0.0, 0.0], P = nothing),)
    if VERSION > v"1.7-"
      # This type instability is due to the handling of non-bitstypes in `accum_param`
      @test Base.return_types(back_implicit, Tuple{Vector{Float64}}) == Any[Union{Tuple{Nothing}, typeof(∇args)}]
      # But the same should infer if implicit parameters are disabled
      @test Base.return_types(back_explicit, Tuple{Vector{Float64}}) == Any[typeof(∇args)]
    end
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
    @test @inferred gradient(usesmod, 1)[1] == 1.0
    @test @inferred gradient(usesmod2, 1)[1] == 1.0
end

# issue 897
@test gradient(x -> sum(norm, collect(eachcol(x))), ones(3, 400))[1] ≈ fill(0.5773502691896258, 3, 400)
