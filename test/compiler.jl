using Zygote, Test
using Zygote: pullback, @adjoint

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
@adjoint bad(x) = x, Δ -> error("bad")

function badly(x)
  x = x + 1
  x = bad(x)
  return x
end

y, back = pullback(badly, 2)
@test y == 3
@test_throws Exception back(1)
bt = try back(1) catch e stacktrace(catch_backtrace()) end

@test trace_contains(bt, nothing, "compiler.jl", 20)
if VERSION >= v"1.6-"
  @test_broken trace_contains(bt, :badly, "compiler.jl", 24)
else
  @test trace_contains(bt, :badly, "compiler.jl", 24)
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
buf = IOBuffer()
Base.show(buf, methods(Base.show))
str_repr = String(take!(buf))
@test !isempty(str_repr)

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

# issue 897
@test gradient(x -> sum(norm, collect(eachcol(x))), ones(3, 400))[1] ≈ fill(0.5773502691896258, 3, 400)
