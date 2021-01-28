using Zygote, Test
using Zygote: Params, gradient, forwarddiff

@testset "gradient checkpointing" begin

    @testset "checkpointed does not change pullback value" begin
        for setup in [
                (f=identity, args = (1.0,), dy=1.0),
                (f=max, args = (1.0,2, 3), dy=1.0),
                (f=sum, args = (cos, [1.0, 2.0],), dy=1.0),
                (f=*, args = (randn(2,2),randn(2,2)), dy=randn(2,2)),
            ]
            y_ref, pb_ref = Zygote.pullback(setup.f, setup.args...)
            y_cp, pb_cp = Zygote.pullback(Zygote.checkpointed, setup.f, setup.args...)
            @test y_cp == y_ref
            @test pb_cp(setup.dy) == (nothing, pb_ref(setup.dy)...)
        end
    end

    mutable struct CountCalls
        f
        ncalls
    end
    CountCalls(f=identity) = CountCalls(f, 0)
    function (o::CountCalls)(x...)
        o.ncalls += 1
        o.f(x...)
    end

    f = CountCalls()
    g = CountCalls()
    h = CountCalls()

    y, pb = Zygote.pullback(h ∘ g ∘ f, 4.0)
    @test y === 4.0
    @test f.ncalls === g.ncalls === h.ncalls === 1
    @test pb(1.0) === (1.0,)
    @test f.ncalls === g.ncalls === h.ncalls === 1

    f = CountCalls()
    g = CountCalls()
    h = CountCalls()
    y,pb = Zygote.pullback(Zygote.checkpointed, h∘g∘f, 4.0)
    @test f.ncalls === g.ncalls === h.ncalls === 1
    @test pb(1.0) === (nothing, 1.0,)
    @test f.ncalls === g.ncalls === h.ncalls === 2
    @test pb(1.0) === (nothing, 1.0,)
    @test f.ncalls === g.ncalls === h.ncalls === 3

    f = CountCalls()
    g = CountCalls()
    h = CountCalls()
    function only_some_checkpointed(x)
        x2 = f(x)
        Zygote.checkpointed(h∘g, x2)
    end
    y,pb = Zygote.pullback(only_some_checkpointed, 4.0)
    @test f.ncalls === g.ncalls === h.ncalls === 1
    @test pb(1.0) === (1.0,)
    @test g.ncalls === h.ncalls === 2
    @test f.ncalls === 1

    @testset "nested checkpointing" begin
        f1 = CountCalls(sin)
        f2 = CountCalls(cos)
        f3 = CountCalls(max)
        function nested_checkpoints(x)
            Zygote.checkpointed() do
                a = f1(x)
                Zygote.checkpointed() do
                    b = f2(a)
                    Zygote.checkpointed() do
                        f3(a,b)
                    end
                end
            end
        end
        function nested_nocheckpoints(x)
            a = f1.f(x)
            b = f2.f(a)
            f3.f(a,b)
        end
        x = randn()
        y,pb = Zygote.pullback(nested_checkpoints, x)
        @test f1.ncalls == f2.ncalls == f3.ncalls
        dy = randn()
        pb(dy)
        @test f1.ncalls == 2
        @test f2.ncalls == 3
        @test f3.ncalls == 4
        y_ref, pb_ref = Zygote.pullback(nested_nocheckpoints, x)
        @test y_ref === y
        @test pb_ref(dy) == pb(dy)
    end
end


add(a, b) = a+b
_relu(x) = x > 0 ? x : 0
f(a, b...) = +(a, b...)

y, back = pullback(identity, 1)
dx = back(2)
@test y == 1
@test dx == (2,)

mul(a, b) = a*b
y, back = pullback(mul, 2, 3)
dx = back(4)
@test y == 6
@test dx == (12, 8)

@test gradient(mul, 2, 3) == (3, 2)

bool = true
b(x) = bool ? 2x : x

y, back = pullback(b, 3)
dx = back(4)
@test y == 6
@test dx == (8,)

fglobal = x -> 5x
gglobal = x -> fglobal(x)

@test gradient(gglobal, 2) == (5,)

global bool = false

y, back = pullback(b, 3)
dx = back(4)
@test y == 3
@test getindex.(dx) == (4,)

y, back = pullback(x -> sum(x.*x), [1, 2, 3])
dxs = back(1)
@test y == 14
@test dxs == ([2,4,6],)

function pow(x, n)
  r = 1
  while n > 0
    n -= 1
    r *= x
  end
  return r
end

@test gradient(pow, 2, 3) == (12,nothing)

function pow_mut(x, n)
  r = Ref(one(x))
  while n > 0
    n -= 1
    r[] = r[] * x
  end
  return r[]
end

@test gradient(pow_mut, 2, 3) == (12,nothing)

r = 1
function pow_global(x, n)
  global r
  while n > 0
    r *= x
    n -= 1
  end
  return r
end

@test gradient(pow_global, 2, 3) == (12,nothing)

@test gradient(x -> 1, 2) == (nothing,)

@test gradient(t -> t[1]*t[2], (2, 3)) == ((3, 2),)

@test gradient(x -> x.re, 2+3im) == ((re = 1, im = nothing),)

@test gradient(x -> x.re*x.im, 2+3im) == ((re = 3, im = 2),)

struct Foo{T}
  a::T
  b::T
end

function mul_struct(a, b)
  c = Foo(a, b)
  c.a * c.b
end

@test gradient(mul_struct, 2, 3) == (3, 2)

function mul_tuple(a, b)
  c = (a, b)
  c[1] * c[2]
end

@test gradient(mul_tuple, 2, 3) == (3, 2)

function mul_lambda(x, y)
  g = z -> x * z
  g(y)
end

@test gradient(mul_lambda, 2, 3) == (3, 2)

@test gradient((a, b...) -> *(a, b...), 2, 3) == (3, 2)

@test gradient((x, a...) -> x, 1) == (1,)
@test gradient((x, a...) -> x, 1, 1) == (1,nothing)
@test gradient((x, a...) -> x == a, 1) == (nothing,)
@test gradient((x, a...) -> x == a, 1, 2) == (nothing,nothing)

kwmul(; a = 1, b) = a*b

mul_kw(a, b) = kwmul(a = a, b = b)

@test gradient(mul_kw, 2, 3) == (3, 2)

function myprod(xs)
  s = 1
  for x in xs
    s *= x
  end
  return s
end

@test gradient(myprod, [1,2,3])[1] == [6,3,2]

function mul_vec(a, b)
  xs = [a, b]
  xs[1] * xs[2]
end

@test gradient(mul_vec, 2, 3) == (3, 2)

@test gradient(2) do x
  d = Dict()
  d[:x] = x
  x * d[:x]
end == (4,)

f(args...;a=nothing,kwargs...) = g(a,args...;kwargs...)
g(args...;x=1,idx=Colon(),kwargs...) = x[idx]
@test gradient(x->sum(f(;x=x,idx=1:1)),ones(2))[1] == [1., 0.]

pow_rec(x, n) = n == 0 ? 1 : x*pow_rec(x, n-1)

@test gradient(pow_rec, 2, 3) == (12, nothing)

# For nested AD, until we support errors
function grad(f, args...)
  y, back = pullback(f, args...)
  return back(1)
end

D(f, x) = grad(f, x)[1]

@test D(x -> D(sin, x), 0.5) == -sin(0.5)
@test D(x -> x*D(y -> x+y, 1), 1) == 1
@test D(x -> x*D(y -> x*y, 1), 4) == 8

@test sin'''(1.0) ==  -cos(1.0)

f(x) = throw(DimensionMismatch("fubar"))

@test_throws DimensionMismatch gradient(f, 1)

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

let
  p = [1]
  θ = Zygote.Params([p])
  θ̄ = gradient(θ) do
    p′ = (p,)[1]
    p′[1]
  end
  @test θ̄[p][1] == 1
end

@test gradient(2) do x
  H = [1 x; 3 4]
  sum(H)
end[1] == 1

@test gradient(2) do x
  if x < 0
    throw("foo")
  end
  return x*5
end[1] == 5

@test gradient(x -> one(eltype(x)), rand(10))[1] === nothing

# Thre-way control flow merge
@test gradient(1) do x
  if x > 0
    x *= 2
  elseif x < 0
    x *= 3
  end
  x
end[1] == 2

# Gradient of closure
grad_closure(x) = 2x

Zygote.@adjoint (f::typeof(grad_closure))(x) = f(x), Δ -> (1, 2)

@test gradient((f, x) -> f(x), grad_closure, 5) == (1, 2)

invokable(x) = 2x
invokable(x::Integer) = 3x
@test gradient(x -> invoke(invokable, Tuple{Any}, x), 5) == (2,)

y, back = Zygote.pullback(x->tuple(x...), [1, 2, 3])
@test back((1, 1, 1)) == ((1,1,1),)

# Test for some compiler errors on complex CFGs
function f(x)
  while true
    true && return
    foo(x) && break
  end
end

@test Zygote.@code_adjoint(f(1)) isa Zygote.Adjoint

@test_throws ErrorException Zygote.gradient(1) do x
  push!([], x)
  return x
end

@test gradient(1) do x
  stk = []
  Zygote._push!(stk, x)
  stk = Zygote.Stack(stk)
  pop!(stk)
end == (1,)

@test gradient(x -> [x][1].a, Foo(1, 1)) == ((a=1, b=nothing),)

@test gradient((a, b) -> Zygote.hook(-, a)*b, 2, 3) == (-3, 2)

@test gradient(5) do x
  forwarddiff(x -> x^2, x)
end == (10,)

@test gradient(1) do x
  if true
  elseif true
    nothing
  end
  x + x
end == (2,)

global_param = 3

@testset "Global Params" begin
  cx = Zygote.Context()
  y, back = Zygote._pullback(cx, x -> x*global_param, 2)
  @test y == 6
  @test back(1) == (nothing, 3)
  Zygote.cache(cx)[GlobalRef(Main, :global_param)] == 2
end

function pow_try(x)
  try
    2x
  catch e
    println("error")
  end
end

@test_broken gradient(pow_try, 1) == (2,)

function pow_simd(x, n)
  r = 1
  @simd for i = 1:n
    r *= x
  end
  return r
end

@test_broken gradient(pow_simd, 2, 3) == (12,nothing)

@testset "tuple getindex" begin
  @test gradient(x -> size(x)[2], ones(2,2,2)) == (nothing,)
  @test gradient(x -> sum(size(x)[1:2]), ones(2,2,2)) == (nothing,)
  @test gradient(x -> sum(size(x)[1:2:3]), ones(2,2,2,2)) == (nothing,)
  @test gradient(x -> sum(size(x)[[1,2,1]]), ones(2,2,2)) == (nothing,)

  @test gradient((x,y,z) -> sum((x,y,z)[1:2]), 7, 8.8, 9.9) == (1.0, 1.0, nothing)
  @test gradient((x,y,z) -> sum((x,y,z)[[1,2,1]]), 1,2,3) == (2, 1, nothing)
end

@testset "@timed" begin
  @test gradient(x -> first(@timed x), 0) == (1,)
end

mutable struct MyMutable
    value::Float64
end

function foo!(m::MyMutable, x)
    m.value = x
end

function baz(args)
    m = MyMutable(0.)
    foo!(m, args...)
    m.value
end

let
  value, back = Zygote.pullback(baz, (1.0,))
  @test back(1.) == ((1.0,),)
end

function type_test()
   Complex{<:Real}
end

@test pullback(type_test)[1] == Complex{<:Real}

@testset "Pairs" begin
    @test (x->10*pairs((a=x, b=2))[1])'(100) === 10
    @test (x->10*pairs((a=x, b=2))[2])'(100) === 0
    foo(;kw...) = 1
    @test gradient(() -> foo(a=1,b=2.0)) === ()

    @test (x->10*(x => 2)[1])'(100) === 10
    @test (x->10*(x => 2)[2])'(100) === 0
end

# https://github.com/JuliaDiff/ChainRules.jl/issues/257
@testset "Keyword Argument Passing" begin
  struct Type1{VJP}
    x::VJP
  end

  struct Type2{compile}
    Type2(compile=false) = new{compile}()
  end

  function loss_adjoint(θ)
    sum(f(sensealg=Type1(Type2(true))))
  end

  i = 1
  global x = Any[nothing,nothing]

  Zygote.@nograd g(x,i,sensealg) = Main.x[i] = sensealg
  function f(;sensealg=nothing)
    g(x,i,sensealg)
    return rand(100)
  end

  loss_adjoint([1.0])
  i = 2
  Zygote.gradient(loss_adjoint,[1.0])
  @test x[1] == x[2]
end
