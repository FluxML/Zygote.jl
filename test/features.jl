using Zygote, Test, LinearAlgebra
using Zygote: Params, gradient, forwarddiff
using FillArrays: Fill

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
@test withgradient(mul, 2, 3) == (val = 6, grad = (3, 2))

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

@test gradient(x -> x.re, 2+3im) === (1.0 + 0.0im,)  # one NamedTuple
@test gradient(x -> x.re*x.im, 2+3im) == (3.0 + 2.0im,)  # two, different fields
@test gradient(x -> x.re*x.im + x.re, 2+3im) == (4.0 + 2.0im,)  # three, with accumulation

@test gradient(x -> abs2(x * x.re), 4+5im) == (456.0 + 160.0im,)   # gradient participates
@test gradient(x -> abs2(x * real(x)), 4+5im) == (456.0 + 160.0im,)   # function not getproperty
@test gradient(x -> abs2(x * getfield(x, :re)), 4+5im) == (456.0 + 160.0im,)

struct Bar{T}
  a::T
  b::T
end

function mul_struct(a, b)
  c = Bar(a, b)
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

@test sin''(1.0) ==  -sin(1.0)
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

# Three-way control flow merge
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

@test gradient(x -> [x][1].a, Bar(1, 1)) == ((a=1, b=nothing),)

@test gradient((a, b) -> Zygote.hook(-, a)*b, 2, 3) == (-3, 2)

@test gradient(5) do x
  forwarddiff(x -> x^2, x)
end == (10,)

@testset "Gradient chunking" begin
  for chunk_threshold in 1:10:100
    x = [1:100;]
    @test gradient(x) do x
      Zygote.forwarddiff(x -> x' * x, x; chunk_threshold = chunk_threshold)
    end == (2 * x,)
  end
end


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
  ref = first(keys(Zygote.cache(cx)))
  @test ref isa GlobalRef
  @test ref.mod == Main
  @test ref.name == :global_param
  @test Zygote.cache(cx)[ref] == 2
end

function pow_try(x)
  try
    2x
  catch e
    println("error")
  end
end

@test gradient(pow_try, 1) == (2,)

function pow_simd(x, n)
  r = 1
  @simd for i = 1:n
    r *= x
  end
  return r
end

@test gradient(pow_simd, 2, 3) == (12,nothing)

@testset "tuple getindex" begin
  @test gradient(x -> size(x)[2], ones(2,2,2)) == (nothing,)
  @test gradient(x -> sum(size(x)[1:2]), ones(2,2,2)) == (nothing,)
  @test gradient(x -> sum(size(x)[1:2:3]), ones(2,2,2,2)) == (nothing,)
  @test gradient(x -> sum(size(x)[[1,2,1]]), ones(2,2,2)) == (nothing,)

  @test gradient((x,y,z) -> sum((x,y,z)[1:2]), 7, 8.8, 9.9) == (1.0, 1.0, nothing)
  @test gradient((x,y,z) -> sum((x,y,z)[[1,2,1]]), 1,2,3) == (2, 1, nothing)

  @test gradient(xs -> sum(x -> x[2], xs), [(1,2,3), (4,5,6)]) == ([(nothing, 1.0, nothing), (nothing, 1.0, nothing)],)
  @test gradient(xs -> sum(x -> prod(x[2:3]), xs), [(1,2,3), (4,5,6)]) == ([(nothing, 3.0, 2.0), (nothing, 6.0, 5.0)],)
  @test gradient(xs -> sum(first, xs), fill((4,3),2)) == ([(1.0, nothing), (1.0, nothing)],)
  @test gradient(xs -> sum(x -> abs2(x[1]), xs), fill((4,3),2)) == ([(8.0, nothing), (8.0, nothing)],)
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

@testset "mutable struct, including Ref" begin
  # Zygote's representation is Base.RefValue{Any}((value = 7.0,)), but the
  # map to ChainRules types and back normalises to (value = 7.0,) same as struct:
  @test gradient(x -> x.value^2 + x.value, MyMutable(3)) === ((value = 7.0,),)

  # Same for Ref. This doesn't seem to affect `pow_mut` test in this file.
  @test gradient(x -> x.x^2 + x.x, Ref(3)) === ((x = 7.0,),)
  @test gradient(x -> real(x.x^2 + im * x.x), Ref(4)) === ((x = 8.0,),)

  # Field access of contents:
  @test gradient(x -> abs2(x.x) + 7 * x.x.re, Ref(1+im)) == ((x = 9.0 + 2.0im,),)
  @test_broken gradient(x -> abs2(x[1].x) + 7 * x[1].x.re, [Ref(1+im)]) == ([(x = 9.0 + 2.0im,)],)
  @test_broken gradient(x -> abs2(x[1].x) + 7 * real(x[1].x), [Ref(1+im)]) == ([(x = 9.0 + 2.0im,)],)  # worked on 0.6.0, 0.6.20

  @test gradient(x -> abs2(x[].x) + 7 * real(x[].x), Ref(Ref(1+im))) == ((x = (x = 9.0 + 2.0im,),),) # gave `nothing` from 0.6.0 to 0.6.41

  # Array of mutables:
  @test gradient(x -> sum(getindex.(x).^2), Ref.(1:3))[1] == [(;x=2i) for i in 1:3]
  @test gradient(x -> sum(abs2∘getindex, x), Ref.(1:3))[1] == [(;x=2i) for i in 1:3]

  @test gradient(x -> (getindex.(x).^2)[1], Ref.(1:3))[1][1] == (x=2.0,)  # rest are (x = 0.0,), but nothing would be OK too
  @test gradient(x -> (prod.(getindex.(x)))[1], Ref.(eachcol([1 2; 3 4])))[1][1] == (x = [3.0, 1.0],)

  # Broadcasting over Ref is handled specially. Tested elsewhere too.
  @test gradient(x -> sum(sum, x .* [1,2,3]), Ref([4,5])) == ((x = [6.0, 6.0],),)
  @test gradient(x -> sum(sum, Ref(x) .* [1,2,3]), [4,5]) == ([6.0, 6.0],)
end

@testset "mutable accum_param bugs" begin
  mutable struct Mut{T}; x::T; end
  struct Imm{T}; x::T; end

  # Indexing a tuple containing a mutable struct gave `nothing`
  x1 = (Mut(3.0),)
  x2 = (Imm(3.0),)
  x3 = (Ref(3.0),)
  @test gradient(x -> x[1].x^2, x1)[1] == ((x = 6.0,),)  # fails on v0.6.0 v0.6.41
  @test gradient(x -> x[1].x^2, x2)[1] == ((x = 6.0,),)
  @test gradient(x -> x[1].x^2, x3)[1] == ((x = 6.0,),)  # fails on v0.6.0 v0.6.41
  i1 = 1
  @test gradient(x -> x[i1].x^2, x1)[1] == ((x = 6.0,),)  # fails on v0.6.0 v0.6.41
  @test gradient(x -> x[i1].x^2, x2)[1] == ((x = 6.0,),)
  @test gradient(x -> x[i1].x^2, x3)[1] == ((x = 6.0,),)  # fails on v0.6.0 v0.6.41

  @test gradient(x -> x[1][1].x^2, [x1])[1] == [((x = 6.0,),)]  # fails on v0.6.0 v0.6.41
  @test gradient(x -> x[1][1].x^2, [x2])[1] == [((x = 6.0,),)]
  @test gradient(x -> x[1][1].x^2, [x3])[1] == [((x = 6.0,),)]  # fails on v0.6.0 v0.6.41

  # When `getfield` returns a mutable struct, it gave `nothing`:
  x4 = Imm(Mut(4.0))
  x5 = Mut(Mut(4.0))
  x6 = Imm(Imm(4.0))
  @test gradient(x -> x.x.x^3, x4)[1] == (x = (x = 48.0,),)  # fails on v0.6.0 v0.6.41
  @test gradient(x -> x.x.x^3, x5)[1] == (x = (x = 48.0,),)  # fails on v0.6.0
  @test gradient(x -> x.x.x^3, x6)[1] == (x = (x = 48.0,),)  # fails on v0.6.41

  @test gradient(x -> x[2].x.x^3, [x4, x4])[1] == [nothing, (x = (x = 48.0,),)]  # fails on v0.6.0 v0.6.41
  @test gradient(x -> x[2].x.x^3, [x4, x5])[1] == [nothing, (x = (x = 48.0,),)]  # fails on v0.6.0
  @test gradient(x -> x[2].x.x^3, [x4, x6])[1] == [nothing, (x = (x = 48.0,),)]  # fails on v0.6.41

  # Check when using implicit parameters, Params cases used to pass:
  y1 = [3.0]
  y2 = (Mut(y1),)
  y3 = (Imm(y1),)
  @test gradient(x -> sum(x[1].x)^2, y2)[1] == ((x = [6.0],),)  # fails on v0.6.0 v0.6.41
  @test gradient(() -> sum(y2[1].x)^2, Params([y1]))[y1] == [6.0]
  @test gradient(x -> sum(x[1].x)^2, y3)[1] == ((x = [6.0],),)
  @test gradient(() -> sum(y3[1].x)^2, Params([y1]))[y1] == [6.0]

  @test gradient(x -> sum(x[1].x .+ x[1].x)^3, y2)[1] == ((x = [216.0],),)  # fails on v0.6.0 v0.6.41
  @test gradient(() -> sum(y2[1].x .+ y2[1].x)^3, Params([y1]))[y1] == [216.0]
  @test gradient(x -> sum(x[1].x .+ x[1].x)^3, y3)[1] == ((x = [216.0],),)
  @test gradient(() -> sum(y3[1].x .+ y3[1].x)^3, Params([y1]))[y1] == [216.0]

  i1 = 1
  @test gradient(x -> sum(x[i1].x .+ x[1].x)^3, y2)[1] == ((x = [216.0],),)  # fails on v0.6.0 v0.6.41
  @test gradient(() -> sum(y2[i1].x .+ y2[1].x)^3, Params([y1]))[y1] == [216.0]
  @test gradient(x -> sum(x[i1].x .+ x[1].x)^3, y3)[1] == ((x = [216.0],),)
  @test gradient(() -> sum(y3[i1].x .+ y3[1].x)^3, Params([y1]))[y1] == [216.0]
end

@testset "NamedTuples" begin
  @test gradient(x -> x.a, (a=1, b=2)) == ((a = 1, b = nothing),)
  @test gradient(x -> x[1].a, [(a=1, b=2)]) == ([(a = 1, b = nothing)],)
  @test gradient(x -> x[1].a, [(a=1, b=2), (a=3, b=4)]) == ([(a = 1, b = nothing), nothing],)

  # Mix with Ref
  @test gradient(x -> x[].a, Ref((a=1, b=2))) == ((x = (a = 1, b = nothing),),)
  @test gradient(x -> x[1][].a, [Ref((a=1, b=2)), Ref((a=3, b=4))]) == ([(x = (a = 1, b = nothing),), nothing],)
  @test gradient(x -> x[1].a, [(a=1, b=2), "three"]) == ([(a = 1, b = nothing), nothing],)

  @testset "indexing kwargs" begin
    inner_lit_index(; kwargs...) = kwargs[:x]
    outer_lit_index(; kwargs...) = inner_lit_index(; x=kwargs[:x])

    inner_dyn_index(k; kwargs...) = kwargs[k]
    outer_dyn_index(k; kwargs...) = inner_dyn_index(k; x=kwargs[k])

    @test gradient(x -> outer_lit_index(; x), 0.0) == (1.0,)
    @test gradient((x, k) -> outer_dyn_index(k; x), 0.0, :x) == (1.0, nothing)
  end
end

function type_test()
   Complex{<:Real}
end

@test pullback(type_test)[1] == Complex{<:Real}

@testset "Pairs" begin
  @test (x->10*pairs((a=x, b=2))[1])'(100) === 10.0
  @test (x->10*pairs((a=x, b=2))[2])'(100) === nothing
  foo(;kw...) = 1
  @test gradient(() -> foo(a=1,b=2.0)) === ()

  @test (x->10*(x => 2)[1])'(100) === 10.0
  @test (x->10*(x => 2)[2])'(100) === nothing

  @test gradient(x-> (:x => x)[2], 17) == (1,)

  d = Dict(:x=>1.0, :y=>3.0);
  @test gradient(d -> Dict(:x => d[:x])[:x], d) == (Dict(:x => 1),)
end

@testset "kwarg splatting, pass in object" begin
  g(; kwargs...) = kwargs[:x] * kwargs[:z]
  h(somedata) = g(; somedata...)
  @test gradient(h, (; x=3.0, y=4.0, z=2.3)) == ((x = 2.3, y = nothing, z = 3.0),)
  @test gradient(h, Dict(:x=>3.0, :y=>4.0, :z=>2.3)) == ((y = nothing, z = 3.0, x = 2.3),)

  # for when no kwargs have grads backpropogated
  no_kwarg_grad(x; kwargs...) = x[kwargs[:i]]
  @test gradient(x -> no_kwarg_grad(x; i=1), [1]) == ([1],)
end

@testset "Iterators" begin
  # enumerate
  @test gradient(1:5) do xs
    sum([x^i for (i,x) in enumerate(xs)])
  end == ([1, 4, 27, 256, 3125],)

  @test gradient([1,10,100]) do xs
    sum([xs[i]^i for (i,x) in enumerate(xs)])
  end == ([1, 2 * 10^1, 3 * 100^2],)

  @test gradient([1,10,100]) do xs
    sum((xs[i]^i for (i,x) in enumerate(xs))) # same without collect
  end == ([1, 2 * 10^1, 3 * 100^2],)

  # zip
  if VERSION >= v"1.5"
    # On Julia 1.4 and earlier, [x/y for (x,y) in zip(10:14, 1:10)] is a DimensionMismatch,
    # while on 1.5 - 1.7 it stops early.

    @test gradient(10:14, 1:10) do xs, ys
      sum([x/y for (x,y) in zip(xs, ys)])
    end[2] ≈ vcat(.-(10:14) ./ (1:5).^2, zeros(5))

    @test_broken gradient(10:14, 1:10) do xs, ys
      sum(x/y for (x,y) in zip(xs, ys))   # same without collect
      # Here @adjoint function Iterators.Zip(xs) gets dy = (is = (nothing, nothing),)
    end[2] ≈ vcat(.-(10:14) ./ (1:5).^2, zeros(5))
  end

  bk_z = pullback((xs,ys) -> sum([abs2(x*y) for (x,y) in zip(xs,ys)]), [1,2], [3im,4im])[2]
  @test bk_z(1.0)[1] isa AbstractVector{<:Real}  # projection

  # Iterators.Filter
  @test gradient(2:9) do xs
    sum([x^2 for x in xs if iseven(x)])
  end == ([4, 0, 8, 0, 12, 0, 16, 0],)

  @test gradient(2:9) do xs
    sum(x^2 for x in xs if iseven(x)) # same without collect
  end == ([4, 0, 8, 0, 12, 0, 16, 0],)

  # Iterators.Product
  @test gradient(1:10, 3:7) do xs, ys
    sum([x^2+y for x in xs, y in ys])
  end == (10:10:100, fill(10, 5))

  @test_broken gradient(1:10, 3:7) do xs, ys
    sum(x^2+y for x in xs, y in ys)  # same without collect
    # Here @adjoint function Iterators.product(xs...) gets dy = (iterators = (nothing, nothing),)
  end == (10:10:100, fill(10, 5))

  # Repeat that test without sum(iterator) -- also receives dy = (iterators = (nothing, nothing),)
  function prod_acc(xs, ys)
    out = 0
    # for (x,y) in Iterators.product(xs, ys)
    #   out += x^2+y
    for xy in Iterators.product(xs, ys)
      out += xy[1]^2 + xy[2]
    end
    out
  end
  @test prod_acc(1:10, 3:7) == sum(x^2+y for x in 1:10, y in 3:7)
  gradient(prod_acc, 1:10, 3:7) == (nothing, nothing) # sadly
  @test_broken gradient(prod_acc, 1:10, 3:7) == (10:10:100, fill(10, 5))

  @test gradient(rand(2,3)) do A
    sum([A[i,j] for i in 1:1, j in 1:2])
  end == ([1 1 0; 0 0 0],)

  @test gradient(ones(3,5), 1:7) do xs, ys
    sum([x+y for x in xs, y in ys])
  end == (fill(7, 3,5), fill(15, 7))

  bk_p = pullback((xs,ys) -> sum([x/y for x in xs, y in ys]), Diagonal([3,4,5]), [6,7]')[2]
  @test bk_p(1.0)[1] isa Diagonal  # projection
  @test bk_p(1.0)[2] isa Adjoint

  # Iterators.Product with enumerate
  @test gradient([2 3; 4 5]) do xs
    sum([x^i+y for (i,x) in enumerate(xs), y in xs])
  end == ([8 112; 36 2004],)
end

@testset "PythonCall custom @adjoint" begin
  using PythonCall: pyimport, pyconvert
  math = pyimport("math")
  pysin(x) = math.sin(x)
  Zygote.@adjoint pysin(x) = pyconvert(Float64, math.sin(x)), δ -> (pyconvert(Float64, δ * math.cos(x)),)
  @test Zygote.gradient(pysin, 1.5) == Zygote.gradient(sin, 1.5)
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

@testset "splats" begin
  @test gradient(x -> max(x...), [1,2,3])[1] == [0,0,1]
  @test gradient(x -> min(x...), (1,2,3))[1] === (1.0, 0.0, 0.0)

  @test gradient(x -> max(x...), [1 2; 3 4])[1] == [0 0; 0 1]
  @test gradient(x -> max(x...), [1,2,3]')[1] == [0 0 1]

  # https://github.com/FluxML/Zygote.jl/issues/599
  @test gradient(w -> sum([w...]), [1,1])[1] isa AbstractVector

  # https://github.com/FluxML/Zygote.jl/issues/866
  f866(x) = reshape(x, fill(2, 2)...)
  @test gradient(x->sum(f866(x)), rand(4))[1] == [1,1,1,1]

  # https://github.com/FluxML/Zygote.jl/issues/731
  f731(x) = sum([x' * x, x...])
  @test_broken gradient(f731, ones(3)) # MethodError: no method matching +(::Tuple{Float64, Float64, Float64}, ::Vector{Float64})
end

@testset "accumulation" begin
  # from https://github.com/FluxML/Zygote.jl/issues/905
  function net(x1)
    x2  = x1
    x3  = x1 + x2
    x4  = x1 + x2 + x3
    x5  = x1 + x2 + x3 + x4
    x6  = x1 + x2 + x3 + x4 + x5
    x7  = x1 + x2 + x3 + x4 + x5 + x6
    x8  = x1 + x2 + x3 + x4 + x5 + x6 + x7
    x9  = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
    x10 = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
  end
  loss(x) = sum(abs2, net(x))
  @test gradient(loss, ones(10,10))[1] == fill(131072, 10, 10)
  @test 150_000_000 > @allocated gradient(loss, ones(1000,1000))

  # https://github.com/FluxML/Zygote.jl/issues/1233
  function defensiveupdate(d, a)
    nd = deepcopy(d)
    nd[1] = d[1] * a
    return nd
  end
  d = Dict(i => ones(1) for i in 1:2)
  @test gradient(d) do d
    nd = defensiveupdate(d, 5)
    return sum(nd[1]) + sum(nd[2])
  end[1] == Dict(1 => Fill(5, 1), 2 => Fill(1, 1))
end

@testset "tricky broadcasting" begin
  @test gradient(x -> sum(x .+ ones(2,2)), (1,2)) == ((2,2),)
  @test gradient(x -> sum(x .+ ones(2,2)), (1,)) == ((4,),)
  @test gradient(x -> sum(x .+ ones(2,1)), (1,2)) == ((1,1),)

  # https://github.com/FluxML/Zygote.jl/issues/975
  gt = gradient((x,p) -> prod(x .^ p), [3,4], (1,2))
  gv = gradient((x,p) -> prod(x .^ p), [3,4], [1,2])
  @test gt[1] == gv[1]
  @test collect(gt[2]) ≈ gv[2]

  # closure captures y -- can't use ForwardDiff
  @test gradient((x,y) -> sum((z->z^2+y[1]).(x)), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])
  @test gradient((x,y) -> sum((z->z^2+y[1]), x), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])
  @test gradient((x,y) -> sum(map((z->z^2+y[1]), x)), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])
  @test gradient((x,y) -> mapreduce((z->z^2+y[1]), +, x), [1,2,3], [4,5]) == ([2, 4, 6], [3, 0])

  # type unstable
  @test gradient(xs -> sum((x -> x<2 ? false : x^2).(xs)), [1,2,3])[1][2:3] == [4, 6]
  @test gradient(xs -> sum((x -> x<2 ? false : x^2), xs), [1,2,3])[1][2:3] == [4, 6]
  @test gradient(xs -> sum(map((x -> x<2 ? false : x^2), xs)), [1,2,3])[1][2:3] == [4, 6]
  @test gradient(xs -> mapreduce((x -> x<2 ? false : x^2), +, xs), [1,2,3])[1][2:3] == [4, 6]

  # with Ref, Val, Symbol
  @test gradient(x -> sum(x .+ Ref(x[1])), [1,2,3]) == ([4,1,1],)
  @test gradient(x -> sum(x .+ (x[1],)), [1,2,3]) == ([4,1,1],)
  @test gradient(x -> sum((first∘tuple).(x, :ignore)), [1,2,3]) == ([1,1,1],)
  @test gradient(x -> sum((first∘tuple).(x, Symbol)), [1,2,3]) == ([1,1,1],)
  _f(x,::Val{y}=Val(2)) where {y} = x/y
  @test gradient(x -> sum(_f.(x, Val(2))), [1,2,3]) == ([0.5, 0.5, 0.5],)
  @test gradient(x -> sum(_f.(x)), [1,2,3]) == ([0.5, 0.5, 0.5],)
  @test gradient(x -> sum(map(_f, x)), [1,2,3]) == ([0.5, 0.5, 0.5],)

  # with Bool
  @test gradient(x -> sum(1 .- (x .> 0)), randn(5)) == (nothing,)
  @test gradient(x -> sum((y->1-y).(x .> 0)), randn(5)) == (nothing,)
  @test gradient(x -> sum(x .- (x .> 0)), randn(5)) == ([1,1,1,1,1],)

  @test gradient(x -> sum(x ./ [1,2,4]), [1,2,pi]) == ([1.0, 0.5, 0.25],)
  @test gradient(x -> sum(map(/, x, [1,2,4])), [1,2,pi]) == ([1.0, 0.5, 0.25],)

  # negative powers
  @test gradient((x,p) -> sum(x .^ p), [1.0,2.0,4.0], [1,-1,2])[1] ≈ [1.0, -0.25, 8.0]
  @test gradient((x,p) -> sum(x .^ p), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625]
  @test gradient((x,p) -> sum(z -> z^p, x), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625]
  @test gradient((x,p) -> mapreduce(z -> z^p, +, x), [1.0,2.0,4.0], -1)[1] ≈ [-1.0, -0.25, -0.0625]

  # second order
  @test gradient(x -> sum(gradient(y -> sum(y.^2), x)[1]), [1, 2])[1] ≈ [2, 2]
  @test gradient(x -> sum(gradient(y -> sum(sin.(y)), x)[1]), [1, 2])[1] ≈ [-0.8414709848078965, -0.9092974268256817]
  @test gradient(x -> sum(abs, gradient(y -> sum(log.(2 .* exp.(y)) .^ 2), x)[1]), [1, 2])[1] ≈ [2,2]

  # getproperty, Tangents, etc
  @test gradient(xs -> sum((x->x.im^2).(xs)), [1+2im,3])[1] == [4im, 0]
  @test gradient(xs -> sum((x->x.im^2), xs), [1+2im,3])[1] == [4im, 0]
  @test gradient(xs -> sum(map(x->x.im^2, xs)), [1+2im,3])[1] == [4im, 0]
  @test gradient(xs -> mapreduce(x->x.im^2, +, xs), [1+2im,3])[1] == [4im, 0]
end

@testset "broadcast fallbacks" begin
  # https://github.com/FluxML/Zygote.jl/issues/1359
  struct MyFloat64 <: Number
    n::Float64
  end

  Base.exp(f::MyFloat64) = MyFloat64(exp(f.n))
  Base.conj(f::MyFloat64) = MyFloat64(conj(f.n))
  Base.:*(x::MyFloat64, y::MyFloat64) = MyFloat64(x.n * y.n)

  x = MyFloat64[1., 2., 3.]
  result, pb = @inferred Zygote.pullback(Base.broadcasted, Base.Broadcast.DefaultArrayStyle{1}(), exp, x)
  @inferred pb(MyFloat64[1., 1., 1.])
end

@testset "Dict" begin
  # issue #717
  @test gradient(x -> (() -> x[:y])(), Dict(:y => 0.4)) == (Dict(:y => 1.0),)

  ntd = (; data = Dict("x" => rand(2)))
  @test gradient(x -> sum(x.data["x"]), ntd)[1] == (; data = Dict("x" => ones(2)))

  # issue #760
  function f760(x)
    d = Dict()
    for i in 1:4
        push!(d, i=>i^x)
    end
    sum(values(d))
  end
  @test gradient(f760, 3)[1] ≈ 123.93054835019153
end

@testset "withgradient" begin
  @test withgradient([1,2,4]) do x
    z = 1 ./ x
    sum(z), z
  end == (val = (1.75, [1.0, 0.5, 0.25]), grad = ([-1.0, -0.25, -0.0625],))

  @test withgradient(3.0, 4.0) do x, y
    (div = x/y, mul = x*y)
  end == (val = (div = 0.75, mul = 12.0), grad = (0.25, -0.1875))

  f3(x) = sum(sin, x), sum(cos, x), sum(tan, x)
  g1 = gradient(first∘f3, [1,2,3.0])
  y2, g2 = withgradient(first∘f3, [1,2,3.0])
  y3, g3 = withgradient(f3, [1,2,3.0])
  @test g1[1] ≈ g2[1] ≈ g3[1]
end

