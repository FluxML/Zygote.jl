using Zygote, Test
using Zygote: Params, gradient, forwarddiff

add(a, b) = a+b
_relu(x) = x > 0 ? x : 0
f(a, b...) = +(a, b...)

y, back = forward(identity, 1)
dx = back(2)
@test y == 1
@test dx == (2,)

mul(a, b) = a*b
y, back = forward(mul, 2, 3)
dx = back(4)
@test y == 6
@test dx == (12, 8)

@test gradient(mul, 2, 3) == (3, 2)

bool = true
b(x) = bool ? 2x : x

y, back = forward(b, 3)
dx = back(4)
@test y == 6
@test dx == (8,)

fglobal = x -> 5x
gglobal = x -> fglobal(x)

@test gradient(gglobal, 2) == (5,)

global bool = false

y, back = forward(b, 3)
dx = back(4)
@test y == 3
@test getindex.(dx) == (4,)

y, back = forward(x -> sum(x.*x), [1, 2, 3])
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

pow_rec(x, n) = n == 0 ? 1 : x*pow_rec(x, n-1)

if !Zygote.usetyped
  @test gradient(pow_rec, 2, 3) == (12, nothing)
end

# For nested AD, until we support errors
function grad(f, args...)
  y, back = forward(f, args...)
  return back(1)
end

D(f, x) = grad(f, x)[1]

@test D(x -> D(sin, x), 0.5) == -sin(0.5)
@test D(x -> x*D(y -> x+y, 1), 1) == 1
@test D(x -> x*D(y -> x*y, 1), 4) == 8
@test_broken sin'''(1.0) == -cos(1.0)

f(x) = throw(DimensionMismatch("fubar"))

@test_throws DimensionMismatch gradient(f, 1)

struct Layer{T}
  W::T
end

(f::Layer)(x) = f.W * x

W = [1 0; 0 1]
x = [1, 2]

y, back = forward(() -> W * x, Params([W]))
@test y == [1, 2]
@test back([1, 1])[W] == [1 2; 1 2]

layer = Layer(W)

y, back = forward(() -> layer(x), Params([W]))
@test y == [1, 2]
@test back([1, 1])[W] == [1 2; 1 2]

@test gradient(() -> sum(W * x), Params([W]))[W] == [1 2; 1 2]

@test gradient(2) do x
  H = [1 x; 3 4]
  sum(H)
end[1] == 1

# FIXME
if !Zygote.usetyped
  @test gradient(2) do x
    if x < 0
      throw("foo")
    end
    return x*5
  end[1] == 5

  @test gradient(x -> one(eltype(x)), rand(10))[1] == nothing
end

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

Zygote.usetyped && Zygote.refresh()

@test gradient((f, x) -> f(x), grad_closure, 5) == (1, 2)

if !Zygote.usetyped
  invokable(x) = 2x
  invokable(x::Integer) = 3x
  @test gradient(x -> invoke(invokable, Tuple{Any}, x), 5) == (2,)
end

y, back = Zygote.forward(x->tuple(x...), [1, 2, 3])
@test back((1, 1, 1))[1] in ((1,1,1),[1,1,1])

# Test for some compiler errors on complex CFGs
function f(x)
  while true
    true && return
    foo(x) && break
  end
end

@test Zygote.@code_adjoint(f(1)) isa Zygote.Adjoint

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

# Mutation

@test gradient([1, 2],3) do x, y
  x[1] = y
  x[1] * x[2]
end == ([0, 3], 2)

using LinearAlgebra

@test gradient([1, 2]) do x
  y = x ⋅ x
  x[1] = 3
  y
end == ([2, 4],)

@test gradient(2) do x
  xs = []
  push!(xs, x)
  pop!(xs)*x
end == (4,)

global_param = 3

@testset "Global Params" begin
  cx = Zygote.Context()
  y, back = Zygote._forward(cx, x -> x*global_param, 2)
  @test y == 6
  @test back(1) == (nothing, 3)
  Zygote.globals(cx)[GlobalRef(Main, :global_param)] == 2
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
