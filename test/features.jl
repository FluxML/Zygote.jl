using Zygote, Test
using Zygote: gradient, roundtrip

add(a, b) = a+b
_relu(x) = x > 0 ? x : 0
f(a, b...) = +(a, b...)

@test roundtrip(add, 1, 2) == 3
@test roundtrip(_relu, 1) == 1
@test roundtrip(Complex, 1, 2) == 1+2im
@test roundtrip(f, 1, 2, 3) == 6

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

@test gradient(pow, 2, 3) == (12, 0)

function pow_mut(x, n)
  r = Ref(one(x))
  while n > 0
    n -= 1
    r[] = r[] * x
  end
  return r[]
end

@test gradient(pow_mut, 2, 3) == (12, 0)

@test gradient(x -> 1, 2) == (nothing,)

@test gradient(t -> t[1]*t[2], (2, 3)) == ((3, 2),)

@test gradient(x -> x.re, 2+3im) == ((re = 1, im = nothing),)

@test gradient(x -> x.re*x.im, 2+3im) == ((re = 3, im = 2),)

struct Foo{T}
  a::T
  b::T
end

function f(a, b)
  c = Foo(a, b)
  c.a * c.b
end

@test gradient(f, 2, 3) == (3, 2)

function f(a, b)
  c = (a, b)
  c[1] * c[2]
end

@test gradient(f, 2, 3) == (3, 2)

function f(x, y)
  g = z -> x * z
  g(y)
end

@test gradient(f, 2, 3) == (3, 2)

@test gradient((a, b...) -> *(a, b...), 2, 3) == (3, 2)

function myprod(xs)
  s = 1
  for x in xs
    s *= x
  end
  return s
end

@test gradient(myprod, [1,2,3])[1] == [6,3,2]

function f(a, b)
  xs = [a, b]
  xs[1] * xs[2]
end

@test gradient(f, 2, 3) == (3, 2)

# For nested AD, until we support errors
function grad(f, args...)
  y, J = forward(f, args...)
  return J(1)
end

D(f, x) = grad(f, x)[1]

@test D(x -> D(sin, x), 0.5) == -sin(0.5)

# FIXME segfaults on beta2 for some reason
# @test D(x -> x*D(y -> x+y, 1), 1) == 1
# @test D(x -> x*D(y -> x*y, 1), 4) == 8

f(x) = throw(DimensionMismatch("fubar"))

@test_throws DimensionMismatch gradient(f, 1)
