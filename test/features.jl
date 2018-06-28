using Zygote, Test
using Zygote: gradient, roundtrip

bool = true
b(x) = bool ? 2x : x

fglobal = x -> 5x
gglobal = x -> fglobal(x)

struct Foo{T}
  a::T
  b::T
end

function pow(x, n)
  r = 1
  while n > 0
    n -= 1
    r *= x
  end
  return r
end

# For nested AD, until we support errors
function grad(f, args...)
  y, J = forward(f, args...)
  return J(1)
end

@testset "Features" begin

add(a, b) = a+b
relu(x) = x > 0 ? x : 0
f(a, b...) = +(a, b...)

@test roundtrip(add, 1, 2) == 3
@test roundtrip(relu, 1) == 1
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

y, back = forward(b, 3)
dx = back(4)
@test y == 6
@test dx == (8,)

@test gradient(gglobal, 2) == (5,)

global bool = false

y, back = forward(b, 3)
dx = back(4)
@test y == 3
@test getindex.(dx) == (4,)

y, back = forward(broadcast, *, [1,2,3], [4,5,6])
dxs = back([1,1,1])
@test y == [4, 10, 18]
@test dxs == (nothing, [4, 5, 6], [1, 2, 3])

@test gradient(pow, 2, 3) == (12, nothing)

@test gradient(x -> 1, 2) == (nothing,)

@test gradient(t -> t[1]*t[2], (2, 3)) == ((3, 2),)

@test gradient(x -> x.re, 2+3im) == ((re = 1, im = nothing),)

@test gradient(x -> x.re*x.im, 2+3im) == ((re = 3, im = 2),)

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

function mysum(xs)
  s = 0
  for x in xs
    s += x
  end
  return s
end

@test_broken gradient(mysum, (1,2,3)) == ((1,1,1),)

function f(a, b)
  xs = [a, b]
  xs[1] * xs[2]
end

@test gradient(f, 2, 3) == (3, 2)

@test grad(x -> grad(sin, x)[1], 0.5) == (-sin(0.5),)

f(x) = throw(DimensionMismatch("fubar"))

@test_throws DimensionMismatch gradient(f, 1)

end
