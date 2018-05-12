using Zygote, Base.Test
using Zygote: gradient

bool = true
function f(x)
  y = bool ? 2x : x
  return y
end

# TODO: use Complex when static params work in the interpreter
struct Foo a; b end

@testset "Zygote" begin

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

y, back = forward(f, 3)
dx = back(4)
@test y == 6
dx == (8,)

bool = false

y, back = forward(f, 3)
dx = back(4)
@test y == 3
@test getindex.(dx) == (4,)

y, back = forward(broadcast, *, [1,2,3], [4,5,6])
dxs = back([1,1,1])
@test y == [4, 10, 18]
@test dxs == (nothing, [4, 5, 6], [1, 2, 3])

function pow(x, n)
  r = 1
  for _ = 1:n
    r *= x
  end
  return r
end

@test gradient(pow, 2, 3) == (12, nothing)

@test gradient(x -> 1, 2) == (nothing,)

@test gradient(t -> t[1]*t[2], (2, 3)) == ((3, 2),)

@test gradient(x -> x.re, 2+3im) == ((re = 1, im = nothing),)

@test gradient(x -> x.re*x.im, 2+3im) == ((re = 3, im = 2),)

function f(a, b)
  c = Foo(a, b)
  c.a * c.b
end

gradient(f, 2, 3) == (3, 2)

end
