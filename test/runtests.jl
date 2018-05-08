using Zygote, Base.Test
using Zygote: gradient

bool = true
function f(x)
  y = bool ? 2x : x
  return y
end

@testset "Zygote" begin

y, J = ∇(identity, 1)
dx = J(2)
@test y == 1
@test dx == (2,)

mul(a, b) = a*b
y, J = ∇(mul, 2, 3)
dx = J(4)
@test y == 6
@test dx == (12, 8)

@test gradient(mul, 2, 3) == (3, 2)

y, J = ∇(f, 3)
dx = J(4)
@test y == 6
dx == (8,)

bool = false

y, J = ∇(f, 3)
dx = J(4)
@test y == 3
@test getindex.(dx) == (4,)

y, J = ∇(broadcast, *, [1,2,3], [4,5,6])
dxs = J([1,1,1])
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

f(t) = t[1]*t[2]

@test gradient(f, (2,3)) == ((3, 2),)

end
