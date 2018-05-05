using Zygote
using Base.Test

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

y, J = ∇(f, 3)
dx = J(4)
@test y == 6
@test getindex.(dx) == (8,)

bool = false

y, J = ∇(f, 3)
dx = J(4)
@test y == 3
@test getindex.(dx) == (4,)

end
