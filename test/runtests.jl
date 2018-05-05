using Zygote
using Base.Test

@testset "Zygote" begin

y, J = ∇(identity, 1)
dy = J(2)
@test y == 1
@test dy == (2,)

mul(a, b) = a*b
y, J = ∇(mul, 2, 3)
dy = J(4)
@test y == 6
@test dy == (12, 8)

end
