using Zygote
using Base.Test

@testset "Zygote" begin

y, J = ∇(identity, 1)
dy = J(2)
@test y == 1
@test dy == 2

end
