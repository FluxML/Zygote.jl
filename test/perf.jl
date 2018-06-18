using Zygote, Test
using Zygote: forward

@testset "Performance" begin

y, back = @inferred forward(*, 2, 3)
@test y == 6

@test @inferred(back(1)) == (3 ,2)

end
