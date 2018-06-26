using Zygote, Test
using Zygote: forward

Zygote.compiled()

@testset "Performance" begin

y, back = @inferred forward(*, 2, 3)
@test y == 6

@test @inferred(back(1)) == (3, 2)

_sincos(x) = sin(cos(x))

y, back = @inferred forward(_sincos, 0.5)
@test y == _sincos(0.5)
@inferred back(1)

y, back = @inferred forward(pow, 2, 3)
@inferred back(1)

end
