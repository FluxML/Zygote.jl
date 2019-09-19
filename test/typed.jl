using Zygote, Test
using Zygote: gradient, pullback

dpow(n, p) = something(gradient(pow, n, p)[1], zero(n))

@test_inferred pullback(pow, 2, 3)
@test_inferred dpow(2, 3)

cube(x) = pow(x, 3)
dcube(x) = something(gradient(cube, x)[1], zero(x))
y, back = @test_inferred pullback(cube, 2)
@test_inferred dcube(2)
