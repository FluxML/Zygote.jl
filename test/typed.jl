using Zygote, Test
using Zygote: gradient, derivative, forward

dpow(n, p) = something(gradient(pow, n, p)[1], zero(n))

@test_inferred forward(pow, 2, 3)
@test_inferred dpow(2, 3)

cube(x) = pow(x, 3)
dcube(x) = something(derivative(cube, x), zero(x))
y, back = @test_inferred forward(cube, 2)
@test_inferred dcube(2)
