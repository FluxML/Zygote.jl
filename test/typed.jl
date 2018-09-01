using Zygote, Test
using Zygote: gradient, derivative, forward

Zygote.refresh()

_sincos(x) = sin(cos(x))

y, back = @test_inferred forward(_sincos, 0.5)
@test_inferred back(1)

dpow(n, p) = something(gradient(pow, n, p)[1], zero(n))

@test_inferred forward(pow, 2, 3)
@test_inferred dpow(2, 3)

cube(x) = pow(x, 3)
dcube(x) = something(derivative(cube, x), zero(x))
y, back = @test_inferred forward(cube, 2)
@test_inferred dcube(2)

f(x) = 3x^2 + 2x + 1

y, back = @test_inferred forward(f, 5)
@test y == 86
@test_inferred(back(1))

y, back = @test_inferred forward(Core._apply, +, (1, 2, 3))
@test_inferred back(1)
