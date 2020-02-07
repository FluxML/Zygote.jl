using Zygote, Test
using Zygote.Forward: _tangent, zerolike

D(f, x) = _tangent((zerolike(f), one(x)), f, x)[2]

@test D(x -> sin(cos(x)), 0.5) == -cos(cos(0.5))*sin(0.5)

@test_broken D(x -> D(cos, x), 0.5) == -cos(0.5)

@test_broken D(x -> x*D(y -> x*y, 1), 4) == 8

relu(x) = x > 0 ? x : 0

D(relu, 5)

@test D(relu, 5) == 1
@test D(relu, -5) == 0
