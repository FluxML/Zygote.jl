using Zygote, Test
using Zygote: forward

macro test_inferred(ex)
  :(let res = nothing
    @test begin
      res = @inferred $ex
      true
    end
    res
  end) |> esc
end

y, back = @test_inferred forward(*, 2, 3)
@test_inferred(back(1))

_sincos(x) = sin(cos(x))

y, back = @test_inferred forward(_sincos, 0.5)
@test_inferred back(1)

dpow(n, p) = something(Zygote.gradient(pow, n, p)[1], zero(n))

@test_inferred forward(pow, 2, 3)
@test_inferred dpow(2, 3)

cube(x) = pow(x, 3)
dcube(x) = something(Zygote.derivative(cube, x), zero(x))
y, back = @test_inferred forward(cube, 2)
@test_inferred dcube(2)
