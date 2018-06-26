using Zygote, Test
using Zygote: forward

macro test_inferred(ex)
  :(let res
    @test begin
      res = @inferred $ex
      true
    end
    res
  end) |> esc
end

@testset "Compiler" begin

y, back = @test_inferred forward(*, 2, 3)
@test_inferred(back(1))

_sincos(x) = sin(cos(x))

y, back = @test_inferred forward(_sincos, 0.5)
@test_inferred back(1)

y, back = @test_inferred forward(pow, 2, 3)
@test_inferred back(1)

end
