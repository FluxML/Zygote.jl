@testset "number.jl" begin
  @testset "nograds" begin
    @test gradient(floor, 1) === (0.0,)
    @test gradient(ceil, 1) === (0.0,)
    @test gradient(round, 1) === (0.0,)
    @test gradient(hash, 1) === (nothing,)
    @test gradient(div, 1, 2) === (nothing, nothing)
  end

  @testset "basics" begin
    @test gradient(Base.literal_pow, ^, 3//2, Val(-5))[2] isa Rational

    @test gradient(convert, Rational, 3.14) == (nothing, 1.0)
    @test gradient(convert, Rational, 2.3) == (nothing, 1.0)
    @test gradient(convert, UInt64, 2) == (nothing, 1.0)
    @test gradient(convert, BigFloat, π) == (nothing, 1.0)

    @test gradient(Rational, 2) == (1//1,)

    @test gradient(Bool, 1) == (1.0,)
    @test gradient(Int32, 2) == (1.0,)
    @test gradient(UInt16, 2) == (1.0,)

    @test gradient(+, 2.0, 3, 4.0, 5.0) == (1.0, 1.0, 1.0, 1.0)

    @test gradient(//, 3, 2) == (1//2, -3//4)
  end

  @testset "literal_pow at zero (#1598)" begin
    # `sqrt(x^2)` has a cusp at `x == 0`: the inner `x^2` derivative is `0` while
    # the outer `sqrt` derivative is `Inf`, which used to multiply to `NaN`.
    @test gradient(x -> sqrt(x^2), 0.0) == (0.0,)
    @test gradient(x -> sqrt(x^2), 2.0) == (1.0,)
    @test gradient(x -> sqrt(x^2), -2.0) == (-1.0,)
    # ordinary `^` gradients must be unaffected
    @test gradient(x -> x^2, 0.0) == (0.0,)
    @test gradient(x -> x^3, 0.0) == (0.0,)
    @test gradient(x -> x^2, 3.0) == (6.0,)
    # broadcasted path
    @test gradient(x -> sum(sqrt.(x.^2)), [0.0, 1.0, -2.0]) == ([0.0, 1.0, -1.0],)
    @test gradient(x -> sum(x.^2), [0.0, 1.0]) == ([0.0, 2.0],)
  end

  @testset "flipsign / copysign" begin
    # Differentiable despite their bit-twiddling implementations: d/dx = ±1 (the
    # sign copied from `y`), d/dy = 0. Without rules these hit a non-diff intrinsic.
    for x in (0.8, -1.4), y in (2.3, -0.7)
      @test gradient(flipsign, x, y)[1] == flipsign(1.0, y)
      @test gradient(copysign, x, y)[1] == flipsign(1.0, x * y)
      @test gradient(flipsign, x, y)[2] === nothing  # sign argument is non-differentiable
      @test gradient(copysign, x, y)[2] === nothing
    end
    @test gradient(x -> flipsign(x, -3.0), 2.0) == (-1.0,)
    @test gradient(x -> copysign(x, -3.0), 2.0) == (-1.0,)
  end

  @testset "Complex numbers" begin
    @test gradient(imag, 3.0) == (0.0,)
    @test gradient(imag, 3.0 + 3.0im) == (0.0 + 1.0im,)

    @test gradient(conj, 3.0) == (1.0,)
    @test gradient(real ∘ conj, 3.0 + 1im) == (1.0 + 0im,)

    @test gradient(real, 3.0) == (1.0,)
    @test gradient(real, 3.0 + 1im) == (1.0 + 0im,)

    @test gradient(abs2, 3.0) == (2*3.0,)
    @test gradient(abs2, 3.0+2im) == (2*3.0 + 2*2.0im,)

    @test gradient(real ∘ Complex, 3.0, 2.0) == (1.0, 0.0)
  end
end
