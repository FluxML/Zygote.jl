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
