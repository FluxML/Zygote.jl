@testset "lib.jl" begin
    @testset "accum" begin
        t1 = (a=1, b=2, c=3)
        t2 = (a=1, b=2)
        @test Zygote.accum(t1, t2) == (a = 2, b = 4, c = 3)
        @test_throws ArgumentError Zygote.accum(t2, t1)
        @test Zygote.accum(fill(0.0), fill(0.0)) == fill(0.0)
    end

    @testset "module getproperty/getfield (#194, #252)" begin
        # Reading a binding from a module reached via an SSA value (qualified access)
        # is non-differentiable; the gradient flows only to the other operand. These
        # used to die with `UndefVarError: j`.
        γ = Base.MathConstants.eulergamma
        @test gradient(x -> Base.MathConstants.eulergamma * x, 1.0)[1] ≈ γ
        @test gradient(x -> getfield(Base.MathConstants, :eulergamma) * x, 1.0)[1] ≈ γ
        @test gradient(x -> x * Base.MathConstants.e, 2.0)[1] ≈ Base.MathConstants.e
    end

    @testset "getfield by dynamic integer index" begin
        # `getfield(x, ::Int)` with a non-literal index stays differentiable
        # (literal indices are already lowered to `literal_getfield` at IR level).
        i = Ref(2)
        @test gradient(p -> getfield(p, i[]), (1.0, 2.0, 3.0)) == ((nothing, 1.0, nothing),)
        @test gradient(p -> getfield(p, i[]), (a=1.0, b=2.0)) == ((a=nothing, b=1.0),)
    end
end
