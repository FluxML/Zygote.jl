@testset "lib.jl" begin
    @testset "accum" begin
        t1 = (a=1, b=2, c=3)
        t2 = (a=1, b=2)
        @test Zygote.accum(t1, t2) == (a = 2, b = 4, c = 3)
        @test_throws ArgumentError Zygote.accum(t2, t1)
        @test Zygote.accum(fill(0.0), fill(0.0)) == fill(0.0)
    end
end
