@testset "lib.jl" begin
    @testset "accum" begin
        t1 = (a=1, b=2, c=3)
        t2 = (a=1, b=2)
        @test Zygote.accum(t1, t2) == (a = 2, b = 4, c = 3)
        @test_throws ArgumentError Zygote.accum(t2, t1)
        @test Zygote.accum(fill(0.0), fill(0.0)) == fill(0.0)

        # HermOrSymSparse accumulation
        S = sparse([1, 2, 2], [1, 1, 2], [1.0, 2.0, 3.0], 2, 2)
        H1 = Hermitian(S + S')
        H2 = Hermitian(2S + 2S')
        @test Zygote.accum(H1, H2) == H1 + H2
        @test Zygote.accum(H1, H2) isa Hermitian{Float64, <:SparseMatrixCSC}

        Sym1 = Symmetric(S + S')
        Sym2 = Symmetric(2S + 2S')
        @test Zygote.accum(Sym1, Sym2) == Sym1 + Sym2
        @test Zygote.accum(Sym1, Sym2) isa Symmetric{Float64, <:SparseMatrixCSC}
    end
end
