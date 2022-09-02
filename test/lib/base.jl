using LinearAlgebra;

@testset "base.jl" begin
    @testset "Dict getindex with implicit params" begin
        d = Dict{String, Vector{Float64}}("key"=>ones(4))
        fn() = d["key"][2]
        result1 = gradient(fn, Params([d["key"]]))[d["key"]]

        x = d["key"]
        fn2() = x[2]
        result2 = gradient(fn2, Params([x]))[x]

        @test result1 == result2
    end

    @testset "Type stability under broadcast" begin
        g_simple = gradient(a->sum(broadcast(x->x+3,a)),Diagonal([1,2,3]));
        g_hard = gradient((a,b)->sum(broadcast(x->x*b,a)),Diagonal([1,2,3]),4);
        @test first(g_simple) isa Diagonal
        @test first(g_hard) isa Diagonal
    end
end
