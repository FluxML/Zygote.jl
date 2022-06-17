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
end
