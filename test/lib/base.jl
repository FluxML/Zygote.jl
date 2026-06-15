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

    @testset "Type preservation under broadcast" begin
        # https://github.com/FluxML/Zygote.jl/pull/1302
        g_simple = gradient(a->sum(broadcast(x->x+3,a)),Diagonal([1,2,3]));
        g_hard = gradient((a,b)->sum(broadcast(x->x*b,a)),Diagonal([1,2,3]),4);
        @test first(g_simple) isa Diagonal
        @test first(g_hard) isa Diagonal
    end

    @testset "Dict get" begin
        # https://github.com/FluxML/Zygote.jl/issues/1610
        d = Dict(:a => 2.0, :b => 3.0)

        # get(d, k, default)
        @test gradient(d -> get(d, :a, 0.0), d)[1] == Dict(:a => 1.0)
        g = gradient((d, def) -> get(d, :missing, def), d, 10.0)
        @test g[1] === nothing
        @test g[2] == 1.0

        # get(default, d, k) -- the do-block form
        hit = Ref(false)
        @test gradient(d -> get(() -> (hit[] = true; 0.0), d, :a), d)[1] == Dict(:a => 1.0)
        @test hit[] == false  # default is not evaluated when the key is present
        @test gradient(y -> get(() -> 2y, d, :missing), 5.0)[1] == 2.0

        # value-typed (array) gradients route through the dict
        da = Dict(:x => ones(3))
        @test gradient(d -> sum(get(d, :x, zeros(3))), da)[1] == Dict(:x => ones(3))
        @test gradient(d -> sum(get(() -> zeros(3), d, :x)), da)[1] == Dict(:x => ones(3))
    end

    @testset "Dict iteration" begin
        # https://github.com/FluxML/Zygote.jl/issues/1065
        function sumkv(d)
            s = zero(d["c"])
            for (k, v) in d
                s += v
                k == :b && (s += v)
            end
            return sum(s)
        end

        function sumvals(d)
            s = zero(d["c"])
            for v in values(d)
                s += v
            end
            return sum(s)
        end

        d_num = Dict(:a => 3, :b => 4, "c" => 5)
        d_arr = Dict(:a => [3], :b => [4], "c" => [5])
        ps = d_arr |> values |> collect |> Params

        @test gradient(sumkv, d_num)[1] == Dict(:a => 1, :b => 2, "c" => 1)
        grads = gradient(() -> sumkv(d_arr), ps)
        @test (grads[d_arr[:a]], grads[d_arr[:b]], grads[d_arr["c"]]) == ([1], [2], [1])

        @test gradient(sumvals, d_num)[1] == Dict(:a => 1, :b => 1, "c" => 1)
        grads = gradient(() -> sumvals(d_arr), ps)
        @test (grads[d_arr[:a]], grads[d_arr[:b]], grads[d_arr["c"]]) == ([1], [1], [1])
    end
end
