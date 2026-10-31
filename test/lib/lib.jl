@testset "lib.jl" begin
    @testset "accum" begin
        t1 = (a=1, b=2, c=3)
        t2 = (a=1, b=2)
        @test Zygote.accum(t1, t2) == (a = 2, b = 4, c = 3)
        @test_throws ArgumentError Zygote.accum(t2, t1)
        @test Zygote.accum(fill(0.0), fill(0.0)) == fill(0.0)
    end

    @testset "accum tuples" begin
        accum = Zygote.accum
        @test accum((1, 2), (3, 4)) == (4, 6)
        @test accum((), ()) === ()
        @test accum((1.0, nothing, 3.0), (nothing, 2.0, 4.0)) === (1.0, 2.0, 7.0)
        @test accum((nothing, nothing), (nothing, nothing)) === (nothing, nothing)
        @test accum((1.0, (2.0, nothing)), (nothing, (3.0, 4.0))) === (1.0, (5.0, 4.0))
        @test accum(((a=1.0, b=nothing), 1.0), ((a=2.0, b=3.0), nothing)) === ((a=3.0, b=3.0), 1.0)
        @test accum(([1.0, 2.0], nothing), (nothing, 3.0)) == ([1.0, 2.0], 3.0)
        @test accum((1.0, 2.0)) === (1.0, 2.0)
        # n-ary: left fold of the binary method
        @test accum((1, nothing), (2, 3), (nothing, 4)) === (3, 7)
        # the shorter tuple wins, as with `map`
        @test accum((1, 2, 3), (10, 20)) === (11, 22)
        @test accum((1, 2), (10, 20, 30)) === (11, 22)
        @test accum((1, 2, 3), (10, 20), (100, 200)) === (111, 222)

        # `nothing` arguments are skipped
        @test accum(nothing, nothing, nothing) === nothing
        @test accum(nothing, (1, 2), nothing, (3, 4)) === (4, 6)
        @test accum((1, nothing), nothing, (2, 3), (nothing, 4)) === (3, 7)
        @test accum(nothing, nothing, 1.0, nothing, 2.0) === 3.0
        @test accum(([1.0], nothing), (nothing, (a=1,)), ([2.0], (a=2,))) == ([3.0], (a=3,))

        # long tuples, past the unrolling limit of `map`
        for n in (31, 32, 33, 64)
            x = ntuple(i -> iseven(i) ? nothing : Float64(i), n)
            y = ntuple(i -> i % 3 == 0 ? nothing : Float64(10i), n)
            z = ntuple(i -> Float64(100i), n)
            @test accum(x, y) == map(accum, x, y)
            @test accum(x, y, z) == map(accum, x, y, z)
            @test @inferred(accum(z, z)) == map(a -> 2a, z)
        end
        # n-ary accumulation over many arguments, past the splat limit of the compiler
        for n in (3, 32, 33, 34, 40, 100)
            xs = ntuple(i -> i % 2 == 0 ? nothing : Float64(i), n)
            @test accum(xs...) == sum(something(x, 0.0) for x in xs)
            ts = ntuple(i -> (Float64(i), nothing, i % 2 == 0 ? nothing : 1.0), n)
            @test accum(ts...) == (n * (n + 1) / 2, nothing, Float64(cld(n, 2)))
        end
        # one-hot tuples, as produced by the `getindex` adjoint
        for n in (3, 32, 33, 64)
            ts = ntuple(i -> ntuple(j -> i == j ? Float64(i) : nothing, n), n)
            @test @inferred(accum(ts...)) === ntuple(Float64, n)
            @test accum(ts..., nothing) === ntuple(Float64, n)
            @test accum(ts..., ts...) === ntuple(i -> 2.0 * i, n)
        end
    end

    @testset "gradient through long tuples" begin
        # product of the entries of a tuple, one sequential update per entry
        @generated function chainprod(t::NTuple{N,Float64}) where N
            quote
                r = 1.0
                $([:(r *= t[$i]) for i in 1:N]...)
                r
            end
        end
        for n in (16, 31, 32, 33, 40, 64, 100)
            t = ntuple(i -> 1 + i / 100, n)
            p = prod(t)
            g, = gradient(chainprod, t)
            @test g isa NTuple{n,Float64}
            @test collect(g) ≈ p ./ collect(t)
        end

        # a flat tuple of 64 layers: the backward pass used to be quadratic and box every element
        struct Scale; w::Float64; end
        (l::Scale)(x) = l.w * x
        @generated function applychain(layers::Tuple{Vararg{Any,N}}, x) where N
            quote
                $([:(x = layers[$i](x)) for i in 1:N]...)
                x
            end
        end
        layers = ntuple(i -> Scale(1 + i / 100), 64)
        _, back = pullback(ls -> applychain(ls, 1.0), layers)
        g, = back(1.0)
        @test [l.w for l in g] ≈ prod(l.w for l in layers) ./ [l.w for l in layers]
        allocated(f) = (f(1.0); @allocated f(1.0))
        @test allocated(back) < 100_000  # was ~2.5 MB

        # layers without parameters have `nothing` cotangents
        mixed = ntuple(i -> isodd(i) ? Scale(1 + i / 100) : tanh, 64)
        g, = pullback(ls -> applychain(ls, 0.1), mixed)[2](1.0)
        gref, = gradient(ls -> foldl((x, l) -> l(x), ls; init = 0.1), collect(Any, mixed))
        @test length(g) == 64
        @test all(i -> isodd(i) ? g[i].w ≈ gref[i].w : g[i] === gref[i] === nothing, 1:64)

        @test gradient(t -> t[1] * t[2] + t[3], (1.0, 2.0, 3.0)) == ((2.0, 1.0, 1.0),)
        @test gradient(t -> t[2], (1.0, 2.0, 3.0)) == ((nothing, 1.0, nothing),)
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
