using ZygoteRules: clamptype
using LinearAlgebra

@info "----- starting type clamp tests"

@testset "clamptype" begin

    # Real & Complex
    @test clamptype(Float32, 1+im) === 1
    @test clamptype(ComplexF64, 1+im) === 1+im

    TA = typeof(rand(3))
    @test clamptype(TA, 1:3) === 1:3
    @test clamptype(TA, (1:3) .+ im) isa Vector{Int}

    # Boolean
    # @test clamptype(Bool, 1+im) === nothing
    # TB = typeof(rand(3) .> 0.5)
    # @test clamptype(TB, rand(3)) === nothing
    # @test clamptype(TB, Diagonal(1:3)) === nothing

    # Structured, I
    TD = typeof(Diagonal(1:3))
    @test clamptype(TD, reshape(1:9, 3, 3)) isa Diagonal{Int,<:Vector}
    @test clamptype(TD, Diagonal((1:3) .+ im)) == Diagonal(1:3)
    
    # Structured, II
    TH = typeof(Hermitian(rand(3,3) .+ im))
    TS = typeof(Symmetric(rand(3,3)))
    @test clamptype(TS, reshape(1:4,2,2) .+ im) == [1 2.5; 2.5 4]
    AH = clamptype(TH, reshape(1:4,2,2) .+ im)
    @test AH == [1 2.5; 2.5 4]
    @test AH isa Hermitian{ComplexF64}
    @test clamptype(TH, reshape(1:4,2,2)) isa Hermitian{Float64}

    # Row vectors
    TA = typeof((1:3)')
    TT = typeof(transpose(1:3))
    TC = typeof(adjoint(rand(3) .+ im))
    @test clamptype(TA, permutedims(1:3)) isa LinearAlgebra.AdjOrTransAbsVec
    @test clamptype(TA, ones(1,3) .+ im) isa LinearAlgebra.AdjOrTrans{Float64,<:Vector}
    @test clamptype(TC, ones(1,3) .+ im) == [1+im 1+im 1+im]

     # Tricky
    # TDB = typeof(Diagonal(rand(3) .> 0.5))
    # @test clamptype(TDB, rand(3,3)) === nothing
    # @test clamptype(TDB, rand(ComplexF32, 3,3)) === nothing
    # TAB = typeof(transpose([true, false]))
    # @test clamptype(TAB, rand(3)') === nothing
end

@testset "clamped gradients" begin  # only the marked tests pass on master

    # Real & Complex
    @test gradient(x -> abs2(x+im), 2) == (4,)
    @test gradient(x -> abs2(x+im), 2+0im) == (4 + 2im,)  # as before

    @test gradient(x -> abs2(sum(x .+ im)), [1, 2])[1] == [6, 6]
    @test gradient(x -> abs2(sum(x .+ im)), Any[1, 2])[1] == [6, 6]
    @test gradient(x -> abs2(sum(x .+ im)), [1, 2+0im])[1] == [6 + 4im, 6 + 4im]  # as before

    # Structured, some zeros
    # (if rules improve, these will end up testing them not the projection)
    @test gradient(x -> sum(x .+ 1), Diagonal(rand(3)))[1] == Diagonal([1,1,1])
    @test gradient(x -> sum(sqrt.(x .+ 1)./2), Diagonal(rand(3)))[1] isa Diagonal
    @test gradient(x -> sum(x .+ 1), UpperTriangular(rand(3,3)))[1] == UpperTriangular(ones(3,3))

    ld = gradient((x,y) -> sum(x * y), LowerTriangular(ones(3,3)), Diagonal(ones(3,3)))
    @test ld[1] isa LowerTriangular
    @test_broken ld[2] isa Diagonal

    # Structured, some symmetry
    @test gradient(x -> sum(x .+ 1), Symmetric(rand(3,3)))[1] isa Symmetric
    @test gradient(x -> x[1,2], Symmetric(rand(3,3)))[1] == [0 1/2 0; 1/2 0 0; 0 0 0]

    @test_broken gradient(x -> sum(x * x'), Symmetric(ones(3,3)))[1] isa Symmetric

    # Row vector restoration
    @test pullback(x -> x.+1, rand(3)')[2](ones(1,3))[1] isa LinearAlgebra.AdjOrTransAbsVec
    @test pullback(x -> x.+1, rand(3)')[2]([1 2 3+im])[1] == [1 2 3]
    @test pullback(x -> x.+1, rand(ComplexF64, 3)')[2]([1 2 3+im])[1] == [1 2 3+im]  # as before

    @test gradient(x -> x[1,2], rand(3)')[1] isa LinearAlgebra.AdjOrTransAbsVec  # worked, broken by _zero change
end

@info "----- done type clamp tests"
