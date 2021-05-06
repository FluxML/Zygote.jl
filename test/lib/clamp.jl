using ZygoteRules: clamptype

@testset "clamptype" begin

    # Real & Complex
    @test clamptype(Float32, 1+im) === 1
    @test clamptype(ComplexF64, 1+im) === 1+im

    TA = typeof(rand(3))
    @test clamptype(TA, 1:3) === 1:3
    @test clamptype(TA, (1:3) .+ im) isa Vector{Int}

    # Boolean
    @test clamptype(Bool, 1+im) === nothing
    TB = typeof(rand(3) .> 0.5)
    @test clamptype(TB, rand(3)) === nothing
    @test clamptype(TB, Diagonal(1:3)) === nothing

    # Structured, II
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

    # Tricky
    TDB = typeof(Diagonal(rand(3) .> 0.5))
    @test clamptype(TDB, rand(3,3)) === nothing
    @test_broken clamptype(TDB, rand(ComplexF32, 3,3)) === nothing

    # Row vectors
    # TA = typeof((1:3)')
    # TT = typeof(transpose(1:3))
    # TC = typeof(adjoint(rand(3) .+ im))
    # @test clamptype(TA, permutedims(1:3)) isa Adjoint
    # @test clamptype(TA, ones(1,3) .+ im) isa Adjoint{Float64,<:Vector}
    # @test clamptype(TC, ones(1,3) .+ im) == [1+im 1+im 1+im]

end
