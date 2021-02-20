using Zygote: Grads

@testset "Params" begin
  @testset "delete!" begin
    w = rand(2,3)
    b = rand(2)
    ps = Params([w,b])
    delete!(ps, b)
    @test length(ps.order) == length(ps.params) == 1
    @test first(ps.order) == first(ps.params) == w
  end

  @testset "copy!" begin
    x = [0,0]
    ps = Params([x])
    copy!(ps, [1, 2])
    @test x == [1, 2]
    
    x = [0,0]
    y = [0]
    ps = Params([x, y])
    copy!(ps, [1, 2, 3])
    @test x == [1, 2]
    @test y == [3]

    ps = Params([[1,2]])
    x = [0, 0]
    copy!(x, ps)
    @test x == [1, 2]
    
    ps = Params([[1,2], [3]])
    x = [0, 0, 0]
    copy!(x, ps)
    @test x == [1, 2, 3]
  end

  @testset "broadcast" begin
    x, y = [1,2], [1]
    ps = Params([x, y])
    @test length.(ps) == length.([x, y]) # 617
    @test all(Params([[1,1]]) .== Params([[1,1]]))
  end

  @testset "indexing" begin
    x, y = [1,2], [1]
    ps = Params([x, y])
    @test ps[1] === x
    @test ps[2] === y
    @test ps[1:2] == [x, y]
  end

  @testset "comparison" begin
    x, y = [1,2], [1]
    ps1 = Params([x, y])
    ps2 = Params([x, y])
    ps3 = Params([y, x])
    @test ps1 == ps2 
    @test ps1 != ps3  # comparison is order dependent
  end
end

@testset "Grads" begin
  @testset "algebra" begin
    w, b = rand(2), rand(2)
    x1, x2 = rand(2), rand(2)
   
    gs1 = gradient(() -> sum(w .* x1), Params([w])) 
    gs2 = gradient(() -> sum(w .* x2), Params([w])) 
  
    @test .- gs1 isa Grads
    @test gs1 .- gs2 isa Grads 
    @test .+ gs1 isa Grads
    @test gs1 .+ gs2 isa Grads 
    @test 2 .* gs1 isa Grads 
    @test (2 .* gs1)[w] ≈ 2 * gs1[w]
    @test gs1 .* 2 isa Grads 
    @test gs1 ./ 2 isa Grads  
    @test (gs1 .+ gs2)[w] ≈ gs1[w] .+ gs2[w] 

    gs12 = gs1 .+ gs2
    gs1 .+= gs2
    @test gs12[w] ≈ gs1[w] 

    gs3 = gradient(() -> sum(w .* x1), Params([w, b])) # grad nothing with respect to b
    gs4 = gradient(() -> sum(w .* x2 .+ b), Params([w, b])) 

    @test .- gs3 isa Grads
    @test gs3 .- gs4 isa Grads 
    @test .+ gs3 isa Grads
    @test gs3 .+ gs4 isa Grads 
    @test 2 .* gs3 isa Grads 
    @test gs3 .* 2 isa Grads 
    @test gs3 ./ 2 isa Grads  
    @test (gs3 .+ gs4)[w] ≈ gs3[w] .+ gs4[w]
    @test (gs3 .+ gs4)[b] ≈ gs4[b] 
    
    @test gs3 .+ Dict(w => similar(w), b => similar(b)) isa Grads
    gs3 .+= Dict(p => randn(size(p)) for p in keys(gs3))
    @test gs3 isa Grads 
  end

  @testset "map and broadcast" begin
    w = rand(2)
    x1 = rand(2)
    x2 = rand(2)
    
    gs1 = gradient(() -> sum(w .* x1), Params([w])) 
    gs2 = gradient(() -> sum(w .* x2), Params([w])) 
    
    @test map(x -> zeros(2), gs1) isa Grads
    
    gs11 = map(x -> clamp.(x, -1e-5, 1e-5), gs1) 
    @test gs11 isa Grads
    @test all(abs.(gs11[w]) .<= 1e-5) 
  
    @test (x -> zeros(2)).(gs1) isa Grads
  end

  @testset "dictionary interface" begin
    w, b, x = rand(2), rand(2), rand(2)
    ps = Params([w, b])
    gs = gradient(() -> sum(tanh.(w .* x .+ b)), ps) 
    
    @test issetequal(keys(gs), ps) 
    @test length(values(gs)) == 2
    @test length(pairs(gs)) == 2
    k, v = first(pairs(gs))
    @test k === first(ps) 
    @test v === gs[first(ps)]  
  end

  @testset "iteration" begin
    w, b, x = rand(2), rand(2), rand(2)
    ps = Params([w, b])
    gs = gradient(() -> sum(tanh.(w .* x .+ b)), ps) 
    
    # value-based iteration
    foreach(x -> clamp!(x, -1e-5, 1e-5), gs)
    @test all(abs.(gs[w]) .<= 1e-5) 
    @test all(abs.(gs[b]) .<= 1e-5) 
  end
end
