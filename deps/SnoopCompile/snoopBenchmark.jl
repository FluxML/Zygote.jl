using SnoopCompile
println("loading infer benchmark")

@snoopi_bench "Zygote loading" begin
    using Zygote
end

@snoopi_bench "Zygote compiler tests" begin
    include(joinpath(@__DIR__,"..","..","test","compiler.jl"))
end
