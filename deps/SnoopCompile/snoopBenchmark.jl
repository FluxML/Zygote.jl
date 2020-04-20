using SnoopCompile

println("Loading benchmark")
@snoopi_bench "Zygote" begin
    using Zygote
end

println("Compiler test benchmark")
@snoopi_bench "Zygote" begin
    using Zygote
    include(joinpath(pkgdir(Zygote),"test","compiler.jl"))
end
