using SnoopCompile

println("Benchmarking `using Zygote`")
@snoopi_bench BotConfig("Zygote") begin
    using Zygote
end

println("Benchmarking `using Zygote` & basic function test")
@snoopi_bench BotConfig("Zygote") begin
    using Zygote
    include(joinpath(pkgdir(Zygote),"src","precompile.jl"))
end
