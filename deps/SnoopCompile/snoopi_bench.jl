using SnoopCompile

println("Benchmarking the inference time of `using Zygote`")
@snoopi_bench BotConfig("Zygote") begin
    using Zygote
end

println("Benchmarking the inference time of `using Zygote` & basic function test")
@snoopi_bench BotConfig("Zygote") begin
    include("example_script.jl")
end
