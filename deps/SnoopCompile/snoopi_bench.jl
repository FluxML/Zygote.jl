using SnoopCompile

println("Benchmarking the inference time of `using Zygote`")
@snoopi_bench BotConfig("Zygote") begin
    using Zygote
end

println("Benchmarking the inference time of `using Zygote` & basic function test")
@snoopi_bench BotConfig("Zygote") begin
    using Zygote
    zygote_rootpath = dirname(dirname(pathof(Zygote)))
    include("$zygote_rootpath/deps/SnoopCompile/example_script.jl")
end
