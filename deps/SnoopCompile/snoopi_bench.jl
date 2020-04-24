using SnoopCompile

println("Benchmarking the inference time of `using Zygote`")
snoopi_bench(
  BotConfig("Zygote"; os = ["linux", "windows", "macos"], version = [v"1.4.1", v"1.0.5"]),
  :(using Zygote)
)


println("Benchmarking the inference time of `using Zygote` & basic function test")
snoopi_bench(
  BotConfig("Zygote"; os = ["linux", "windows", "macos"], version = [v"1.4.1", v"1.0.5"]),
  "$(@__DIR__)/example_script.jl",
)
