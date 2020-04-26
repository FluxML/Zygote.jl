using SnoopCompile

botconfig = BotConfig(
  "Zygote";
  os = ["linux", "windows", "macos"],
  version = [v"1.4.1", v"1.3.1"],
  blacklist = ["SqEuclidean"],
  exhaustive = false,
)


println("Benchmarking the inference time of `using Zygote`")
snoopi_bench(
  botconfig,
  :(using Zygote),
)


println("Benchmarking the inference time of `using Zygote` & basic function test")
snoopi_bench(
  botconfig,
  "$(@__DIR__)/example_script.jl",
)
