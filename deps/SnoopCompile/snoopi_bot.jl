using SnoopCompile

botconfig = BotConfig(
  "Zygote";
  os = ["linux", "windows", "macos"],
  version = [v"1.4.1", v"1.3.1"],
  blacklist = ["SqEuclidean"],
  exhaustive = false,
)

snoopi_bot(
  botconfig,
  "$(@__DIR__)/example_script.jl",
)
