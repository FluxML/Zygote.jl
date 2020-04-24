using SnoopCompile

snoopi_bot(
  BotConfig("Zygote"; os = ["linux", "windows", "macos"], version = [v"1.4.1", v"1.0.5"]),
  "$(@__DIR__)/example_script.jl",
)
